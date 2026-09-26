"""BMViT-style MViTv2 token detector.

This model follows the BMViT/STAR research direction more closely than the
BMViT-lite probe: video transformer tokens are the prediction units, not a
YOLO-ST convolutional pyramid.
"""

import torch
import torch.nn as nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3):
        super().__init__()
        layers = []
        for i in range(depth):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == depth - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < depth - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class YOLOST_MViTBMViT(nn.Module):
    """MViTv2-S tokens with BMViT-style actor/action heads."""

    def __init__(
        self,
        num_classes=24,
        img_size=224,
        clip_length=64,
        backbone_frames=16,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        pretrained=True,
        stop_before_final_pool=True,
        hidden_dim=384,
        head_depth=3,
        action_temporal_pool=True,
    ):
        super().__init__()
        if img_size % 16 != 0:
            raise ValueError("BMViT input size must be divisible by 16")
        if clip_length % backbone_frames != 0:
            raise ValueError("clip_length must be divisible by backbone_frames")

        self.num_classes = num_classes
        self.img_size = img_size
        self.clip_length = clip_length
        self.backbone_frames = backbone_frames
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.stop_before_final_pool = stop_before_final_pool
        self.action_temporal_pool = action_temporal_pool

        weights = MViT_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = mvit_v2_s(weights=weights)

        # MViTv2 uses decomposed relative position embeddings, which are
        # interpolated inside attention. Updating the declared token geometry
        # is sufficient for fixed-square 256px inputs and longer clips.
        self.backbone.pos_encoding.temporal_size = backbone_frames // 2
        self.backbone.pos_encoding.spatial_size = (img_size // 4, img_size // 4)

        self.stop_block = 14 if stop_before_final_pool else len(self.backbone.blocks)
        token_dim = 384 if stop_before_final_pool else 768
        self.token_norm = nn.LayerNorm(token_dim)

        self.cls_head = MLP(token_dim, hidden_dim, num_classes, depth=head_depth)
        self.obj_head = MLP(token_dim, hidden_dim, 1, depth=head_depth)
        self.box_head = MLP(token_dim, hidden_dim, 4, depth=head_depth)

        # Removing final pooling gives 8x16x16 tokens for the paper's
        # 16x256x256 recipe. Geometry scales consistently for ablations.
        self.token_t = backbone_frames // 2
        self.grid_size = img_size // (16 if stop_before_final_pool else 32)
        self.spatial_strides = [img_size // self.grid_size]
        self.temporal_strides = [clip_length // self.token_t]

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "kinetics_mean",
            torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "kinetics_std",
            torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1),
            persistent=False,
        )

        self._configure_backbone_training()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone or self.unfreeze_last_n_blocks <= 0:
            self.backbone.eval()
        return self

    def _configure_backbone_training(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if self.freeze_backbone or self.unfreeze_last_n_blocks <= 0:
            self.backbone.eval()
            return

        start = max(0, self.stop_block - int(self.unfreeze_last_n_blocks))
        for block in self.backbone.blocks[start:self.stop_block]:
            for p in block.parameters():
                p.requires_grad_(True)
        self.backbone.train()

    def _sample_backbone_frames(self, x):
        b, c, t, h, w = x.shape
        if t == self.backbone_frames:
            return x
        idx = torch.linspace(0, t - 1, self.backbone_frames, device=x.device).round().long()
        return x.index_select(2, idx)

    def _to_kinetics_norm(self, x):
        # Dataset clips are already ImageNet-normalized. Convert back to [0, 1]
        # and then to the normalization used by torchvision video weights.
        raw = x * self.imagenet_std + self.imagenet_mean
        return (raw - self.kinetics_mean) / self.kinetics_std

    def _extract_tokens(self, x):
        x = self._sample_backbone_frames(x)
        x = self._to_kinetics_norm(x)
        grad_enabled = (not self.freeze_backbone) and self.training
        context = torch.enable_grad() if grad_enabled else torch.no_grad()

        with context:
            x = self.backbone.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.backbone.pos_encoding(x)
            thw = (
                self.backbone.pos_encoding.temporal_size,
                *self.backbone.pos_encoding.spatial_size,
            )
            for block in self.backbone.blocks[:self.stop_block]:
                x, thw = block(x, thw)

        x = self.token_norm(x)
        patch_tokens = x[:, 1:]
        b = patch_tokens.shape[0]
        t, h, w = thw
        if (t, h, w) != (self.token_t, self.grid_size, self.grid_size):
            raise RuntimeError(f"Unexpected MViT token grid: got {(t, h, w)}")
        return patch_tokens.reshape(b, t, h, w, -1)

    def forward(self, x):
        tokens = self._extract_tokens(x)
        if self.action_temporal_pool:
            action_tokens = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
        else:
            action_tokens = tokens

        cls = self.cls_head(action_tokens)
        obj = self.obj_head(tokens)
        reg = self.box_head(tokens)

        cls = cls.permute(0, 4, 1, 2, 3).contiguous()
        obj = obj.permute(0, 4, 1, 2, 3).contiguous()
        reg = reg.permute(0, 4, 1, 2, 3).contiguous()
        return [(cls, reg, obj)]
