"""YOLO-ST variant with a frozen DINOv3 frame backbone.

This is the first backbone-expansion experiment. DINOv3 is image-native, so
we run it on frames, reshape patch tokens into dense spatial maps, then add a
small trainable temporal projection pyramid before the existing boundary head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .head_boundary import DecoupledHeadBoundary


class ConvBNAct3D(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv3d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TemporalAdapterBlock(nn.Module):
    """Residual temporal mixing block for DINO dense video maps."""

    def __init__(self, channels, expansion=2, kernel_size=5, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        hidden = channels * expansion
        self.dw_temporal = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(channels)
        self.pw1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.pw2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, x):
        y = self.dw_temporal(x)
        y = self.bn(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.pw2(y)
        return x + y


class YOLOST_DINOv3(nn.Module):
    """Frozen DINOv3 frame features + YOLO-ST temporal detection head."""

    def __init__(
        self,
        num_classes=24,
        img_size=224,
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        micro_batch=8,
        dtype="float16",
        layer=-1,
        temporal_adapter=False,
        temporal_adapter_depth=2,
        temporal_adapter_kernel=5,
        temporal_adapter_expansion=2,
        temporal_adapter_dropout=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.micro_batch = micro_batch
        self.layer = layer
        self.temporal_adapter_enabled = temporal_adapter
        self.backbone_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)

        self.backbone = AutoModel.from_pretrained(model_id, dtype=self.backbone_dtype)
        hidden = int(getattr(self.backbone.config, "hidden_size"))
        patch = int(getattr(self.backbone.config, "patch_size", 16))
        if img_size % patch != 0:
            raise ValueError(f"img_size={img_size} must be divisible by DINOv3 patch={patch}")

        self._configure_backbone_training()

        self.patch_size = patch
        self.grid_size = img_size // patch

        # DINOv3 gives one dense map at stride 16 for 224px: 14x14.
        # Build a YOLO-like pyramid:
        # P3: T=64, 28x28, stride 8
        # P4: T=32, 14x14, stride 16
        # P5: T=16, 7x7, stride 32
        self.reduce = ConvBNAct3D(hidden, 256, k=1)
        if temporal_adapter:
            self.temporal_adapter = nn.Sequential(*[
                TemporalAdapterBlock(
                    256,
                    expansion=temporal_adapter_expansion,
                    kernel_size=temporal_adapter_kernel,
                    dropout=temporal_adapter_dropout,
                )
                for _ in range(temporal_adapter_depth)
            ])
        else:
            self.temporal_adapter = nn.Identity()
        self.p3_refine = ConvBNAct3D(256, 256, k=3)
        self.p4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            ConvBNAct3D(256, 512, k=3),
        )
        self.p5 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvBNAct3D(512, 512, k=3),
        )

        self.spatial_strides = [8, 16, 32]
        self.temporal_strides = [1, 2, 4]
        self.head = DecoupledHeadBoundary(
            in_channels_list=[256, 512, 512],
            num_classes=num_classes,
            strides=self.spatial_strides,
            img_size=img_size,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone or self.unfreeze_last_n_blocks <= 0:
            self.backbone.eval()
        return self

    def _configure_backbone_training(self):
        """Freeze DINOv3 by default, optionally unfreezing final ViT blocks."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if self.freeze_backbone or self.unfreeze_last_n_blocks <= 0:
            self.backbone.eval()
            return

        layers = getattr(getattr(self.backbone, "model", None), "layer", None)
        if layers is None:
            raise ValueError("Could not find DINOv3 transformer layers at backbone.model.layer")

        n = min(int(self.unfreeze_last_n_blocks), len(layers))
        for block in layers[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)

        self.backbone.train()

    def _extract_dinov3_maps(self, x):
        """Extract dense DINOv3 maps from video tensor.

        Args:
            x: (B, 3, T, H, W), ImageNet-normalized.
        Returns:
            dense: (B, C, T, Gh, Gw)
        """
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        maps = []
        grad_enabled = (not self.freeze_backbone) and self.training

        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            for start in range(0, frames.shape[0], self.micro_batch):
                batch = frames[start:start + self.micro_batch].to(dtype=self.backbone_dtype)
                out = self.backbone(pixel_values=batch, output_hidden_states=True)
                tokens = out.hidden_states[self.layer]
                gh = h // self.patch_size
                gw = w // self.patch_size
                num_patches = gh * gw
                patch_tokens = tokens[:, -num_patches:, :]
                dense = patch_tokens.transpose(1, 2).reshape(batch.shape[0], -1, gh, gw)
                maps.append(dense.float())

        dense = torch.cat(maps, dim=0).reshape(b, t, -1, h // self.patch_size, w // self.patch_size)
        return dense.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x):
        dense = self._extract_dinov3_maps(x)
        base = self.reduce(dense)
        base = self.temporal_adapter(base)
        p3 = F.interpolate(base, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        p3 = self.p3_refine(p3)
        p4 = self.p4(base)
        p5 = self.p5(p4)
        return self.head([p3, p4, p5])
