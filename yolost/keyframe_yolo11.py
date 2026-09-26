"""Frozen YOLO11 frame pyramid for optional VideoMAE spatial fusion."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class YOLO11KeyframePyramid(nn.Module):
    """Extract high-to-low-resolution P3/P4/P5 maps from sampled frames."""

    _CHANNELS = {
        "n": (64, 128, 256),
        "s": (128, 256, 512),
        "m": (192, 384, 768),
        "l": (256, 512, 512),
        "x": (320, 640, 1280),
    }

    def __init__(self, version="n", pretrained_path=None, frames=16,
                 micro_batch=16, freeze=True, bn_eval=True,
                 grad_checkpoint=False):
        super().__init__()
        version = str(version).lower()
        if version not in self._CHANNELS:
            raise ValueError(f"Unsupported YOLO11 version: {version}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required when model.keyframe_pyramid is enabled"
            ) from exc

        model_path = pretrained_path or f"yolo11{version}.pt"
        detector = YOLO(model_path)
        self.layers = nn.ModuleList(list(detector.model.model.children())[:17])
        self.version = version
        self.out_channels = self._CHANNELS[version]
        self.frames = int(frames)
        self.micro_batch = max(1, int(micro_batch))
        self.freeze_backbone = bool(freeze)
        # A trainable pyramid keeps BatchNorm in eval mode: statistics from a
        # few correlated frames per clip would drift from the COCO values the
        # convolutions were trained with.
        self.bn_eval = bool(bn_eval)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        if self.freeze_backbone:
            for parameter in self.layers.parameters():
                parameter.requires_grad_(False)
            self.layers.eval()
        else:
            # Ultralytics checkpoints load with requires_grad False.
            for parameter in self.layers.parameters():
                parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.layers.eval()
        elif self.bn_eval:
            for module in self.layers.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def _forward_frames(self, frames):
        intermediates = {}
        features = {}
        value = frames
        for index, layer in enumerate(self.layers):
            if index == 12:
                value = layer([value, intermediates[6]])
            elif index == 15:
                value = layer([value, intermediates[4]])
            else:
                value = layer(value)
            if index in (4, 6, 10):
                intermediates[index] = value
            if index in (10, 13, 16):
                features[index] = value
        return features[16], features[13], features[10]

    def forward(self, clip):
        batch, _, time, _, _ = clip.shape
        sample_count = min(self.frames, time)
        indices = torch.linspace(
            0, time - 1, sample_count, device=clip.device
        ).round().long()
        frames = clip.index_select(2, indices).permute(0, 2, 1, 3, 4)
        frames = frames.reshape(batch * sample_count, *frames.shape[2:])
        frames = (frames * self.image_std + self.image_mean).clamp(0.0, 1.0)

        outputs = [[], [], []]
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            for start in range(0, frames.shape[0], self.micro_batch):
                chunk = frames[start:start + self.micro_batch]
                if (self.grad_checkpoint and self.training
                        and not self.freeze_backbone and torch.is_grad_enabled()):
                    chunk_outputs = checkpoint(
                        self._forward_frames, chunk, use_reentrant=False
                    )
                else:
                    chunk_outputs = self._forward_frames(chunk)
                for level, feature in enumerate(chunk_outputs):
                    outputs[level].append(feature)

        pyramid = []
        for level_outputs in outputs:
            feature = torch.cat(level_outputs, dim=0)
            feature = feature.reshape(
                batch, sample_count, *feature.shape[1:]
            ).permute(0, 2, 1, 3, 4).contiguous()
            pyramid.append(feature.float())
        return pyramid
