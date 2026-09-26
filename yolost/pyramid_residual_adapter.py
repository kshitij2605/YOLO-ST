"""Parent-preserving residual adapters for dense temporal-pyramid logits."""

import math

import torch
import torch.nn as nn


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class FactorizedPyramidResidual(nn.Module):
    """Predict zero-initialized logit corrections from one pyramid level."""

    def __init__(self, in_channels, num_classes, hidden_ratio=0.125,
                 temporal_kernel=3, class_residual=True, box_residual=True,
                 object_residual=True):
        super().__init__()
        temporal_kernel = int(temporal_kernel)
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        if hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be positive")
        if not any((class_residual, box_residual, object_residual)):
            raise ValueError("at least one residual output must be enabled")

        hidden = max(32, int(math.ceil(in_channels * float(hidden_ratio) / 8) * 8))
        self.class_residual = bool(class_residual)
        self.box_residual = bool(box_residual)
        self.object_residual = bool(object_residual)
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                groups=hidden, bias=False,
            ),
            nn.Conv3d(
                hidden, hidden, kernel_size=(temporal_kernel, 1, 1),
                padding=(temporal_kernel // 2, 0, 0), groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(inplace=True),
        )
        self.class_projection = (
            nn.Conv3d(hidden, num_classes, kernel_size=1)
            if self.class_residual else None
        )
        self.box_projection = (
            nn.Conv3d(hidden, 4, kernel_size=1)
            if self.box_residual else None
        )
        self.object_projection = (
            nn.Conv3d(hidden, 1, kernel_size=1)
            if self.object_residual else None
        )
        for projection in (
            self.class_projection, self.box_projection, self.object_projection
        ):
            if projection is not None:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def forward(self, feature, output):
        if len(output) < 3:
            raise ValueError("dense output requires class, box, and object logits")
        class_logits, box_logits, object_logits, *rest = output
        residual = self.body(feature.detach())
        if self.class_projection is not None:
            class_logits = class_logits + self.class_projection(residual)
        if self.box_projection is not None:
            box_logits = box_logits + self.box_projection(residual)
        if self.object_projection is not None:
            object_logits = object_logits + self.object_projection(residual)
        return (class_logits, box_logits, object_logits, *rest)


class PyramidAlignedResidualAdapters(nn.Module):
    """Apply independent residual corrections at P3, P4, and P5."""

    def __init__(self, in_channels, num_classes, hidden_ratio=0.125,
                 temporal_kernels=(3, 3, 3), scales=(True, True, True),
                 class_residual=True, box_residual=True,
                 object_residual=True):
        super().__init__()
        if len(in_channels) != 3 or len(temporal_kernels) != 3 or len(scales) != 3:
            raise ValueError("pyramid residual settings must contain P3/P4/P5 values")
        self.enabled_scales = tuple(bool(value) for value in scales)
        if not any(self.enabled_scales):
            raise ValueError("at least one pyramid residual scale must be enabled")
        self.adapters = nn.ModuleList([
            FactorizedPyramidResidual(
                channels,
                num_classes,
                hidden_ratio=hidden_ratio,
                temporal_kernel=kernel,
                class_residual=class_residual,
                box_residual=box_residual,
                object_residual=object_residual,
            ) if enabled else nn.Identity()
            for channels, kernel, enabled in zip(
                in_channels, temporal_kernels, self.enabled_scales
            )
        ])

    def forward(self, features, outputs):
        if len(features) != 3 or len(outputs) != 3:
            raise ValueError("pyramid residual adapter expects P3/P4/P5 tensors")
        return [
            adapter(feature, output) if enabled else output
            for adapter, enabled, feature, output in zip(
                self.adapters, self.enabled_scales, features, outputs
            )
        ]
