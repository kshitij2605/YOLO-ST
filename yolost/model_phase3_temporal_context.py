"""YOLO-ST Phase3 with lightweight temporal context gates.

This is a controlled architecture probe toward tube/query-style temporal
reasoning: each FPN scale builds global per-frame tokens, runs a small temporal
Transformer, and uses the resulting context to gate dense features before the
existing boundary head.
"""

import torch
import torch.nn as nn

from .backbone import YOLOST_Backbone
from .head_boundary import DecoupledHeadBoundary
from .neck_pyramid import TemporalPyramidNeck


class TemporalContextGate(nn.Module):
    """Global temporal self-attention gate for one 3D feature map."""

    def __init__(self, channels, d_model=128, num_heads=4, depth=1, dropout=0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(channels)
        self.proj_in = nn.Linear(channels, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_gate = nn.Linear(d_model, channels)
        self.proj_bias = nn.Linear(d_model, channels)

        nn.init.zeros_(self.proj_gate.weight)
        nn.init.zeros_(self.proj_gate.bias)
        nn.init.zeros_(self.proj_bias.weight)
        nn.init.zeros_(self.proj_bias.bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        tokens = x.mean(dim=(-1, -2)).transpose(1, 2)  # (B, T, C)
        ctx = self.encoder(self.proj_in(self.norm_in(tokens)))
        gate = torch.tanh(self.proj_gate(ctx)).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        bias = self.proj_bias(ctx).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gate) + bias


class YOLOST_Phase3TemporalContext(nn.Module):
    """Phase3 model plus temporal context gates before the detection head."""

    def __init__(self, num_classes=24, img_size=224,
                 backbone_channels=None, depths=None, fpn_channels=None,
                 target_T=None, clip_length=64, context_dim=128,
                 context_heads=4, context_depth=1, context_dropout=0.0):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [3, 32, 64, 128, 256, 512, 1024]
        if depths is None:
            depths = [2, 3, 3, 2]
        if fpn_channels is None:
            fpn_channels = [256, 512, 512]
        if target_T is None:
            target_T = [64, 32, 16]

        self.backbone = YOLOST_Backbone(channels=backbone_channels, depths=depths)
        self.neck = TemporalPyramidNeck(
            backbone_channels=backbone_channels,
            fpn_channels=fpn_channels,
            target_T=target_T,
        )
        self.context = nn.ModuleList([
            TemporalContextGate(
                channels=channels,
                d_model=context_dim,
                num_heads=context_heads,
                depth=context_depth,
                dropout=context_dropout,
            )
            for channels in fpn_channels
        ])

        self.spatial_strides = [8, 16, 32]
        self.temporal_strides = [clip_length // t for t in target_T]
        self.head = DecoupledHeadBoundary(
            in_channels_list=fpn_channels,
            num_classes=num_classes,
            strides=self.spatial_strides,
            img_size=img_size,
        )

        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
        features = self.neck(f1, f2, f3, f4)
        features = [gate(feat) for gate, feat in zip(self.context, features)]
        return self.head(features)
