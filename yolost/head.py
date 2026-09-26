"""YOLO-ST Detection Head — spatial-only (1,3,3) convolutions.

Matching YOLOST_OLD reference: temporal context from backbone/neck is
already embedded, so the head operates per-frame independently.
"""

import math
import torch
import torch.nn as nn


class SingleScaleHead(nn.Module):
    """Decoupled head for one FPN scale using spatial-only convolutions."""

    def __init__(self, in_channels, num_classes, num_convs=2):
        super().__init__()
        hidden = max(in_channels // 2, 128)

        # Classification branch — spatial-only (1,3,3) convolutions
        cls_layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else hidden
            cls_layers.extend([
                nn.Conv3d(c_in, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(hidden),
                nn.SiLU(inplace=True),
            ])
        self.cls_convs = nn.Sequential(*cls_layers)
        self.cls_pred = nn.Conv3d(hidden, num_classes, kernel_size=1)

        # Regression branch — spatial-only (1,3,3) convolutions
        reg_layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else hidden
            reg_layers.extend([
                nn.Conv3d(c_in, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(hidden),
                nn.SiLU(inplace=True),
            ])
        self.reg_convs = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv3d(hidden, 4, kernel_size=1)

        # Objectness (from regression features)
        self.obj_pred = nn.Conv3d(hidden, 1, kernel_size=1)

    def bias_init(self, num_classes, stride, img_size=224):
        """Initialize biases for stable early training."""
        S = img_size / stride
        self.cls_pred.bias.data[:] = math.log(5 / num_classes / (S ** 2))
        self.obj_pred.bias.data[:] = math.log(5 / 1 / (S ** 2))
        self.reg_pred.bias.data[:] = 1.0

    def forward(self, x):
        # x: (B, C, T, S, S)
        cls_feat = self.cls_convs(x)
        cls_out = self.cls_pred(cls_feat)   # (B, nc, T, S, S)

        reg_feat = self.reg_convs(x)
        reg_out = self.reg_pred(reg_feat)   # (B, 4, T, S, S)
        obj_out = self.obj_pred(reg_feat)   # (B, 1, T, S, S)

        return cls_out, reg_out, obj_out


class DecoupledHead(nn.Module):
    """Multi-scale detection head."""

    def __init__(self, in_channels_list=None, num_classes=24,
                 strides=(8, 16, 32), img_size=224):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [128, 256, 512]

        self.heads = nn.ModuleList([
            SingleScaleHead(c, num_classes)
            for c in in_channels_list
        ])
        # Initialize biases
        for head, stride in zip(self.heads, strides):
            head.bias_init(num_classes, stride, img_size)

    def forward(self, features):
        return [head(feat) for feat, head in zip(features, self.heads)]
