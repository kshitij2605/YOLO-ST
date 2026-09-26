"""YOLO-ST Detection Head with Boundary Prediction (Phase 3A).

Adds a boundary output (1 channel) from classification features.
Predicts whether a detection cell is at an action boundary frame.
"""

import math
import torch
import torch.nn as nn

from .dfl import regression_channels


class SingleScaleHeadBoundary(nn.Module):
    """Decoupled head with boundary output for one FPN scale."""

    def __init__(self, in_channels, num_classes, num_convs=2,
                 reg_max=0):
        super().__init__()
        hidden = max(in_channels // 2, 128)

        # Classification branch
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

        # Boundary branch (from cls features)
        self.bnd_pred = nn.Conv3d(hidden, 1, kernel_size=1)

        # Regression branch
        reg_layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else hidden
            reg_layers.extend([
                nn.Conv3d(c_in, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(hidden),
                nn.SiLU(inplace=True),
            ])
        self.reg_convs = nn.Sequential(*reg_layers)
        self.reg_max = int(reg_max)
        self.reg_pred = nn.Conv3d(
            hidden, regression_channels(self.reg_max), kernel_size=1
        )

        # Objectness (from regression features)
        self.obj_pred = nn.Conv3d(hidden, 1, kernel_size=1)

    def bias_init(self, num_classes, stride, img_size=224):
        from .geometry import image_hw
        img_h, img_w = image_hw(img_size)
        if img_h == img_w:
            cells = (img_h / stride) ** 2
        else:
            cells = (img_h / stride) * (img_w / stride)
        self.cls_pred.bias.data[:] = math.log(5 / num_classes / cells)
        self.obj_pred.bias.data[:] = math.log(5 / 1 / cells)
        self.reg_pred.bias.data[:] = 1.0
        # Boundary: most frames are NOT boundaries, so bias negative
        self.bnd_pred.bias.data[:] = -2.0

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        cls_out = self.cls_pred(cls_feat)
        bnd_out = self.bnd_pred(cls_feat)

        reg_feat = self.reg_convs(x)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)

        return cls_out, reg_out, obj_out, bnd_out


class DecoupledHeadBoundary(nn.Module):
    """Multi-scale detection head with boundary prediction."""

    def __init__(self, in_channels_list=None, num_classes=24,
                 strides=(8, 16, 32), img_size=224, reg_max=0):
        super().__init__()
        if in_channels_list is None:
            in_channels_list = [128, 256, 512]

        self.reg_max = int(reg_max)
        self.heads = nn.ModuleList([
            SingleScaleHeadBoundary(c, num_classes,
                                    reg_max=self.reg_max)
            for c in in_channels_list
        ])
        for head, stride in zip(self.heads, strides):
            head.bias_init(num_classes, stride, img_size)

    def forward(self, features):
        return [head(feat) for feat, head in zip(features, self.heads)]
