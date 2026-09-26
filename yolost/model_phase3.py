"""YOLO-ST Phase 3 — Temporal Pyramid with Boundary Head.

Based on best Phase 2 config (Dense Pyramid [64,32,16]).
Adds boundary prediction for tube segmentation.
"""

import torch
import torch.nn as nn

from .backbone import YOLOST_Backbone
from .neck_pyramid import TemporalPyramidNeck
from .head_boundary import DecoupledHeadBoundary


class YOLOST_Phase3(nn.Module):
    """YOLO-ST with temporal pyramid + boundary head."""

    def __init__(self, num_classes=24, img_size=224,
                 backbone_channels=None, depths=None, fpn_channels=None,
                 target_T=None, clip_length=64):
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
        self.neck = TemporalPyramidNeck(backbone_channels=backbone_channels,
                                         fpn_channels=fpn_channels,
                                         target_T=target_T)

        self.spatial_strides = [8, 16, 32]
        self.temporal_strides = [clip_length // t for t in target_T]

        self.head = DecoupledHeadBoundary(
            in_channels_list=fpn_channels,
            num_classes=num_classes,
            strides=self.spatial_strides,
            img_size=img_size)

        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        """
        Returns:
            List of (cls, reg, obj, bnd) per scale.
        """
        f1, f2, f3, f4 = self.backbone(x)
        p3, p4, p5 = self.neck(f1, f2, f3, f4)
        return self.head([p3, p4, p5])
