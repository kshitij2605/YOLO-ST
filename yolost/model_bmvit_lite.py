"""BMViT-lite probe model.

This keeps the existing YOLO-ST backbone and temporal pyramid so evaluation can
reuse the current dense decoder, but trains it with a one-to-one bipartite
matching objective instead of the TAL/grid assignment. It is a controlled first
step toward BMViT/STAR-style token prediction.
"""

import torch.nn as nn

from .backbone import YOLOST_Backbone
from .head import DecoupledHead
from .neck_pyramid import TemporalPyramidNeck


class YOLOST_BMViTLite(nn.Module):
    """YOLO-ST temporal pyramid with non-boundary dense heads."""

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
        self.neck = TemporalPyramidNeck(
            backbone_channels=backbone_channels,
            fpn_channels=fpn_channels,
            target_T=target_T,
        )
        self.head = DecoupledHead(
            in_channels_list=fpn_channels,
            num_classes=num_classes,
            strides=(8, 16, 32),
            img_size=img_size,
        )

        self.spatial_strides = [8, 16, 32]
        self.temporal_strides = [clip_length // t for t in target_T]
        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)
        features = self.neck(f1, f2, f3, f4)
        return self.head(features)
