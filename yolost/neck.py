"""YOLO-ST Neck — matching YOLOST_OLD reference.

Takes F1(T=64), F2(T=32), F3(T=16), F4(T=8) from backbone.
Outputs P3(T=64), P4(T=64), P5(T=64) for per-frame detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNSiLU, C3k2


class TemporalUpsample(nn.Module):
    """Upsample temporal dim by 2x via transposed conv."""
    def __init__(self, ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(ch, ch, (2,1,1), stride=(2,1,1), bias=False)
        self.bn = nn.BatchNorm3d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.up(x)))


class SpatialUpsample(nn.Module):
    """Upsample spatial dims by 2x via nearest + 1×1×1 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNSiLU(in_ch, out_ch, (1,1,1), (1,1,1), (0,0,0))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        return self.conv(x)


class YOLOST_Neck(nn.Module):
    """PANet neck matching YOLOST_OLD reference.

    All outputs at T=64 for per-frame detection.
    """

    def __init__(self, backbone_channels=None, fpn_channels=None, **kwargs):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [3, 32, 64, 128, 256, 512, 1024]
        if fpn_channels is None:
            fpn_channels = [256, 512, 512]

        bk4 = backbone_channels[4]  # F2: 256
        bk5 = backbone_channels[5]  # F3: 512
        bk6 = backbone_channels[6]  # F4: 1024
        fp0, fp1, fp2 = fpn_channels  # P3:256, P4:512, P5:512

        # --- Top-down ---
        self.reduce_f4 = ConvBNSiLU(bk6, fp2, (1,1,1), (1,1,1), (0,0,0))

        self.up_f4_spatial = SpatialUpsample(fp2, fp2)
        self.up_f4_temporal = TemporalUpsample(fp2)  # T:8→16
        self.fuse_f4f3 = C3k2(fp2 + bk5, fp1, n=2, shortcut=False)

        self.up_f3_spatial = SpatialUpsample(fp1, fp0)
        self.up_f3_temporal = TemporalUpsample(fp0)  # T:16→32
        self.fuse_f3f2 = C3k2(fp0 + bk4, fp0, n=2, shortcut=False)

        # --- Temporal upsample to T=64 ---
        self.p3_tup = TemporalUpsample(fp0)     # 32→64

        self.p4_tup1 = TemporalUpsample(fp1)    # 16→32
        self.p4_tup2 = TemporalUpsample(fp1)    # 32→64

        self.p5_tup1 = TemporalUpsample(fp2)    # 8→16
        self.p5_tup2 = TemporalUpsample(fp2)    # 16→32
        self.p5_tup3 = TemporalUpsample(fp2)    # 32→64

        # --- Bottom-up (all at T=64, spatial-only stride) ---
        self.down_p3 = ConvBNSiLU(fp0, fp0, (3,3,3), (1,2,2), (1,1,1))
        self.fuse_p3p4 = C3k2(fp0 + fp1, fp1, n=2, shortcut=False)

        self.down_p4 = ConvBNSiLU(fp1, fp1, (3,3,3), (1,2,2), (1,1,1))
        self.fuse_p4p5 = C3k2(fp1 + fp2, fp2, n=2, shortcut=False)

    def forward(self, f1, f2, f3, f4):
        # Top-down
        td4 = self.reduce_f4(f4)
        up4 = self.up_f4_temporal(self.up_f4_spatial(td4))
        td3 = self.fuse_f4f3(torch.cat([up4, f3], dim=1))

        up3 = self.up_f3_temporal(self.up_f3_spatial(td3))
        td2 = self.fuse_f3f2(torch.cat([up3, f2], dim=1))

        # Temporal upsample to T=64
        p3 = self.p3_tup(td2)
        p4_pre = self.p4_tup2(self.p4_tup1(td3))
        p5_pre = self.p5_tup3(self.p5_tup2(self.p5_tup1(td4)))

        # Bottom-up (all at T=64)
        p4 = self.fuse_p3p4(torch.cat([self.down_p3(p3), p4_pre], dim=1))
        p5 = self.fuse_p4p5(torch.cat([self.down_p4(p4), p5_pre], dim=1))

        return p3, p4, p5
