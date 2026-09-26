"""YOLO-ST Configurable Temporal Pyramid Neck (Phase 2 — Novelty N2).

Supports arbitrary temporal resolutions per scale via `target_T` parameter:
  - target_T=[32, 16, 8]  — original pyramid (stride 2/4/8)
  - target_T=[64, 16, 8]  — hybrid (P3 at full temporal, P4/P5 reduced)
  - target_T=[64, 32, 16] — dense pyramid (higher temporal everywhere)
  - target_T=[64, 64, 64] — equivalent to uniform (Phase 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConvBNSiLU, C3k2


class TemporalUpsample(nn.Module):
    """Upsample temporal dim by 2x."""
    def __init__(self, ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(ch, ch, (2,1,1), stride=(2,1,1), bias=False)
        self.bn = nn.BatchNorm3d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.up(x)))


class SpatialUpsample(nn.Module):
    """Upsample spatial dims by 2x."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNSiLU(in_ch, out_ch, (1,1,1), (1,1,1), (0,0,0))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        return self.conv(x)


class TemporalDownsample(nn.Module):
    """Downsample temporal dim by a given ratio."""
    def __init__(self, ch, ratio):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, (ratio + 1, 1, 1), stride=(ratio, 1, 1),
                              padding=(ratio // 2, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _build_temporal_adjust_chain(ch, from_T, to_T):
    """Build chain of modules to go from from_T to to_T (up or down)."""
    layers = nn.ModuleList()
    if from_T < to_T:
        # Upsample
        t = from_T
        while t < to_T:
            layers.append(TemporalUpsample(ch))
            t *= 2
        assert t == to_T, f"Cannot upsample from T={from_T} to T={to_T} (must be power-of-2 ratio)"
    elif from_T > to_T:
        # Downsample
        ratio = from_T // to_T
        assert from_T == to_T * ratio, f"Bad temporal adjust: {from_T} -> {to_T}"
        layers.append(TemporalDownsample(ch, ratio))
    # from_T == to_T: empty list (identity)
    return layers


def _build_temporal_downsample(ch, from_T, to_T):
    """Build a single temporal downsample conv from from_T to to_T."""
    ratio = from_T // to_T
    assert ratio >= 1 and from_T == to_T * ratio, f"Bad temporal downsample: {from_T} -> {to_T}"
    if ratio == 1:
        return nn.Identity()
    # Use strided temporal conv; for ratio=2, stride=2; for ratio=4, stride=4, etc.
    return ConvBNSiLU(ch, ch, (ratio + 1, 1, 1), (ratio, 1, 1), (ratio // 2, 0, 0))


class TemporalPyramidNeck(nn.Module):
    """Configurable Temporal Pyramid PANet Neck.

    Args:
        backbone_channels: [3, 32, 64, 128, 256, 512, 1024]
        fpn_channels: [256, 512, 512] for P3, P4, P5
        target_T: Target temporal resolution per scale [T_P3, T_P4, T_P5].
                  Default [32, 16, 8] = original pyramid.
    """

    def __init__(self, backbone_channels=None, fpn_channels=None, target_T=None):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [3, 32, 64, 128, 256, 512, 1024]
        if fpn_channels is None:
            fpn_channels = [256, 512, 512]
        if target_T is None:
            target_T = [32, 16, 8]

        self.target_T = target_T

        bk4 = backbone_channels[4]  # F2: 256
        bk5 = backbone_channels[5]  # F3: 512
        bk6 = backbone_channels[6]  # F4: 1024
        fp0, fp1, fp2 = fpn_channels

        # Backbone output temporal resolutions: F2@T=32, F3@T=16, F4@T=8

        # --- Top-down pathway ---
        # F4 (T=8) → reduce channels
        self.reduce_f4 = ConvBNSiLU(bk6, fp2, (1,1,1), (1,1,1), (0,0,0))

        # Upsample F4 to match F3 (spatial 2x + temporal 8→16)
        self.up_f4_spatial = SpatialUpsample(fp2, fp2)
        self.up_f4_temporal = TemporalUpsample(fp2)  # T:8→16
        self.fuse_f4f3 = C3k2(fp2 + bk5, fp1, n=2, shortcut=False)

        # Upsample fused F3 to match F2 (spatial 2x + temporal 16→32)
        self.up_f3_spatial = SpatialUpsample(fp1, fp0)
        self.up_f3_temporal = TemporalUpsample(fp0)  # T:16→32
        self.fuse_f3f2 = C3k2(fp0 + bk4, fp0, n=2, shortcut=False)

        # After top-down: td2@T=32, td3@T=16, td4@T=8

        # --- Temporal adjust to target resolutions ---
        # P3: td2 (T=32) → target_T[0]
        self.p3_tup = _build_temporal_adjust_chain(fp0, 32, target_T[0])

        # P4: td3 (T=16) → target_T[1]
        self.p4_tup = _build_temporal_adjust_chain(fp1, 16, target_T[1])

        # P5: td4 (T=8) → target_T[2]
        self.p5_tup = _build_temporal_adjust_chain(fp2, 8, target_T[2])

        # --- Bottom-up pathway (all at their target temporal resolutions) ---
        # P3 → downsample spatial + temporal to match P4
        self.down_p3_spatial = ConvBNSiLU(fp0, fp0, (3,3,3), (1,2,2), (1,1,1))
        self.down_p3_temporal = _build_temporal_downsample(fp0, target_T[0], target_T[1])
        self.fuse_p3p4 = C3k2(fp0 + fp1, fp1, n=2, shortcut=False)

        # P4 → downsample spatial + temporal to match P5
        self.down_p4_spatial = ConvBNSiLU(fp1, fp1, (3,3,3), (1,2,2), (1,1,1))
        self.down_p4_temporal = _build_temporal_downsample(fp1, target_T[1], target_T[2])
        self.fuse_p4p5 = C3k2(fp1 + fp2, fp2, n=2, shortcut=False)

    def forward(self, f1, f2, f3, f4):
        # Top-down
        td4 = self.reduce_f4(f4)
        up4 = self.up_f4_temporal(self.up_f4_spatial(td4))
        td3 = self.fuse_f4f3(torch.cat([up4, f3], dim=1))

        up3 = self.up_f3_temporal(self.up_f3_spatial(td3))
        td2 = self.fuse_f3f2(torch.cat([up3, f2], dim=1))

        # Temporal upsample to target resolutions
        p3 = td2
        for tup in self.p3_tup:
            p3 = tup(p3)

        p4_pre = td3
        for tup in self.p4_tup:
            p4_pre = tup(p4_pre)

        p5_pre = td4
        for tup in self.p5_tup:
            p5_pre = tup(p5_pre)

        # Bottom-up
        down_p3 = self.down_p3_temporal(self.down_p3_spatial(p3))
        p4 = self.fuse_p3p4(torch.cat([down_p3, p4_pre], dim=1))

        down_p4 = self.down_p4_temporal(self.down_p4_spatial(p4))
        p5 = self.fuse_p4p5(torch.cat([down_p4, p5_pre], dim=1))

        return p3, p4, p5
