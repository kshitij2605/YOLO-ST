"""YOLO-ST (2+1)D Backbone — matching YOLOST_OLD reference architecture.

Channels: [3, 32, 64, 128, 256, 512, 1024], depths [2, 3, 3, 2].
Outputs F1(T=64, stride 4), F2(T=32, stride 8), F3(T=16, stride 16), F4(T=8, stride 32).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Conv2Plus1D(nn.Module):
    """(2+1)D conv: spatial (1,d,d) → BN+ReLU → temporal (t,1,1) → BN.
    No activation after temporal (matching reference)."""

    def __init__(self, in_ch, out_ch, kernel_size=(3,3,3),
                 stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        t, d, _ = kernel_size
        st, ss, _ = stride
        pt, ps, _ = padding

        mid = int((t * d * d * in_ch * out_ch) / (d * d * in_ch + t * out_ch))
        mid = max(mid, 1)

        self.spatial = nn.Conv3d(in_ch, mid, (1, d, d), (1, ss, ss), (0, ps, ps), bias=False)
        self.bn1 = nn.BatchNorm3d(mid)
        self.relu = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(mid, out_ch, (t, 1, 1), (st, 1, 1), (pt, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial(x)))
        x = self.bn2(self.temporal(x))
        return x


class ConvBNSiLU(nn.Module):
    """(2+1)D Conv + BN + SiLU. Bypasses decomposition for 1×1×1 kernels."""

    def __init__(self, in_ch, out_ch, kernel_size=(3,3,3),
                 stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        if kernel_size == (1,1,1) or kernel_size == [1,1,1]:
            self.conv = nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
            self.bn = nn.BatchNorm3d(out_ch)
            self._pw = True
        else:
            self.conv = Conv2Plus1D(in_ch, out_ch, kernel_size, stride, padding)
            self.bn = None
            self._pw = False
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self._pw:
            return self.act(self.bn(self.conv(x)))
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Bottleneck: Conv1×1 → Conv3×3 with residual (matching reference)."""

    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5):
        super().__init__()
        hid = int(out_ch * expansion)
        self.conv1 = ConvBNSiLU(in_ch, hid, (1,1,1), (1,1,1), (0,0,0))
        self.conv2 = ConvBNSiLU(hid, out_ch, (3,3,3), (1,1,1), (1,1,1))
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.shortcut else out


class C3k2(nn.Module):
    """CSP Bottleneck with (2+1)D convolutions (matching reference)."""

    def __init__(self, in_ch, out_ch, n=2, shortcut=True, expansion=0.5):
        super().__init__()
        hid = int(out_ch * expansion)
        self.conv1 = ConvBNSiLU(in_ch, 2 * hid, (1,1,1), (1,1,1), (0,0,0))
        self.bottlenecks = nn.ModuleList([
            Bottleneck(hid, hid, shortcut=shortcut) for _ in range(n)])
        self.conv2 = ConvBNSiLU(hid * (1 + n), out_ch, (1,1,1), (1,1,1), (0,0,0))

    def forward(self, x):
        x = self.conv1(x)
        y, z = x.chunk(2, dim=1)  # z = passthrough, y = bottleneck path
        outputs = [z]
        for bn in self.bottlenecks:
            y = bn(y)
            outputs.append(y)
        return self.conv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (spatial-only, preserves temporal)."""

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hid = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, hid, (1,1,1), (1,1,1), (0,0,0))
        self.pool = nn.MaxPool3d((1, k, k), stride=1, padding=(0, k//2, k//2))
        self.cv2 = ConvBNSiLU(hid * 4, out_ch, (1,1,1), (1,1,1), (0,0,0))

    def forward(self, x):
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class YOLOST_Backbone(nn.Module):
    """(2+1)D Backbone matching YOLOST_OLD reference.

    Channels: [3, 32, 64, 128, 256, 512, 1024]
    Depths: [2, 3, 3, 2]
    Outputs: F1(T=64), F2(T=32), F3(T=16), F4(T=8)
    """

    def __init__(self, channels=None, depths=None, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if channels is None:
            channels = [3, 32, 64, 128, 256, 512, 1024]
        if depths is None:
            depths = [2, 3, 3, 2]

        # Stem: 3→32, stride_s=2, stride_t=1
        self.stem = ConvBNSiLU(channels[0], channels[1], (3,3,3), (1,2,2), (1,1,1))

        # Stage 1: 32→64→128, stride_s=2, stride_t=1 → F1 at T=64
        self.down1 = ConvBNSiLU(channels[1], channels[2], (3,3,3), (1,2,2), (1,1,1))
        self.stage1 = C3k2(channels[2], channels[3], n=depths[0])

        # Stage 2: 128→256, stride_s=2, stride_t=2 → F2 at T=32
        self.down2 = ConvBNSiLU(channels[3], channels[3], (3,3,3), (2,2,2), (1,1,1))
        self.stage2 = C3k2(channels[3], channels[4], n=depths[1])

        # Stage 3: 256→512, stride_s=2, stride_t=2 → F3 at T=16
        self.down3 = ConvBNSiLU(channels[4], channels[4], (3,3,3), (2,2,2), (1,1,1))
        self.stage3 = C3k2(channels[4], channels[5], n=depths[2])

        # Stage 4: 512→1024 + SPPF, stride_s=2, stride_t=2 → F4 at T=8
        self.down4 = ConvBNSiLU(channels[5], channels[5], (3,3,3), (2,2,2), (1,1,1))
        self.stage4 = C3k2(channels[5], channels[6], n=depths[3])
        self.sppf = SPPF(channels[6], channels[6])

    def _stem_stage1(self, x):
        x = self.stem(x)
        x = self.down1(x)
        return self.stage1(x)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            f1 = checkpoint(self._stem_stage1, x, use_reentrant=False)
        else:
            f1 = self._stem_stage1(x)

        f2 = self.stage2(self.down2(f1))
        f3 = self.stage3(self.down3(f2))
        f4 = self.sppf(self.stage4(self.down4(f3)))
        return f1, f2, f3, f4
