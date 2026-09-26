"""Pretrained R(2+1)D-18 backbone from torchvision.

Extracts multi-scale features from a pretrained R(2+1)D-18 model.
Used as Phase 1B fallback if custom backbone struggles.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


class R2Plus1D_Backbone(nn.Module):
    """Pretrained R(2+1)D-18 backbone.

    Extracts features from layer2, layer3, layer4.
    Input: (B, 3, T, H, W) with T=64
    Output: F2 (B, 128, T/4, H/8, W/8)
            F3 (B, 256, T/8, H/16, W/16)
            F4 (B, 512, T/16, H/32, W/32)

    Note: R(2+1)D-18 does temporal stride=1 in stem+layer1, then stride=2 in layer2-4.
    With T=64 input: stem→T=32(stride2), layer1→T=32, layer2→T=16, layer3→T=8, layer4→T=4.
    Our neck expects T=32,16,8, so we take layer1,layer2,layer3.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        model = video_models.r2plus1d_18(pretrained=pretrained)

        # stem: (B,3,T,H,W) → (B,64,T/2,H/2,W/2) [temporal stride 1, but actually
        # the default r2plus1d_18 stem has temporal stride=1]
        self.stem = model.stem

        # layer1: (B,64,T/2,H/2,W/2) → (B,64,T/2,H/2,W/2) [no spatial/temporal downsampling]
        self.layer1 = model.layer1

        # layer2: → (B,128,T/4,H/4,W/4)  [stride_s=2, stride_t=2]
        self.layer2 = model.layer2

        # layer3: → (B,256,T/8,H/8,W/8) [stride_s=2, stride_t=2]
        self.layer3 = model.layer3

        # layer4: → (B,512,T/16,H/16,W/16) [stride_s=2, stride_t=2]
        self.layer4 = model.layer4

    def forward(self, x):
        """
        Returns:
            f2: layer2 output
            f3: layer3 output
            f4: layer4 output
        """
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f2, f3, f4
