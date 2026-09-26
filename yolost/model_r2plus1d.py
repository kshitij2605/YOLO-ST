"""YOLO-ST with pretrained R(2+1)D-18 backbone from torchvision.

The R(2+1)D-18 backbone produces 2x larger spatial features than our custom
backbone (stride 4,8,16 vs 8,16,32). This gives more detection cells but
requires adjusting the spatial strides accordingly.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models

from .neck import YOLOST_Neck
from .head import DecoupledHead


class R2Plus1D_Backbone(nn.Module):
    """Pretrained R(2+1)D-18 backbone."""

    def __init__(self, pretrained=True):
        super().__init__()
        model = video_models.r2plus1d_18(
            weights='KINETICS400_V1' if pretrained else None)
        self.stem = model.stem
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.stem(x)       # (B, 64, 32, 112, 112)
        x = self.layer1(x)     # (B, 64, 32, 56, 56)
        f2 = self.layer2(x)    # (B, 128, 16, 28, 28)
        f3 = self.layer3(f2)   # (B, 256, 8, 14, 14)
        f4 = self.layer4(f3)   # (B, 512, 4, 7, 7)
        return f2, f3, f4


class YOLOST_R2Plus1D(nn.Module):
    """YOLO-ST with pretrained R(2+1)D-18 backbone.

    Backbone temporal strides: stem=2, layer2=2, layer3=2, layer4=2
    So: f2@T=16, f3@T=8, f4@T=4

    Spatial strides: stem=4(2+2), layer2=2, layer3=2, layer4=2
    So: f2@S/8, f3@S/16, f4@S/32 (same spatial strides as custom backbone!)
    """

    def __init__(self, num_classes=24, img_size=224, pretrained=True):
        super().__init__()

        self.backbone = R2Plus1D_Backbone(pretrained=pretrained)

        # f2=128ch, f3=256ch, f4=512ch — same channel dims as custom backbone
        neck_in = [128, 256, 512]
        neck_out = [128, 256, 512]
        # Uniform mode: upsample all to same T as f2 (T=16)
        self.neck = YOLOST_Neck(in_channels=neck_in, out_channels=neck_out,
                                uniform=True)

        self.spatial_strides = [8, 16, 32]
        self.head = DecoupledHead(in_channels_list=neck_out,
                                  num_classes=num_classes,
                                  strides=self.spatial_strides,
                                  img_size=img_size)

        self.num_classes = num_classes
        self.img_size = img_size
        # R2+1D: f2@T=16 from T=64 input → temporal stride = 4
        # Uniform mode ups everything to T=16
        self.temporal_strides = [4, 4, 4]

    def forward(self, x):
        f2, f3, f4 = self.backbone(x)
        p3, p4, p5 = self.neck(f2, f3, f4)
        return self.head([p3, p4, p5])

    def decode_predictions(self, outputs, conf_thresh=0.3):
        """Same as YOLOST.decode_predictions."""
        batch_size = outputs[0][0].shape[0]
        all_dets = [[] for _ in range(batch_size)]
        img = self.img_size

        for si, (cls_pred, reg_pred, obj_pred) in enumerate(outputs):
            t_stride = self.temporal_strides[si]
            s_stride = self.spatial_strides[si]
            B, nc, T, S, _ = cls_pred.shape

            obj_sig = torch.sigmoid(obj_pred[:, 0])
            cls_sig = torch.sigmoid(cls_pred)
            cls_perm = cls_sig.permute(0, 2, 3, 4, 1)
            combined = obj_sig.unsqueeze(-1) * cls_perm
            max_conf, max_cls = combined.max(dim=-1)

            for b in range(B):
                mask = max_conf[b] > conf_thresh
                if not mask.any():
                    continue
                t_idx, h_idx, w_idx = torch.where(mask)
                box_raw = reg_pred[b, :, t_idx, h_idx, w_idx].T

                step = s_stride / img
                cx = (w_idx.float() + torch.sigmoid(box_raw[:, 0])) * step
                cy = (h_idx.float() + torch.sigmoid(box_raw[:, 1])) * step
                w = torch.exp(box_raw[:, 2].clamp(max=5.0)) * step
                h = torch.exp(box_raw[:, 3].clamp(max=5.0)) * step
                boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], -1)
                frames = t_idx.float() * t_stride

                all_dets[b].append({
                    'boxes': boxes,
                    'scores': max_conf[b][mask],
                    'labels': max_cls[b][mask],
                    'frames': frames,
                })

        results = []
        for b in range(batch_size):
            if not all_dets[b]:
                results.append({k: torch.zeros(0, 4) if k == 'boxes' else torch.zeros(0)
                               for k in ('boxes', 'scores', 'labels', 'frames')})
            else:
                results.append({k: torch.cat([d[k] for d in all_dets[b]])
                               for k in ('boxes', 'scores', 'labels', 'frames')})
        return results
