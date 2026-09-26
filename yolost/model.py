"""YOLO-ST model — matching YOLOST_OLD reference architecture.

Backbone outputs F1-F4, neck upsamples all to T=64, head predicts per-frame.
"""

import torch
import torch.nn as nn

from .backbone import YOLOST_Backbone
from .neck import YOLOST_Neck
from .head import DecoupledHead


class YOLOST(nn.Module):
    """YOLO-ST spatio-temporal action detector."""

    def __init__(self, num_classes=24, img_size=224,
                 backbone_channels=None, depths=None, fpn_channels=None,
                 **kwargs):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [3, 32, 64, 128, 256, 512, 1024]
        if depths is None:
            depths = [2, 3, 3, 2]
        if fpn_channels is None:
            fpn_channels = [256, 512, 512]

        self.backbone = YOLOST_Backbone(channels=backbone_channels, depths=depths)
        self.neck = YOLOST_Neck(backbone_channels=backbone_channels, fpn_channels=fpn_channels)

        self.spatial_strides = [8, 16, 32]
        self.head = DecoupledHead(
            in_channels_list=fpn_channels,
            num_classes=num_classes,
            strides=self.spatial_strides,
            img_size=img_size)

        self.num_classes = num_classes
        self.img_size = img_size
        # All scales at T=64 (temporal stride = 1 from clip frame)
        self.temporal_strides = [1, 1, 1]

    def forward(self, x):
        """
        Args:
            x: (B, 3, T, H, W) video clip, T=64.

        Returns:
            List of (cls, reg, obj) per scale:
                cls: (B, nc, 64, S, S), reg: (B, 4, 64, S, S), obj: (B, 1, 64, S, S)
        """
        f1, f2, f3, f4 = self.backbone(x)
        p3, p4, p5 = self.neck(f1, f2, f3, f4)
        return self.head([p3, p4, p5])

    def decode_predictions(self, outputs, conf_thresh=0.3):
        """Decode to per-frame boxes in normalized [0,1] coordinates."""
        B = outputs[0][0].shape[0]
        all_dets = [[] for _ in range(B)]
        img = self.img_size

        for si, (cls_pred, reg_pred, obj_pred) in enumerate(outputs):
            s_stride = self.spatial_strides[si]
            _, nc, T, S, _ = cls_pred.shape
            step = s_stride / img

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

                cx = (w_idx.float() + torch.sigmoid(box_raw[:, 0])) * step
                cy = (h_idx.float() + torch.sigmoid(box_raw[:, 1])) * step
                w = torch.exp(box_raw[:, 2].clamp(max=5.0)) * step
                h = torch.exp(box_raw[:, 3].clamp(max=5.0)) * step
                boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1).clamp(0, 1)

                all_dets[b].append({
                    'boxes': boxes,
                    'scores': max_conf[b][mask],
                    'labels': max_cls[b][mask],
                    'frames': t_idx.float(),  # T=64, frame index = clip frame
                })

        results = []
        for b in range(B):
            if not all_dets[b]:
                results.append({k: torch.zeros(0, 4) if k == 'boxes' else torch.zeros(0)
                               for k in ('boxes', 'scores', 'labels', 'frames')})
            else:
                results.append({k: torch.cat([d[k] for d in all_dets[b]])
                               for k in ('boxes', 'scores', 'labels', 'frames')})
        return results
