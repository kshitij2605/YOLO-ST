"""YOLO-ST with Temporal Pyramid Neck (Phase 2 — Novelty N2).

Same backbone as Phase 1, but neck preserves native temporal resolutions:
  P3 @ T=32 (temporal stride 2) — small actors, fast actions
  P4 @ T=16 (temporal stride 4) — medium actors/actions
  P5 @ T=8  (temporal stride 8) — large actors, slow actions
"""

import torch
import torch.nn as nn

from .backbone import YOLOST_Backbone
from .neck_pyramid import TemporalPyramidNeck
from .head import DecoupledHead


class YOLOST_Pyramid(nn.Module):
    """YOLO-ST with temporal pyramid detection."""

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
            target_T = [32, 16, 8]

        self.backbone = YOLOST_Backbone(channels=backbone_channels, depths=depths)
        self.neck = TemporalPyramidNeck(backbone_channels=backbone_channels,
                                         fpn_channels=fpn_channels,
                                         target_T=target_T)

        self.spatial_strides = [8, 16, 32]
        # Temporal stride = clip_length / target_T for each scale
        self.temporal_strides = [clip_length // t for t in target_T]

        self.head = DecoupledHead(
            in_channels_list=fpn_channels,
            num_classes=num_classes,
            strides=self.spatial_strides,
            img_size=img_size)

        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        """
        Returns:
            List of (cls, reg, obj) per scale:
                P3: (B, C, 32, S3, S3)
                P4: (B, C, 16, S4, S4)
                P5: (B, C, 8, S5, S5)
        """
        f1, f2, f3, f4 = self.backbone(x)
        p3, p4, p5 = self.neck(f1, f2, f3, f4)
        return self.head([p3, p4, p5])

    def decode_predictions(self, outputs, conf_thresh=0.3):
        """Decode to per-frame boxes. Each scale maps to different clip frames."""
        B = outputs[0][0].shape[0]
        all_dets = [[] for _ in range(B)]
        img = self.img_size

        for si, (cls_pred, reg_pred, obj_pred) in enumerate(outputs):
            s_stride = self.spatial_strides[si]
            t_stride = self.temporal_strides[si]
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

                # Map detection frame to clip frame
                frames = t_idx.float() * t_stride

                all_dets[b].append({
                    'boxes': boxes,
                    'scores': max_conf[b][mask],
                    'labels': max_cls[b][mask],
                    'frames': frames,
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
