"""YOLO-ST Loss with Boundary Head (Phase 3A).

Extends YOLOSTLoss with boundary prediction loss.
Boundary target: 1 if GT frame is within margin of tube start/end.
"""

import torch
import torch.nn as nn

from .loss import YOLOSTLoss


class YOLOSTLossBoundary(YOLOSTLoss):
    """Loss with additional boundary prediction term."""

    def __init__(self, num_classes=24, lambda_cls=0.5, lambda_box=7.5,
                 lambda_obj=1.5, lambda_bnd=0.5, lambda_tube_cls=0.0,
                 lambda_tube_box=0.0, tube_obj_power=0.5, img_size=224):
        super().__init__(num_classes, lambda_cls, lambda_box, lambda_obj, img_size)
        self.lambda_bnd = lambda_bnd
        self.lambda_tube_cls = lambda_tube_cls
        self.lambda_tube_box = lambda_tube_box
        self.tube_obj_power = tube_obj_power
        self.bce_bnd = nn.BCEWithLogitsLoss(reduction='none')

    def _preprocess_targets_with_boundary(self, targets, B, T_det, device, t_stride=2):
        """Same as _preprocess_targets but also returns boundary flags per GT box."""
        all_labels = []
        all_boxes = []
        all_boundary = []
        all_counts = []

        for b in range(B):
            boxes_b = targets['boxes'][b]
            labels_b = targets['labels'][b]
            boundary_b = targets.get('boundary')
            if boundary_b is not None:
                boundary_b = boundary_b[b]

            for t in range(T_det):
                if boxes_b.numel() == 0:
                    if targets['labels'].ndim == 3:
                        all_labels.append(torch.zeros(0, self.nc, device=device))
                    else:
                        all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                    all_boxes.append(torch.zeros(0, 4, device=device))
                    all_boundary.append(torch.zeros(0, device=device))
                    all_counts.append(0)
                else:
                    det_frame = boxes_b[:, 0].long() // t_stride
                    frame_mask = (det_frame == t) & (boxes_b.sum(dim=-1) > 0)
                    gt_b = boxes_b[frame_mask, 1:5]
                    gt_l = labels_b[frame_mask]
                    if boundary_b is not None:
                        gt_bnd = boundary_b[frame_mask].float()
                    else:
                        gt_bnd = torch.zeros(frame_mask.sum(), device=device)
                    all_labels.append(gt_l.to(device))
                    all_boxes.append(gt_b.to(device))
                    all_boundary.append(gt_bnd.to(device))
                    all_counts.append(len(gt_l))

        BT = B * T_det
        max_gt = max(all_counts) if all_counts else 0
        multilabel = targets['labels'].ndim == 3
        label_shape = (BT, 0, self.nc) if multilabel else (BT, 0, 1)
        if max_gt == 0:
            return (torch.zeros(label_shape, dtype=torch.float32 if multilabel else torch.long, device=device),
                    torch.zeros(BT, 0, 4, device=device),
                    torch.zeros(BT, 0, 1, device=device),
                    torch.zeros(BT, 0, device=device))

        if multilabel:
            gt_labels = torch.zeros(BT, max_gt, self.nc, dtype=torch.float32, device=device)
        else:
            gt_labels = torch.zeros(BT, max_gt, 1, dtype=torch.long, device=device)
        gt_boxes = torch.zeros(BT, max_gt, 4, device=device)
        mask_gt = torch.zeros(BT, max_gt, 1, device=device)
        gt_boundary = torch.zeros(BT, max_gt, device=device)

        for i in range(BT):
            n = all_counts[i]
            if n > 0:
                if multilabel:
                    gt_labels[i, :n] = all_labels[i].float()
                else:
                    gt_labels[i, :n, 0] = all_labels[i]
                gt_boxes[i, :n] = all_boxes[i]
                mask_gt[i, :n, 0] = 1.0
                gt_boundary[i, :n] = all_boundary[i]

        return gt_labels, gt_boxes, mask_gt, gt_boundary

    def forward(self, predictions, targets, temporal_strides=None,
                spatial_strides=None, img_size=None):
        """
        Args:
            predictions: list of (cls, reg, obj, bnd) per scale.
        """
        if img_size is not None:
            self.img_size = img_size

        device = predictions[0][0].device
        B = predictions[0][0].shape[0]
        s_strides = spatial_strides or [8, 16, 32]
        t_strides = temporal_strides or [2, 2, 2]

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_bnd = torch.tensor(0.0, device=device)
        total_tube_cls = torch.tensor(0.0, device=device)
        total_tube_box = torch.tensor(0.0, device=device)

        _target_cache = {}

        for si, preds in enumerate(predictions):
            cls_pred, reg_pred, obj_pred = preds[0], preds[1], preds[2]
            bnd_pred = preds[3] if len(preds) > 3 else None

            stride = s_strides[si]
            t_stride = t_strides[si]
            nc = cls_pred.shape[1]
            T = cls_pred.shape[2]
            S = cls_pred.shape[3]

            cache_key = (T, t_stride)
            if cache_key not in _target_cache:
                _target_cache[cache_key] = self._preprocess_targets_with_boundary(
                    targets, B, T, device, t_stride=t_stride)
            gt_labels, gt_boxes, mask_gt, gt_boundary = _target_cache[cache_key]

            anchors = self._make_anchors(S, stride, device)

            cls_flat = cls_pred.permute(0, 2, 1, 3, 4).reshape(B*T, nc, S*S).permute(0, 2, 1)
            reg_flat = reg_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 4, S*S).permute(0, 2, 1)
            obj_flat = obj_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 1, S*S).permute(0, 2, 1)

            pred_boxes = self._decode_boxes(reg_flat, anchors, stride)

            target_bboxes, target_scores, fg_mask = self.assigner(
                cls_flat.detach().sigmoid(),
                pred_boxes.detach(),
                anchors,
                gt_labels, gt_boxes, mask_gt,
            )

            num_fg = max(fg_mask.sum().item(), 1)
            normalizer = max(target_scores.sum().item(), num_fg)

            cls_loss = self.bce_cls(cls_flat, target_scores).sum() / normalizer
            total_cls = total_cls + cls_loss

            if fg_mask.any():
                from .loss import bbox_iou_ciou
                pred_fg = pred_boxes[fg_mask]
                target_fg = target_bboxes[fg_mask]
                iou = bbox_iou_ciou(pred_fg, target_fg)
                weight = target_scores.sum(-1)[fg_mask]
                box_loss = ((1.0 - iou) * weight).sum() / normalizer
                total_box = total_box + box_loss

            obj_target = fg_mask.unsqueeze(-1).float()
            obj_loss = self.bce_obj(obj_flat, obj_target).mean()
            total_obj = total_obj + obj_loss

            if T > 1 and (self.lambda_tube_cls > 0 or self.lambda_tube_box > 0):
                cls_prob = cls_flat.sigmoid().reshape(B, T, S * S, nc)
                box_seq = pred_boxes.reshape(B, T, S * S, 4)
                obj_prob = obj_flat.sigmoid().reshape(B, T, S * S)
                tube_weight = (obj_prob[:, 1:] * obj_prob[:, :-1]).clamp(min=0)
                if self.tube_obj_power != 1.0:
                    tube_weight = tube_weight.pow(self.tube_obj_power)
                tube_weight = tube_weight.detach()
                denom = tube_weight.sum().clamp(min=1.0)

                if self.lambda_tube_cls > 0:
                    cls_delta = (cls_prob[:, 1:] - cls_prob[:, :-1]).pow(2).sum(-1)
                    total_tube_cls = total_tube_cls + (cls_delta * tube_weight).sum() / denom

                if self.lambda_tube_box > 0:
                    box_delta = torch.nn.functional.smooth_l1_loss(
                        box_seq[:, 1:],
                        box_seq[:, :-1],
                        reduction='none',
                        beta=0.02,
                    ).sum(-1)
                    total_tube_box = total_tube_box + (box_delta * tube_weight).sum() / denom

            # Boundary loss (foreground cells only)
            if bnd_pred is not None and fg_mask.any():
                bnd_flat = bnd_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 1, S*S).permute(0, 2, 1)

                # Get boundary target for each foreground cell
                # fg_mask: (BT, A), we need the GT index each fg cell is matched to
                gt_idx = self.assigner_gt_idx if hasattr(self, 'assigner_gt_idx') else None

                # Simpler approach: for each fg cell, find which GT it matched
                # and look up that GT's boundary flag
                # The assigner returns target_bboxes which we can use to find GT idx
                # But we need to reconstruct from mask_pos... let's use a direct approach

                # For each (bt, anchor) in fg_mask, find closest GT box
                bnd_target = torch.zeros(B*T, S*S, device=device)
                for bt in range(B*T):
                    fg_anchors = fg_mask[bt].nonzero(as_tuple=True)[0]
                    if len(fg_anchors) == 0:
                        continue
                    # Match fg predictions to GT by finding closest box
                    pred_fg_bt = pred_boxes[bt, fg_anchors]  # (nfg, 4)
                    gt_boxes_bt = gt_boxes[bt]  # (max_gt, 4)
                    n_gt = int(mask_gt[bt, :, 0].sum().item())
                    if n_gt == 0:
                        continue
                    # IoU between fg preds and GT
                    ious = self._pairwise_iou(pred_fg_bt, gt_boxes_bt[:n_gt])  # (nfg, n_gt)
                    matched_gt = ious.argmax(dim=1)  # (nfg,)
                    bnd_target[bt, fg_anchors] = gt_boundary[bt, matched_gt]

                bnd_loss = self.bce_bnd(
                    bnd_flat[fg_mask].squeeze(-1),
                    bnd_target[fg_mask]
                ).mean()
                total_bnd = total_bnd + bnd_loss

        ns = len(predictions)
        total_cls /= ns
        total_box /= ns
        total_obj /= ns
        total_bnd /= ns
        total_tube_cls /= ns
        total_tube_box /= ns

        loss = (self.lambda_cls * total_cls + self.lambda_box * total_box +
                self.lambda_obj * total_obj + self.lambda_bnd * total_bnd +
                self.lambda_tube_cls * total_tube_cls +
                self.lambda_tube_box * total_tube_box)

        loss_dict = {
            'loss': (loss * B).item(),
            'cls_loss': total_cls.item(),
            'box_loss': total_box.item(),
            'obj_loss': total_obj.item(),
            'bnd_loss': total_bnd.item(),
            'num_fg': int(fg_mask.sum().item()) if 'fg_mask' in dir() else 0,
        }
        if self.lambda_tube_cls > 0:
            loss_dict['tube_cls_loss'] = total_tube_cls.item()
        if self.lambda_tube_box > 0:
            loss_dict['tube_box_loss'] = total_tube_box.item()
        return loss * B, loss_dict

    @staticmethod
    def _pairwise_iou(boxes1, boxes2):
        """Compute pairwise IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2[None, :] - inter
        return inter / union.clamp(min=1e-7)
