"""Bipartite matching loss for the BMViT-lite probe."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import YOLOSTLoss

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is in requirements.txt.
    linear_sum_assignment = None


def _box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def pairwise_iou(boxes1, boxes2):
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = _box_area(boxes1)[:, None] + _box_area(boxes2)[None, :] - inter
    return inter / union.clamp(min=1e-7)


def pairwise_giou(boxes1, boxes2):
    lt_i = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_i = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_i = (rb_i - lt_i).clamp(min=0)
    inter = wh_i[:, :, 0] * wh_i[:, :, 1]
    area1 = _box_area(boxes1)[:, None]
    area2 = _box_area(boxes2)[None, :]
    union = (area1 + area2 - inter).clamp(min=1e-7)
    iou = inter / union

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = (wh[:, :, 0] * wh[:, :, 1]).clamp(min=1e-7)
    return iou - (enclosing - union) / enclosing


class YOLOSTBipartiteLoss(YOLOSTLoss):
    """One-to-one Hungarian matching loss over dense predictions per frame."""

    def __init__(self, num_classes=24, lambda_cls=1.0, lambda_box=5.0,
                 lambda_giou=2.0, lambda_obj=1.0, img_size=224,
                 cost_cls=2.0, cost_box=5.0, cost_giou=2.0, cost_obj=1.0,
                 no_object_weight=0.05):
        super().__init__(num_classes, lambda_cls, lambda_box, lambda_obj, img_size)
        self.lambda_giou = lambda_giou
        self.cost_cls = cost_cls
        self.cost_box = cost_box
        self.cost_giou = cost_giou
        self.cost_obj = cost_obj
        self.no_object_weight = no_object_weight

    @torch.no_grad()
    def _match(self, cls_logits, obj_logits, pred_boxes, gt_labels, gt_boxes, mask_gt):
        """Return prediction and GT indices for one detection frame."""
        n_gt = int(mask_gt[:, 0].sum().item())
        if n_gt == 0:
            empty = torch.empty(0, dtype=torch.long, device=cls_logits.device)
            return empty, empty
        if linear_sum_assignment is None:
            raise RuntimeError("scipy is required for Hungarian matching")

        labels = gt_labels[:n_gt, 0].long()
        boxes = gt_boxes[:n_gt]
        cls_prob = cls_logits.sigmoid()
        obj_prob = obj_logits.sigmoid().squeeze(-1)

        cls_cost = -cls_prob[:, labels]
        obj_cost = -obj_prob[:, None]
        box_cost = torch.cdist(pred_boxes, boxes, p=1)
        giou_cost = -pairwise_giou(pred_boxes, boxes)
        cost = (
            self.cost_cls * cls_cost +
            self.cost_obj * obj_cost +
            self.cost_box * box_cost +
            self.cost_giou * giou_cost
        )
        row, col = linear_sum_assignment(cost.detach().cpu().float().numpy())
        return (
            torch.as_tensor(row, dtype=torch.long, device=cls_logits.device),
            torch.as_tensor(col, dtype=torch.long, device=cls_logits.device),
        )

    def forward(self, predictions, targets, temporal_strides=None,
                spatial_strides=None, img_size=None):
        if img_size is not None:
            self.img_size = img_size

        device = predictions[0][0].device
        B = predictions[0][0].shape[0]
        s_strides = spatial_strides or [8, 16, 32]
        t_strides = temporal_strides or [1, 2, 4]

        total_cls = torch.tensor(0.0, device=device)
        total_box = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_fg = 0
        _target_cache = {}

        for si, preds in enumerate(predictions):
            cls_pred, reg_pred, obj_pred = preds[0], preds[1], preds[2]
            stride = s_strides[si]
            t_stride = t_strides[si]
            nc = cls_pred.shape[1]
            T = cls_pred.shape[2]
            S = cls_pred.shape[3]

            cache_key = (T, t_stride)
            if cache_key not in _target_cache:
                _target_cache[cache_key] = self._preprocess_targets(
                    targets, B, T, device, t_stride=t_stride)
            gt_labels, gt_boxes, mask_gt = _target_cache[cache_key]

            anchors = self._make_anchors(S, stride, device)
            cls_flat = cls_pred.permute(0, 2, 1, 3, 4).reshape(B*T, nc, S*S).permute(0, 2, 1)
            reg_flat = reg_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 4, S*S).permute(0, 2, 1)
            obj_flat = obj_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 1, S*S).permute(0, 2, 1)
            pred_boxes = self._decode_boxes(reg_flat, anchors, stride).clamp(0, 1)

            obj_target = torch.zeros_like(obj_flat)
            scale_cls = torch.tensor(0.0, device=device)
            scale_box = torch.tensor(0.0, device=device)
            scale_giou = torch.tensor(0.0, device=device)
            scale_fg = 0

            for bt in range(B * T):
                pred_idx, gt_idx = self._match(
                    cls_flat[bt], obj_flat[bt], pred_boxes[bt],
                    gt_labels[bt], gt_boxes[bt], mask_gt[bt],
                )
                if pred_idx.numel() == 0:
                    continue
                labels = gt_labels[bt, gt_idx, 0].long()
                boxes = gt_boxes[bt, gt_idx]
                obj_target[bt, pred_idx, 0] = 1.0
                scale_cls = scale_cls + F.binary_cross_entropy_with_logits(
                    cls_flat[bt, pred_idx],
                    F.one_hot(labels, self.nc).to(cls_flat.dtype),
                    reduction="sum",
                )
                scale_box = scale_box + F.l1_loss(pred_boxes[bt, pred_idx], boxes, reduction="sum")
                giou = pairwise_giou(pred_boxes[bt, pred_idx], boxes).diag()
                scale_giou = scale_giou + (1.0 - giou).sum()
                scale_fg += int(pred_idx.numel())

            normalizer = max(scale_fg, 1)
            obj_weight = torch.full_like(obj_target, self.no_object_weight)
            obj_weight[obj_target > 0] = 1.0
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_flat, obj_target, weight=obj_weight, reduction="sum"
            ) / max(obj_target.numel() / 1024.0, 1.0)

            total_cls = total_cls + scale_cls / normalizer
            total_box = total_box + scale_box / normalizer
            total_giou = total_giou + scale_giou / normalizer
            total_obj = total_obj + obj_loss
            total_fg += scale_fg

        ns = len(predictions)
        total_cls /= ns
        total_box /= ns
        total_giou /= ns
        total_obj /= ns

        loss = (
            self.lambda_cls * total_cls +
            self.lambda_box * total_box +
            self.lambda_giou * total_giou +
            self.lambda_obj * total_obj
        )
        return loss * B, {
            "loss": (loss * B).item(),
            "cls_loss": total_cls.item(),
            "box_loss": total_box.item(),
            "giou_loss": total_giou.item(),
            "obj_loss": total_obj.item(),
            "num_fg": total_fg,
        }
