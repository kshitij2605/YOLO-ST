"""YOLO-ST Loss — rewritten to match YOLOST_OLD reference (85% f-mAP).

Key insight: Reshape (B, C, T, S, S) → (B*T, S*S, C) so each frame is
treated as an independent 2D detection problem. This lets us use a standard
batched TAL assigner over B*T "images" at once.
"""

import math
import torch
import torch.nn as nn


from .dfl import decode_ltrb, distribution_expectation


def bbox_iou_ciou(box1, box2, eps=1e-7):
    """CIoU between paired boxes (element-wise)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    w1, h1 = (b1_x2 - b1_x1).clamp(0), (b1_y2 - b1_y1).clamp(0)
    w2, h2 = (b2_x2 - b2_x1).clamp(0), (b2_y2 - b2_y1).clamp(0)

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = (((b2_x1 + b2_x2) - (b1_x1 + b1_x2)).pow(2) +
            ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)).pow(2)) / 4

    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    return iou - rho2 / c2 - alpha * v


class TaskAlignedAssigner(nn.Module):
    """Vectorized TAL assigner, matching YOLOST_OLD reference exactly."""

    def __init__(self, top_k=10, nc=24, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.top_k = top_k
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, cls_scores, pred_boxes, anchor_points, gt_labels,
                gt_boxes, mask_gt):
        """
        Args:
            cls_scores: (BT, A, nc) sigmoid'd.
            pred_boxes: (BT, A, 4) decoded x1y1x2y2 normalized.
            anchor_points: (A, 2) normalized (cx, cy).
            gt_labels: (BT, max_gt, 1) class indices.
            gt_boxes: (BT, max_gt, 4) x1y1x2y2 normalized.
            mask_gt: (BT, max_gt, 1) valid mask.

        Returns:
            target_bboxes: (BT, A, 4)
            target_scores: (BT, A, nc)
            fg_mask: (BT, A) bool
        """
        bs = cls_scores.shape[0]
        na = pred_boxes.shape[1]
        n_max = gt_boxes.shape[1]

        if n_max == 0:
            return (torch.zeros_like(pred_boxes),
                    torch.zeros_like(cls_scores),
                    torch.zeros(bs, na, dtype=torch.bool, device=cls_scores.device))

        # Which anchors fall inside which GT boxes
        lt, rb = gt_boxes.view(-1, 1, 4).chunk(2, 2)
        box_delta = torch.cat((anchor_points[None] - lt, rb - anchor_points[None]), dim=2)
        mask_in_gts = box_delta.view(bs, n_max, na, -1).amin(3).gt_(self.eps)
        mask_gts = (mask_in_gts * mask_gt.squeeze(-1).unsqueeze(-1)).bool()

        # Compute overlaps and cls scores at GT class
        overlaps = torch.zeros(bs, n_max, na, dtype=pred_boxes.dtype, device=pred_boxes.device)
        bbox_scores = torch.zeros(bs, n_max, na, dtype=cls_scores.dtype, device=cls_scores.device)

        multilabel = gt_labels.ndim == 3 and gt_labels.shape[-1] == self.nc
        if multilabel:
            label_weights = gt_labels.to(dtype=cls_scores.dtype)
            cls_per_gt = (cls_scores[:, None, :, :] * label_weights[:, :, None, :]).amax(-1)
            bbox_scores[mask_gts] = cls_per_gt[mask_gts]
        else:
            ind = torch.zeros(2, bs, n_max, dtype=torch.long, device=cls_scores.device)
            ind[0] = torch.arange(bs, device=cls_scores.device).view(-1, 1).expand(-1, n_max)
            ind[1] = gt_labels.squeeze(-1).long()
            bbox_scores[mask_gts] = cls_scores[ind[0], :, ind[1]][mask_gts]

        pd = pred_boxes.unsqueeze(1).expand(-1, n_max, -1, -1)[mask_gts]
        gt_exp = gt_boxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gts]
        overlaps[mask_gts] = bbox_iou_ciou(pd, gt_exp).clamp_(0)

        # Alignment metric
        metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        # Top-k selection
        top_mask = mask_gt.expand(-1, -1, self.top_k).bool()
        top_metrics, top_id = torch.topk(metric, self.top_k, dim=-1, largest=True)
        top_id.masked_fill_(~top_mask, 0)

        count_tensor = torch.zeros(metric.shape, dtype=torch.int8, device=cls_scores.device)
        ones = torch.ones_like(top_id[:, :, :1], dtype=torch.int8)
        for k in range(self.top_k):
            count_tensor.scatter_add_(-1, top_id[:, :, k:k+1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        mask_pos = count_tensor.to(metric.dtype) * mask_in_gts * mask_gt.squeeze(-1).unsqueeze(-1)

        # Resolve multi-GT conflicts
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max, -1)
            max_over = torch.zeros_like(mask_pos)
            max_over.scatter_(1, overlaps.argmax(1).unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi, max_over, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        gt_idx = mask_pos.argmax(-2)
        batch_ind = torch.arange(bs, dtype=torch.long, device=cls_scores.device).unsqueeze(-1)
        gt_idx_flat = gt_idx + batch_ind * n_max

        if multilabel:
            target_labels = gt_labels.reshape(bs * n_max, self.nc)[gt_idx_flat].float()
        else:
            target_labels = gt_labels.long().flatten()[gt_idx_flat]
        target_bboxes = gt_boxes.view(-1, 4)[gt_idx_flat]
        if multilabel:
            target_scores = target_labels.to(dtype=cls_scores.dtype, device=cls_scores.device)
        else:
            target_labels.clamp_(0)
            target_scores = torch.zeros(bs, na, self.nc, dtype=torch.long, device=cls_scores.device)
            target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        # Normalize by alignment metric (soft labels)
        metric *= mask_pos
        pos_metrics = metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_metric = metric * pos_overlaps / (pos_metrics + self.eps)
        target_scores = target_scores.float() * norm_metric.amax(-2).unsqueeze(-1)

        return target_bboxes, target_scores, fg_mask.bool()


class YOLOSTLoss(nn.Module):
    """YOLO-ST loss matching the reference implementation.

    Reshapes (B, C, T, S, S) → (B*T, S*S, C) to treat each frame independently.
    """

    def __init__(self, num_classes=24, lambda_cls=0.5, lambda_box=7.5,
                 lambda_obj=1.5, img_size=224):
        super().__init__()
        self.nc = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.img_size = img_size
        # Weight of the distribution term; only used when the head
        # emits more than four regression channels. YOLOv8 uses 1.5
        # against a box weight of 7.5.
        self.lambda_dfl = 1.5
        self.assigner = TaskAlignedAssigner(top_k=10, nc=num_classes)
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

    def _make_anchors(self, S, stride, device, width=None):
        """Normalized anchor centers (S*width, 2); width defaults to S (square)."""
        from .geometry import image_hw
        img_h, img_w = image_hw(self.img_size)
        columns = S if width is None else width
        sy = (torch.arange(S, device=device, dtype=torch.float32) + 0.5) * stride / img_h
        sx = (torch.arange(columns, device=device, dtype=torch.float32) + 0.5) * stride / img_w
        gy, gx = torch.meshgrid(sy, sx, indexing='ij')
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def _decode_boxes(self, reg_flat, anchors, stride):
        """Decode (BT, S*S, C) → boxes in normalized coords.

        C == 4 is the original centre/size parameterisation. C == 4*(reg_max+1)
        is distribution regression over LTRB distances; the format is inferred
        from the channel count so the head and the loss cannot disagree.
        """
        channels = reg_flat.shape[-1]
        if channels != 4:
            if channels % 4:
                raise ValueError(
                    f'regression channels {channels} is neither 4 nor a '
                    'multiple of 4'
                )
            reg_max = channels // 4 - 1
            distance = distribution_expectation(reg_flat, reg_max)
            from .geometry import image_hw
            img_h, img_w = image_hw(self.img_size)
            return decode_ltrb(distance, anchors, (stride / img_w, stride / img_h))

        dx = torch.sigmoid(reg_flat[..., 0])
        dy = torch.sigmoid(reg_flat[..., 1])
        dw = reg_flat[..., 2]
        dh = reg_flat[..., 3]

        from .geometry import image_hw
        img_h, img_w = image_hw(self.img_size)
        step_x = stride / img_w
        step_y = stride / img_h
        cx = anchors[:, 0] + (dx - 0.5) * step_x
        cy = anchors[:, 1] + (dy - 0.5) * step_y
        w = torch.exp(dw.clamp(max=5.0)) * step_x
        h = torch.exp(dh.clamp(max=5.0)) * step_y

        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    def _preprocess_targets(self, targets, B, T_det, device, t_stride=2):
        """Convert per-clip targets to per-detection-frame format.

        GT frame indices (0..T_clip-1) are mapped to detection frame indices
        (0..T_det-1) by integer division: det_frame = clip_frame // t_stride.
        Multiple GT boxes mapping to the same detection frame are all kept.

        Returns:
            gt_labels: (B*T_det, max_gt, 1)
            gt_boxes: (B*T_det, max_gt, 4)
            mask_gt: (B*T_det, max_gt, 1)
        """
        all_labels = []
        all_boxes = []
        all_counts = []

        for b in range(B):
            boxes_b = targets['boxes'][b]    # (N, 5): (frame_idx, x1, y1, x2, y2)
            labels_b = targets['labels'][b]  # (N,) or (N, nc)

            for t in range(T_det):
                if boxes_b.numel() == 0:
                    if targets['labels'].ndim == 3:
                        all_labels.append(torch.zeros(0, self.nc, device=device))
                    else:
                        all_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                    all_boxes.append(torch.zeros(0, 4, device=device))
                    all_counts.append(0)
                else:
                    # Map: detection frame t covers clip frames [t*t_stride .. (t+1)*t_stride-1]
                    det_frame = boxes_b[:, 0].long() // t_stride
                    frame_mask = (det_frame == t) & (boxes_b.sum(dim=-1) > 0)
                    gt_b = boxes_b[frame_mask, 1:5]
                    gt_l = labels_b[frame_mask]
                    all_labels.append(gt_l.to(device))
                    all_boxes.append(gt_b.to(device))
                    all_counts.append(len(gt_l))

        BT = B * T_det
        max_gt = max(all_counts) if all_counts else 0
        multilabel = targets['labels'].ndim == 3
        label_shape = (BT, 0, self.nc) if multilabel else (BT, 0, 1)
        if max_gt == 0:
            return (torch.zeros(label_shape, dtype=torch.float32 if multilabel else torch.long, device=device),
                    torch.zeros(BT, 0, 4, device=device),
                    torch.zeros(BT, 0, 1, device=device))

        if multilabel:
            gt_labels = torch.zeros(BT, max_gt, self.nc, dtype=torch.float32, device=device)
        else:
            gt_labels = torch.zeros(BT, max_gt, 1, dtype=torch.long, device=device)
        gt_boxes = torch.zeros(BT, max_gt, 4, device=device)
        mask_gt = torch.zeros(BT, max_gt, 1, device=device)

        for i in range(BT):
            n = all_counts[i]
            if n > 0:
                if multilabel:
                    gt_labels[i, :n] = all_labels[i].float()
                else:
                    gt_labels[i, :n, 0] = all_labels[i]
                gt_boxes[i, :n] = all_boxes[i]
                mask_gt[i, :n, 0] = 1.0

        return gt_labels, gt_boxes, mask_gt

    def forward(self, predictions, targets, temporal_strides=None,
                spatial_strides=None, img_size=None):
        """
        Args:
            predictions: list of (cls, reg, obj) per scale.
                cls: (B, nc, T, S, S), reg: (B, 4, T, S, S), obj: (B, 1, T, S, S)
            targets: dict with 'boxes' (B, maxN, 5) and 'labels' (B, maxN).
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

        # Cache preprocessed targets per unique T value
        _target_cache = {}

        for si, (cls_pred, reg_pred, obj_pred) in enumerate(predictions):
            stride = s_strides[si]
            t_stride = t_strides[si]
            nc = cls_pred.shape[1]
            T = cls_pred.shape[2]
            S = cls_pred.shape[3]

            # Preprocess targets for this temporal resolution (cache to avoid recomputing)
            cache_key = (T, t_stride)
            if cache_key not in _target_cache:
                _target_cache[cache_key] = self._preprocess_targets(
                    targets, B, T, device, t_stride=t_stride)
            gt_labels, gt_boxes, mask_gt = _target_cache[cache_key]

            anchors = self._make_anchors(S, stride, device)

            # Reshape (B, C, T, S, S) → (B*T, S*S, C)
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

            # Classification loss (all cells, soft targets from alignment)
            cls_loss = self.bce_cls(cls_flat, target_scores).sum() / normalizer
            total_cls = total_cls + cls_loss

            # Box loss (foreground only, weighted by alignment score)
            if fg_mask.any():
                pred_fg = pred_boxes[fg_mask]
                target_fg = target_bboxes[fg_mask]
                iou = bbox_iou_ciou(pred_fg, target_fg)
                weight = target_scores.sum(-1)[fg_mask]
                box_loss = ((1.0 - iou) * weight).sum() / normalizer
                total_box = total_box + box_loss

            # Objectness loss
            obj_target = fg_mask.unsqueeze(-1).float()
            obj_loss = self.bce_obj(obj_flat, obj_target).mean()
            total_obj = total_obj + obj_loss

        ns = len(predictions)
        total_cls /= ns
        total_box /= ns
        total_obj /= ns

        loss = self.lambda_cls * total_cls + self.lambda_box * total_box + self.lambda_obj * total_obj

        return loss * B, {
            'loss': (loss * B).item(),
            'cls_loss': total_cls.item(),
            'box_loss': total_box.item(),
            'obj_loss': total_obj.item(),
            'num_fg': int(fg_mask.sum().item()) if 'fg_mask' in dir() else 0,
        }
