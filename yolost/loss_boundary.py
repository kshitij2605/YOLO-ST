"""YOLO-ST Loss with Boundary Head (Phase 3A).

Extends YOLOSTLoss with boundary prediction loss.
Boundary target: 1 if GT frame is within margin of tube start/end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import YOLOSTLoss
from .tube_query import tube_query_loss


from .geometry import image_hw as _image_hw


class YOLOSTLossBoundary(YOLOSTLoss):
    """Loss with additional boundary prediction term."""

    def __init__(self, num_classes=24, lambda_cls=0.5, lambda_box=7.5,
                 lambda_obj=1.5, lambda_bnd=0.5, lambda_tube_cls=0.0,
                 lambda_tube_box=0.0, tube_obj_power=0.5,
                 lambda_dn_cls=0.0, lambda_dn_box=0.0, lambda_memory=0.0,
                 lambda_query_cls=0.0, lambda_query_box=0.0,
                 lambda_query_giou=0.0, lambda_query_visibility=0.0,
                 lambda_query_boundary=0.0, lambda_query_velocity=0.0,
                 lambda_query_acceleration=0.0,
                 lambda_query_start=0.0, lambda_query_end=0.0,
                 lambda_query_interval_iou=0.0, lambda_query_coverage=0.0,
                 lambda_query_fragmentation=0.0,
                 lambda_query_boundary_distance=0.0,
                 lambda_query_boundary_distance_slope=0.0,
                 lambda_query_quality=0.0,
                 lambda_query_transport=0.0,
                 lambda_query_geometry_preservation=0.0,
                 query_cost_cls=2.0, query_cost_box=5.0,
                 query_cost_giou=2.0, query_cost_visibility=1.0,
                 query_cost_interval=0.0, query_cost_coverage=0.0,
                 query_cost_fragmentation=0.0,
                 query_cost_transport=0.0,
                  query_boundary_pos_weight=1.0,
                  query_boundary_focal_gamma=0.0,
                  query_class_focal_alpha=None, query_class_focal_gamma=2.0,
                  query_quality_target_mode="sqrt_product",
                  query_quality_strict_blend=0.5,
                  clip_length=64,
                  img_size=224):
        super().__init__(num_classes, lambda_cls, lambda_box, lambda_obj, img_size)
        self.lambda_bnd = lambda_bnd
        self.lambda_tube_cls = lambda_tube_cls
        self.lambda_tube_box = lambda_tube_box
        self.tube_obj_power = tube_obj_power
        self.lambda_dn_cls = lambda_dn_cls
        self.lambda_dn_box = lambda_dn_box
        self.lambda_memory = lambda_memory
        self.lambda_query_cls = lambda_query_cls
        self.lambda_query_box = lambda_query_box
        self.lambda_query_giou = lambda_query_giou
        self.lambda_query_visibility = lambda_query_visibility
        self.lambda_query_boundary = lambda_query_boundary
        self.lambda_query_velocity = lambda_query_velocity
        self.lambda_query_acceleration = lambda_query_acceleration
        self.lambda_query_start = lambda_query_start
        self.lambda_query_end = lambda_query_end
        self.lambda_query_interval_iou = lambda_query_interval_iou
        self.lambda_query_coverage = lambda_query_coverage
        self.lambda_query_fragmentation = lambda_query_fragmentation
        self.lambda_query_boundary_distance = lambda_query_boundary_distance
        self.lambda_query_boundary_distance_slope = (
            lambda_query_boundary_distance_slope
        )
        self.lambda_query_quality = lambda_query_quality
        self.lambda_query_transport = lambda_query_transport
        self.lambda_query_geometry_preservation = (
            lambda_query_geometry_preservation
        )
        self.query_cost_cls = query_cost_cls
        self.query_cost_box = query_cost_box
        self.query_cost_giou = query_cost_giou
        self.query_cost_visibility = query_cost_visibility
        self.query_cost_interval = query_cost_interval
        self.query_cost_coverage = query_cost_coverage
        self.query_cost_fragmentation = query_cost_fragmentation
        self.query_cost_transport = query_cost_transport
        self.query_boundary_pos_weight = query_boundary_pos_weight
        self.query_boundary_focal_gamma = query_boundary_focal_gamma
        self.query_class_focal_alpha = query_class_focal_alpha
        self.query_class_focal_gamma = query_class_focal_gamma
        self.query_quality_target_mode = query_quality_target_mode
        self.query_quality_strict_blend = query_quality_strict_blend
        self.clip_length = clip_length
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
        tube_denoising = None
        memory_consistency = None
        tube_queries = None
        if isinstance(predictions, dict):
            tube_denoising = predictions.get("tube_denoising")
            memory_consistency = predictions.get("memory_consistency")
            tube_queries = predictions.get("tube_queries")
            predictions = predictions["dense"]

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
        total_dfl = torch.tensor(0.0, device=device)
        total_tube_cls = torch.tensor(0.0, device=device)
        total_tube_box = torch.tensor(0.0, device=device)
        total_dn_cls = torch.tensor(0.0, device=device)
        total_dn_box = torch.tensor(0.0, device=device)
        total_memory = torch.tensor(0.0, device=device)
        query_losses = None

        _target_cache = {}

        for si, preds in enumerate(predictions):
            cls_pred, reg_pred, obj_pred = preds[0], preds[1], preds[2]
            bnd_pred = preds[3] if len(preds) > 3 else None

            stride = s_strides[si]
            t_stride = t_strides[si]
            nc = cls_pred.shape[1]
            T = cls_pred.shape[2]
            S = cls_pred.shape[3]
            A = S * cls_pred.shape[4]

            cache_key = (T, t_stride)
            if cache_key not in _target_cache:
                _target_cache[cache_key] = self._preprocess_targets_with_boundary(
                    targets, B, T, device, t_stride=t_stride)
            gt_labels, gt_boxes, mask_gt, gt_boundary = _target_cache[cache_key]

            # Rows (clip, detection frame) holding a labelled frame. None means
            # every frame is labelled, as on UCF101-24 and JHMDB.
            row_mask = None
            supervised = targets.get('supervised_frames')
            if supervised is not None:
                rows = torch.zeros(B, T, dtype=torch.bool, device=device)
                for b in range(B):
                    for frame in supervised[b].tolist():
                        if frame >= 0:
                            rows[b, min(int(frame) // t_stride, T - 1)] = True
                row_mask = rows.reshape(B * T)

            # Patch 0031: per-row weight of the dense cls/obj terms. A row covers
            # t_stride clip frames and takes their minimum weight.
            row_weight = None
            frame_weights = targets.get('frame_weights')
            if frame_weights is not None:
                frame_weights = frame_weights.to(device=device, dtype=torch.float32)
                usable = min(frame_weights.shape[1] // t_stride, T)
                per_row = torch.ones(B, T, device=device)
                per_row[:, :usable] = frame_weights[:, :usable * t_stride].reshape(
                    B, usable, t_stride).amin(-1)
                row_weight = per_row.reshape(B * T, 1, 1)

            anchors = self._make_anchors(S, stride, device, width=cls_pred.shape[4])

            cls_flat = cls_pred.permute(0, 2, 1, 3, 4).reshape(B*T, nc, A).permute(0, 2, 1)
            reg_channels = reg_pred.shape[1]
            reg_flat = reg_pred.permute(0, 2, 1, 3, 4).reshape(
                B*T, reg_channels, A).permute(0, 2, 1)
            obj_flat = obj_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 1, A).permute(0, 2, 1)

            pred_boxes = self._decode_boxes(reg_flat, anchors, stride)

            target_bboxes, target_scores, fg_mask = self.assigner(
                cls_flat.detach().sigmoid(),
                pred_boxes.detach(),
                anchors,
                gt_labels, gt_boxes, mask_gt,
            )

            num_fg = max(fg_mask.sum().item(), 1)
            normalizer = max(target_scores.sum().item(), num_fg)

            cls_terms = self.bce_cls(cls_flat, target_scores)
            if row_weight is not None:
                cls_terms = cls_terms * row_weight
            if row_mask is not None:
                cls_terms = cls_terms[row_mask]
            cls_loss = cls_terms.sum() / normalizer
            total_cls = total_cls + cls_loss

            if fg_mask.any():
                from .loss import bbox_iou_ciou
                pred_fg = pred_boxes[fg_mask]
                target_fg = target_bboxes[fg_mask]
                iou = bbox_iou_ciou(pred_fg, target_fg)
                weight = target_scores.sum(-1)[fg_mask]
                box_loss = ((1.0 - iou) * weight).sum() / normalizer
                total_box = total_box + box_loss

                if reg_channels != 4:
                    from .dfl import (
                        distribution_focal_loss as _dfl_loss,
                        encode_ltrb as _encode_ltrb,
                    )
                    reg_max = reg_channels // 4 - 1
                    # Encode over every cell so anchors broadcast correctly,
                    # then select the foreground rows.
                    target_distance = _encode_ltrb(
                        target_bboxes, anchors,
                        (stride / _image_hw(self.img_size)[1], stride / _image_hw(self.img_size)[0]),
                        reg_max
                    )[fg_mask]
                    dfl = _dfl_loss(
                        reg_flat[fg_mask], target_distance, reg_max
                    )
                    total_dfl = total_dfl + (
                        (dfl * weight).sum() / normalizer
                    )

            obj_target = fg_mask.unsqueeze(-1).float()
            obj_terms = self.bce_obj(obj_flat, obj_target)
            if row_weight is not None:
                obj_terms = obj_terms * row_weight
            if row_mask is not None:
                obj_terms = obj_terms[row_mask]
            obj_loss = obj_terms.mean()
            total_obj = total_obj + obj_loss

            if T > 1 and (self.lambda_tube_cls > 0 or self.lambda_tube_box > 0):
                cls_prob = cls_flat.sigmoid().reshape(B, T, A, nc)
                box_seq = pred_boxes.reshape(B, T, A, 4)
                obj_prob = obj_flat.sigmoid().reshape(B, T, A)
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
                bnd_flat = bnd_pred.permute(0, 2, 1, 3, 4).reshape(B*T, 1, A).permute(0, 2, 1)

                # Get boundary target for each foreground cell
                # fg_mask: (BT, A), we need the GT index each fg cell is matched to
                gt_idx = self.assigner_gt_idx if hasattr(self, 'assigner_gt_idx') else None

                # Simpler approach: for each fg cell, find which GT it matched
                # and look up that GT's boundary flag
                # The assigner returns target_bboxes which we can use to find GT idx
                # But we need to reconstruct from mask_pos... let's use a direct approach

                # For each (bt, anchor) in fg_mask, find closest GT box
                bnd_target = torch.zeros(B*T, A, device=device)
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
        total_dfl /= ns
        total_tube_cls /= ns
        total_tube_box /= ns

        if tube_denoising is not None:
            tube_mask = tube_denoising["mask"]
            if tube_mask.any():
                box_error = F.smooth_l1_loss(
                    tube_denoising["pred_boxes"],
                    tube_denoising["target_boxes"],
                    reduction="none",
                    beta=0.05,
                ).sum(dim=-1)
                total_dn_box = box_error[tube_mask].mean()
                total_dn_cls = F.cross_entropy(
                    tube_denoising["pred_logits"], tube_denoising["target_labels"]
                )
        if memory_consistency is not None:
            total_memory = memory_consistency
        if tube_queries is not None:
            query_losses = tube_query_loss(
                tube_queries, targets, self.clip_length,
                cost_cls=self.query_cost_cls, cost_box=self.query_cost_box,
                cost_giou=self.query_cost_giou,
                cost_visibility=self.query_cost_visibility,
                cost_interval=self.query_cost_interval,
                cost_coverage=self.query_cost_coverage,
                cost_fragmentation=self.query_cost_fragmentation,
                cost_transport=self.query_cost_transport,
                boundary_pos_weight=self.query_boundary_pos_weight,
                boundary_focal_gamma=self.query_boundary_focal_gamma,
                class_focal_alpha=self.query_class_focal_alpha,
                class_focal_gamma=self.query_class_focal_gamma,
                quality_target_mode=self.query_quality_target_mode,
                quality_strict_blend=self.query_quality_strict_blend,
            )

        frame_loss = (self.lambda_cls * total_cls +
                      self.lambda_box * total_box +
                      self.lambda_obj * total_obj +
                      getattr(self, 'lambda_dfl', 1.5) * total_dfl)
        tube_boundary_loss = (
            self.lambda_bnd * total_bnd +
            self.lambda_tube_cls * total_tube_cls +
            self.lambda_tube_box * total_tube_box
        )
        auxiliary_loss = (self.lambda_dn_cls * total_dn_cls +
                          self.lambda_dn_box * total_dn_box +
                          self.lambda_memory * total_memory)
        if query_losses is not None:
            tube_boundary_loss = (
                tube_boundary_loss +
                self.lambda_query_cls * query_losses['query_cls_loss'] +
                self.lambda_query_box * query_losses['query_box_loss'] +
                self.lambda_query_giou * query_losses['query_giou_loss'] +
                self.lambda_query_visibility * query_losses['query_visibility_loss'] +
                self.lambda_query_boundary * query_losses['query_boundary_loss'] +
                self.lambda_query_velocity * query_losses['query_velocity_loss'] +
                self.lambda_query_acceleration * query_losses['query_acceleration_loss'] +
                self.lambda_query_start * query_losses['query_start_loss'] +
                self.lambda_query_end * query_losses['query_end_loss'] +
                self.lambda_query_interval_iou * query_losses['query_interval_iou_loss'] +
                self.lambda_query_coverage * query_losses['query_coverage_loss'] +
                self.lambda_query_fragmentation * query_losses['query_fragmentation_loss'] +
                self.lambda_query_boundary_distance * query_losses['query_boundary_distance_loss'] +
                self.lambda_query_boundary_distance_slope * query_losses['query_boundary_distance_slope_loss'] +
                self.lambda_query_quality * query_losses['query_quality_loss'] +
                self.lambda_query_transport * query_losses['query_transport_loss'] +
                self.lambda_query_geometry_preservation *
                query_losses['query_geometry_preservation_loss']
            )

        loss = frame_loss + tube_boundary_loss + auxiliary_loss
        # Retain differentiable objective partitions for optional conflict
        # diagnostics. Training behavior is unchanged because they sum to loss.
        self.last_loss_components = {
            'frame': frame_loss * B,
            'tube_boundary': tube_boundary_loss * B,
            'auxiliary': auxiliary_loss * B,
        }

        loss_dict = {
            'loss': (loss * B).item(),
            'cls_loss': total_cls.item(),
            'box_loss': total_box.item(),
            'obj_loss': total_obj.item(),
            'bnd_loss': total_bnd.item(),
            'dfl_loss': total_dfl.item(),
            'num_fg': int(fg_mask.sum().item()) if 'fg_mask' in dir() else 0,
        }
        if self.lambda_tube_cls > 0:
            loss_dict['tube_cls_loss'] = total_tube_cls.item()
        if self.lambda_tube_box > 0:
            loss_dict['tube_box_loss'] = total_tube_box.item()
        if self.lambda_dn_cls > 0:
            loss_dict['dn_cls_loss'] = total_dn_cls.item()
        if self.lambda_dn_box > 0:
            loss_dict['dn_box_loss'] = total_dn_box.item()
        if self.lambda_memory > 0:
            loss_dict['memory_loss'] = total_memory.item()
        if query_losses is not None:
            loss_dict.update({key: value.item() if torch.is_tensor(value) else value
                              for key, value in query_losses.items()})
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
