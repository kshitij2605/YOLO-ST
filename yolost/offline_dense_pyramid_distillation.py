"""Reliability-aware full-track supervision for dense pyramid residuals."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolost.loss import TaskAlignedAssigner, bbox_iou_ciou


def _dense_outputs(outputs):
    return outputs["dense"] if isinstance(outputs, dict) else outputs


class OfflineDensePyramidDistillationLoss(nn.Module):
    """Project cached tracks onto P3/P4/P5 without teaching background cells."""

    def __init__(
        self,
        num_classes,
        class_weight=1.0,
        object_weight=0.25,
        box_weight=1.0,
        giou_weight=1.0,
        velocity_weight=0.0,
        velocity_scale_weights=None,
        velocity_rate_normalize=False,
        confidence_floor=0.05,
        interpolated_weight=0.25,
        top_k=10,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.weights = {
            "class": float(class_weight),
            "object": float(object_weight),
            "box": float(box_weight),
            "giou": float(giou_weight),
            "velocity": float(velocity_weight),
        }
        if min(self.weights.values()) < 0 or not any(self.weights.values()):
            raise ValueError(
                "offline dense-pyramid weights must be non-negative and nonzero"
            )
        self.velocity_scale_weights = (
            None
            if velocity_scale_weights is None
            else tuple(float(weight) for weight in velocity_scale_weights)
        )
        if (
            self.velocity_scale_weights is not None
            and (
                min(self.velocity_scale_weights, default=-1.0) < 0
                or (
                    self.weights["velocity"] > 0
                    and not any(self.velocity_scale_weights)
                )
            )
        ):
            raise ValueError(
                "velocity scale weights must be non-negative and include "
                "a positive scale"
            )
        self.velocity_rate_normalize = bool(velocity_rate_normalize)
        self.confidence_floor = float(confidence_floor)
        self.interpolated_weight = float(interpolated_weight)
        if not 0.0 < self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in (0, 1]")
        if not 0.0 <= self.interpolated_weight <= 1.0:
            raise ValueError("interpolated_weight must be in [0, 1]")
        self.assigner = TaskAlignedAssigner(
            top_k=int(top_k), nc=self.num_classes
        )

    @staticmethod
    def _anchors(size, stride, img_size, device):
        step = float(stride) / float(img_size)
        coordinates = (
            torch.arange(size, device=device, dtype=torch.float32) + 0.5
        ) * step
        grid_y, grid_x = torch.meshgrid(
            coordinates, coordinates, indexing="ij"
        )
        return torch.stack(
            (grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1
        )

    @staticmethod
    def _decode_boxes(regression, anchors, stride, img_size):
        step = float(stride) / float(img_size)
        center_x = (
            anchors[:, 0]
            + (regression[..., 0].float().sigmoid() - 0.5) * step
        )
        center_y = (
            anchors[:, 1]
            + (regression[..., 1].float().sigmoid() - 0.5) * step
        )
        width = regression[..., 2].float().clamp(max=5.0).exp() * step
        height = regression[..., 3].float().clamp(max=5.0).exp() * step
        return torch.stack(
            (
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            ),
            dim=-1,
        )

    def _aggregate_targets(self, targets, time_count, temporal_stride):
        boxes = targets["offline_boxes"]
        labels = targets["offline_labels"]
        track_ids = targets["offline_track_ids"]
        scores = targets["offline_scores"].float()
        qualities = targets["offline_quality"].float()
        observed = targets["offline_observed"].float()
        batch_size, observation_slots = track_ids.shape
        device = boxes.device

        valid = (
            (track_ids >= 0)
            & (labels >= 0)
            & (labels < self.num_classes)
            & (boxes[..., 1:5].sum(dim=-1) > 0)
            & torch.isfinite(boxes).all(dim=-1)
            & torch.isfinite(scores)
            & torch.isfinite(qualities)
        )
        if not valid.any():
            return (
                torch.zeros(
                    batch_size * time_count, 0, 1,
                    dtype=torch.long, device=device,
                ),
                boxes.new_zeros(batch_size * time_count, 0, 4),
                boxes.new_zeros(batch_size * time_count, 0, 1),
                boxes.new_zeros(batch_size * time_count, 0),
                torch.full(
                    (batch_size * time_count, 0), -1,
                    dtype=torch.long, device=device,
                ),
                0,
                0,
            )

        sample_indices = torch.arange(
            batch_size, device=device
        )[:, None].expand_as(track_ids)
        times = (
            boxes[..., 0].long() // int(temporal_stride)
        ).clamp(0, time_count - 1)
        slots_per_time = observation_slots + 1
        group_keys = (
            (sample_indices * time_count + times) * slots_per_time
            + track_ids.clamp_min(0)
        )

        flat_valid = valid.reshape(-1)
        flat_keys = group_keys.reshape(-1)[flat_valid]
        unique_keys, inverse = torch.unique(
            flat_keys, sorted=True, return_inverse=True
        )
        group_count = unique_keys.numel()
        group_bt = unique_keys // slots_per_time
        group_batch = group_bt // time_count
        group_track_ids = unique_keys % slots_per_time

        flat_boxes = boxes[..., 1:5].reshape(-1, 4)[flat_valid].float()
        flat_labels = labels.reshape(-1)[flat_valid]
        flat_scores = scores.reshape(-1)[flat_valid].clamp(0.0, 1.0)
        flat_qualities = qualities.reshape(-1)[flat_valid].clamp(0.0, 1.0)
        flat_observed = observed.reshape(-1)[flat_valid].clamp(0.0, 1.0)
        observation_factor = (
            self.interpolated_weight
            + flat_observed * (1.0 - self.interpolated_weight)
        )
        box_weights = (
            flat_scores.clamp_min(self.confidence_floor) * observation_factor
        )

        group_box_sum = flat_boxes.new_zeros(group_count, 4)
        group_box_sum.scatter_add_(
            0, inverse[:, None].expand(-1, 4),
            flat_boxes * box_weights[:, None],
        )
        group_box_weight = flat_boxes.new_zeros(group_count)
        group_box_weight.scatter_add_(0, inverse, box_weights)
        group_boxes = (
            group_box_sum / group_box_weight[:, None].clamp_min(1e-6)
        ).clamp(0.0, 1.0)

        group_labels = torch.zeros(
            group_count, dtype=torch.long, device=device
        )
        group_labels.scatter_(0, inverse, flat_labels)
        group_sizes = flat_boxes.new_zeros(group_count)
        group_sizes.scatter_add_(0, inverse, torch.ones_like(flat_scores))
        group_score = flat_boxes.new_zeros(group_count)
        group_score.scatter_add_(0, inverse, flat_scores)
        group_score = group_score / group_sizes.clamp_min(1.0)
        group_quality = flat_boxes.new_zeros(group_count)
        group_quality.scatter_add_(0, inverse, flat_qualities)
        group_quality = group_quality / group_sizes.clamp_min(1.0)
        group_observed = flat_boxes.new_zeros(group_count)
        group_observed.scatter_add_(0, inverse, flat_observed)
        group_observed = group_observed / group_sizes.clamp_min(1.0)

        group_reliability = (
            group_score
            * group_quality
            * (
                self.interpolated_weight
                + group_observed * (1.0 - self.interpolated_weight)
            )
        ).clamp_min(self.confidence_floor)
        reliability_sum = flat_boxes.new_zeros(batch_size)
        reliability_count = flat_boxes.new_zeros(batch_size)
        reliability_sum.scatter_add_(0, group_batch, group_reliability)
        reliability_count.scatter_add_(
            0, group_batch, torch.ones_like(group_reliability)
        )
        reliability_mean = (
            reliability_sum / reliability_count.clamp_min(1.0)
        )
        group_reliability = (
            group_reliability
            / reliability_mean.index_select(0, group_batch).clamp_min(1e-6)
        ).clamp(
            min=self.confidence_floor,
            max=1.0 / self.confidence_floor,
        )

        counts = torch.bincount(
            group_bt, minlength=batch_size * time_count
        )
        max_targets = int(counts.max().item())
        starts = counts.cumsum(dim=0) - counts
        local_indices = (
            torch.arange(group_count, device=device)
            - torch.repeat_interleave(starts, counts)
        )
        padded_labels = torch.zeros(
            batch_size * time_count, max_targets, 1,
            dtype=torch.long, device=device,
        )
        padded_boxes = boxes.new_zeros(
            batch_size * time_count, max_targets, 4
        )
        padded_mask = boxes.new_zeros(
            batch_size * time_count, max_targets, 1
        )
        padded_reliability = boxes.new_zeros(
            batch_size * time_count, max_targets
        )
        padded_track_ids = torch.full(
            (batch_size * time_count, max_targets), -1,
            dtype=torch.long, device=device,
        )
        padded_labels[group_bt, local_indices, 0] = group_labels
        padded_boxes[group_bt, local_indices] = group_boxes.to(boxes.dtype)
        padded_mask[group_bt, local_indices, 0] = 1.0
        padded_reliability[group_bt, local_indices] = group_reliability.to(
            boxes.dtype
        )
        padded_track_ids[group_bt, local_indices] = group_track_ids
        observed_groups = int((group_observed > 0.0).sum().item())
        return (
            padded_labels,
            padded_boxes,
            padded_mask,
            padded_reliability,
            padded_track_ids,
            int(group_count),
            observed_groups,
        )

    @staticmethod
    def _matched_reliability(
        target_boxes, gt_boxes, gt_mask, gt_reliability
    ):
        distance = (
            target_boxes[:, :, None].float()
            - gt_boxes[:, None].float()
        ).abs().sum(dim=-1)
        distance = distance.masked_fill(
            ~gt_mask.squeeze(-1).bool()[:, None], float("inf")
        )
        matched_indices = distance.argmin(dim=-1)
        return (
            gt_reliability.gather(1, matched_indices),
            matched_indices,
        )

    @staticmethod
    def _motion_representation(boxes):
        centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4).log()
        return torch.cat((centers, sizes), dim=-1)

    def _velocity_loss(
        self,
        decoded_boxes,
        foreground,
        matched_indices,
        positive_weight,
        gt_boxes,
        gt_mask,
        gt_reliability,
        gt_track_ids,
        batch_size,
        time_count,
        temporal_stride=1,
    ):
        """Match actor displacement across adjacent pyramid time bins."""
        zero = decoded_boxes.new_zeros((), dtype=torch.float32)
        target_slots = gt_boxes.shape[1]
        if time_count < 2 or target_slots == 0 or not foreground.any():
            return zero, 0

        bt_count, anchor_count = foreground.shape
        bt_indices = (
            torch.arange(bt_count, device=decoded_boxes.device)[:, None]
            .expand(bt_count, anchor_count)
        )
        flat_indices = (
            bt_indices[foreground] * target_slots
            + matched_indices[foreground]
        )
        selected_weight = positive_weight[foreground].float().clamp_min(
            self.confidence_floor
        )
        prediction_sum = decoded_boxes.new_zeros(
            bt_count * target_slots, 4, dtype=torch.float32
        ).index_add(
            0,
            flat_indices,
            decoded_boxes[foreground].float() * selected_weight[:, None],
        )
        weight_sum = decoded_boxes.new_zeros(
            bt_count * target_slots, dtype=torch.float32
        ).index_add(0, flat_indices, selected_weight)
        predicted_boxes = (
            prediction_sum / weight_sum[:, None].clamp_min(1e-6)
        ).reshape(batch_size, time_count, target_slots, 4)
        predicted_valid = (
            weight_sum.reshape(batch_size, time_count, target_slots) > 0
        )

        teacher_boxes = gt_boxes.float().reshape(
            batch_size, time_count, target_slots, 4
        )
        teacher_valid = gt_mask.squeeze(-1).bool().reshape(
            batch_size, time_count, target_slots
        )
        track_ids = gt_track_ids.reshape(
            batch_size, time_count, target_slots
        )
        reliability = gt_reliability.float().reshape(
            batch_size, time_count, target_slots
        )

        predicted_motion = self._motion_representation(predicted_boxes)
        teacher_motion = self._motion_representation(teacher_boxes)
        left_ids = track_ids[:, :-1, :, None]
        right_ids = track_ids[:, 1:, None, :]
        pair_mask = (
            (left_ids >= 0)
            & (left_ids == right_ids)
            & predicted_valid[:, :-1, :, None]
            & predicted_valid[:, 1:, None, :]
            & teacher_valid[:, :-1, :, None]
            & teacher_valid[:, 1:, None, :]
        )
        pair_count = int(pair_mask.sum().item())
        if not pair_count:
            return zero, 0

        predicted_delta = (
            predicted_motion[:, 1:, None, :, :]
            - predicted_motion[:, :-1, :, None, :]
        )
        teacher_delta = (
            teacher_motion[:, 1:, None, :, :]
            - teacher_motion[:, :-1, :, None, :]
        )
        if self.velocity_rate_normalize:
            elapsed = max(float(temporal_stride), 1.0)
            predicted_delta = predicted_delta / elapsed
            teacher_delta = teacher_delta / elapsed
        error = F.smooth_l1_loss(
            predicted_delta,
            teacher_delta,
            reduction="none",
            beta=0.02,
        ).mean(dim=-1)
        pair_weight = (
            reliability[:, :-1, :, None]
            * reliability[:, 1:, None, :]
        ).clamp_min(self.confidence_floor ** 2).sqrt()
        selected_weight = pair_weight[pair_mask]
        loss = (
            error[pair_mask] * selected_weight
        ).sum() / selected_weight.sum().clamp_min(1.0)
        return loss, pair_count

    def forward(
        self,
        model_outputs,
        targets,
        temporal_strides,
        spatial_strides,
        img_size,
    ):
        dense_outputs = _dense_outputs(model_outputs)
        if not dense_outputs:
            raise ValueError("offline dense-pyramid loss requires dense outputs")
        required = (
            "offline_boxes",
            "offline_labels",
            "offline_track_ids",
            "offline_scores",
            "offline_quality",
            "offline_observed",
        )
        if any(name not in targets for name in required):
            raise ValueError(
                "offline dense-pyramid targets require tracks, confidence, "
                "quality, and observed flags"
            )
        if (
            len(dense_outputs) != len(temporal_strides)
            or len(dense_outputs) != len(spatial_strides)
        ):
            raise ValueError("dense output and stride counts differ")
        if (
            self.velocity_scale_weights is not None
            and len(self.velocity_scale_weights) != len(dense_outputs)
        ):
            raise ValueError(
                "velocity scale weight count must match dense pyramid levels"
            )

        reference = dense_outputs[0][0]
        totals = {
            name: reference.new_zeros((), dtype=torch.float32)
            for name in self.weights
        }
        foreground_count = 0
        target_count = 0
        observed_target_count = 0
        contributing_scales = 0
        velocity_pair_count = 0
        velocity_scales = 0
        velocity_scale_weight_sum = 0.0

        for scale_index, (scale, temporal_stride, spatial_stride) in enumerate(
            zip(dense_outputs, temporal_strides, spatial_strides)
        ):
            class_logits, regression_logits, object_logits = scale[:3]
            batch_size, class_count, time_count, size, width = (
                class_logits.shape
            )
            if class_count != self.num_classes or width != size:
                raise ValueError("dense pyramid output has incompatible shape")
            (
                gt_labels,
                gt_boxes,
                gt_mask,
                gt_reliability,
                gt_track_ids,
                scale_target_count,
                scale_observed_count,
            ) = self._aggregate_targets(
                targets, time_count, temporal_stride
            )
            target_count += scale_target_count
            observed_target_count += scale_observed_count
            if not scale_target_count:
                continue

            anchor_count = size * size
            class_flat = class_logits.permute(
                0, 2, 3, 4, 1
            ).reshape(batch_size * time_count, anchor_count, class_count)
            regression_flat = regression_logits.permute(
                0, 2, 3, 4, 1
            ).reshape(batch_size * time_count, anchor_count, 4)
            object_flat = object_logits.permute(
                0, 2, 3, 4, 1
            ).reshape(batch_size * time_count, anchor_count)
            anchors = self._anchors(
                size, spatial_stride, img_size, class_logits.device
            )
            decoded_boxes = self._decode_boxes(
                regression_flat, anchors, spatial_stride, img_size
            )
            target_boxes, target_scores, foreground = self.assigner(
                class_flat.detach().float().sigmoid(),
                decoded_boxes.detach(),
                anchors,
                gt_labels,
                gt_boxes.float(),
                gt_mask,
            )
            if not foreground.any():
                continue

            reliability, matched_indices = self._matched_reliability(
                target_boxes, gt_boxes, gt_mask, gt_reliability
            )
            matched_labels = gt_labels.squeeze(-1).gather(
                1, matched_indices
            )
            alignment = target_scores.sum(dim=-1).detach().clamp_min(
                self.confidence_floor
            )
            positive_weight = reliability.float() * alignment.float()
            selected_weight = positive_weight[foreground]
            selected_weight = (
                selected_weight
                / selected_weight.mean().clamp_min(1e-6)
            ).clamp(
                min=self.confidence_floor,
                max=1.0 / self.confidence_floor,
            )
            denominator = selected_weight.sum().clamp_min(1.0)
            selected_classes = class_flat[foreground].float()
            selected_labels = matched_labels[foreground]
            positive_class_logits = selected_classes.gather(
                1, selected_labels[:, None]
            ).squeeze(1)
            totals["class"] = totals["class"] + (
                F.softplus(-positive_class_logits) * selected_weight
            ).sum() / denominator
            totals["object"] = totals["object"] + (
                F.softplus(-object_flat[foreground].float())
                * selected_weight
            ).sum() / denominator

            selected_boxes = decoded_boxes[foreground]
            selected_targets = target_boxes[foreground].float()
            smooth_l1 = F.smooth_l1_loss(
                selected_boxes,
                selected_targets,
                reduction="none",
                beta=0.05,
            ).mean(dim=-1)
            ciou = bbox_iou_ciou(selected_boxes, selected_targets)
            totals["box"] = totals["box"] + (
                smooth_l1 * selected_weight
            ).sum() / denominator
            totals["giou"] = totals["giou"] + (
                (1.0 - ciou) * selected_weight
            ).sum() / denominator
            velocity_scale_weight = (
                1.0
                if self.velocity_scale_weights is None
                else self.velocity_scale_weights[scale_index]
            )
            if self.weights["velocity"] > 0 and velocity_scale_weight > 0:
                velocity_loss, velocity_pairs = self._velocity_loss(
                    decoded_boxes,
                    foreground,
                    matched_indices,
                    positive_weight,
                    gt_boxes,
                    gt_mask,
                    gt_reliability,
                    gt_track_ids,
                    batch_size,
                    time_count,
                    temporal_stride,
                )
                if velocity_pairs:
                    totals["velocity"] = (
                        totals["velocity"]
                        + velocity_loss * velocity_scale_weight
                    )
                    velocity_pair_count += velocity_pairs
                    velocity_scales += 1
                    velocity_scale_weight_sum += velocity_scale_weight
            foreground_count += int(foreground.sum().item())
            contributing_scales += 1

        if contributing_scales:
            totals = {
                name: (
                    value
                    if name == "velocity"
                    else value / contributing_scales
                )
                for name, value in totals.items()
            }
        if velocity_scale_weight_sum:
            totals["velocity"] = (
                totals["velocity"] / velocity_scale_weight_sum
            )
        total = sum(
            self.weights[name] * value for name, value in totals.items()
        )
        metrics = {
            "offline_dense_loss": float(total.detach().item()),
            "offline_dense_class": float(totals["class"].detach().item()),
            "offline_dense_object": float(totals["object"].detach().item()),
            "offline_dense_box": float(totals["box"].detach().item()),
            "offline_dense_giou": float(totals["giou"].detach().item()),
            "offline_dense_velocity": float(
                totals["velocity"].detach().item()
            ),
            "offline_dense_velocity_pairs": velocity_pair_count,
            "offline_dense_velocity_scales": velocity_scales,
            "offline_dense_velocity_rate_normalized": (
                self.velocity_rate_normalize
            ),
            "offline_dense_targets": target_count,
            "offline_dense_observed_targets": observed_target_count,
            "offline_dense_foreground": foreground_count,
            "offline_dense_scales": contributing_scales,
        }
        return total, metrics
