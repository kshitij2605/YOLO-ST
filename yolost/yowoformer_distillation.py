"""Pyramid-aligned distillation from YOWOFormer into dense YOLO-ST outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolost.loss import bbox_iou_ciou

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy is a training dependency
    linear_sum_assignment = None


def _dense_outputs(outputs):
    return outputs["dense"] if isinstance(outputs, dict) else outputs


def _bernoulli_kl_from_probability(student_logits, teacher_probability,
                                   temperature):
    # FP16 cannot represent 1 - 1e-6, so clamping before promotion can leave
    # an exact probability of one and make torch.logit return infinity.
    student_logits = student_logits.float()
    probability = teacher_probability.detach().float().clamp(
        1e-6, 1.0 - 1e-6
    )
    teacher_logits = torch.logit(probability)
    softened_probability = (teacher_logits / temperature).sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        student_logits / temperature, softened_probability, reduction="none"
    )
    teacher_entropy = F.binary_cross_entropy_with_logits(
        teacher_logits / temperature, softened_probability, reduction="none"
    )
    return (cross_entropy - teacher_entropy) * (temperature ** 2)


def _xywh_to_xyxy(boxes):
    center, extent = boxes.split(2, dim=-1)
    half_extent = extent / 2
    return torch.cat((center - half_extent, center + half_extent), dim=-1)


def _box_iou_matrix(first, second):
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp_min(1e-8)


def _class_aware_nms(boxes, scores, labels, threshold, max_detections):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    retained = []
    while order.numel() and len(retained) < max_detections:
        current = order[0]
        retained.append(current)
        order = order[1:]
        if not order.numel():
            break
        overlap = _box_iou_matrix(
            boxes.index_select(0, current.reshape(1)),
            boxes.index_select(0, order),
        )[0]
        same_class = labels.index_select(0, order) == labels[current]
        order = order[~(same_class & (overlap > threshold))]
    return torch.stack(retained)


class PyramidAlignedYOWOFormerDistillationLoss(nn.Module):
    """Align decoded YOWOFormer cells with YOLO-ST cells at selected times.

    Both detectors use P3/P4/P5 grids at the same 224-pixel resolution. The
    frozen teacher emits decoded boxes and class probabilities for one keyframe;
    the student supplies dense logits at the corresponding temporal index.
    """

    def __init__(self, num_classes=24, class_weight=1.0, object_weight=1.0,
                 box_weight=1.0, temperature=2.0,
                 min_teacher_confidence=0.03, background_floor=0.01,
                 box_ciou_weight=1.0, query_class_weight=0.0,
                 query_box_weight=0.0, query_visibility_weight=0.0,
                 query_max_detections=16, query_nms_iou=0.5,
                 query_track_iou=0.3, query_match_class_cost=2.0,
                 query_match_box_cost=5.0,
                 query_match_visibility_cost=1.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_weight = float(class_weight)
        self.object_weight = float(object_weight)
        self.box_weight = float(box_weight)
        self.temperature = float(temperature)
        self.min_teacher_confidence = float(min_teacher_confidence)
        self.background_floor = float(background_floor)
        self.box_ciou_weight = float(box_ciou_weight)
        self.query_class_weight = float(query_class_weight)
        self.query_box_weight = float(query_box_weight)
        self.query_visibility_weight = float(query_visibility_weight)
        self.query_max_detections = int(query_max_detections)
        self.query_nms_iou = float(query_nms_iou)
        self.query_track_iou = float(query_track_iou)
        self.query_match_class_cost = float(query_match_class_cost)
        self.query_match_box_cost = float(query_match_box_cost)
        self.query_match_visibility_cost = float(query_match_visibility_cost)
        if min(
            self.class_weight, self.object_weight, self.box_weight,
            self.background_floor, self.box_ciou_weight,
            self.query_class_weight, self.query_box_weight,
            self.query_visibility_weight,
        ) < 0:
            raise ValueError("YOWOFormer distillation weights must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.query_max_detections < 1:
            raise ValueError("query_max_detections must be positive")
        if not 0 <= self.query_nms_iou <= 1 or not 0 <= self.query_track_iou <= 1:
            raise ValueError("query NMS and tracking IoUs must lie in [0, 1]")

    def _teacher_detections(self, teacher_view, batch_index, img_size):
        teacher = teacher_view["outputs"][batch_index].detach()
        box_values = teacher[:4].transpose(0, 1)
        class_values = teacher[4:].transpose(0, 1)
        finite = (
            torch.isfinite(box_values).all(dim=-1)
            & torch.isfinite(class_values).all(dim=-1)
        )
        boxes = _xywh_to_xyxy(
            torch.nan_to_num(
                box_values, nan=0.0, posinf=float(img_size), neginf=0.0
            ) / float(img_size)
        ).clamp(0.0, 1.0)
        classes = torch.nan_to_num(
            class_values, nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)
        scores, labels = classes.max(dim=-1)
        extent = boxes[:, 2:] - boxes[:, :2]
        valid = (
            finite
            & (scores >= self.min_teacher_confidence)
            & (extent > 1e-4).all(dim=-1)
        )
        indices = torch.nonzero(valid, as_tuple=True)[0]
        if not indices.numel():
            return {
                "boxes": boxes[:0], "classes": classes[:0],
                "scores": scores[:0], "labels": labels[:0],
            }
        pre_nms_limit = min(indices.numel(), self.query_max_detections * 16)
        relative = scores.index_select(0, indices).topk(pre_nms_limit).indices
        indices = indices.index_select(0, relative)
        selected_boxes = boxes.index_select(0, indices)
        selected_classes = classes.index_select(0, indices)
        selected_scores = scores.index_select(0, indices)
        selected_labels = labels.index_select(0, indices)
        keep = _class_aware_nms(
            selected_boxes, selected_scores, selected_labels,
            self.query_nms_iou, self.query_max_detections,
        )
        return {
            "boxes": selected_boxes.index_select(0, keep),
            "classes": selected_classes.index_select(0, keep),
            "scores": selected_scores.index_select(0, keep),
            "labels": selected_labels.index_select(0, keep),
        }

    def _teacher_tracks(self, teacher_views, batch_index, img_size):
        tracks = []
        for view in sorted(teacher_views, key=lambda item: int(item["endpoint"])):
            endpoint = int(view["endpoint"])
            detections = self._teacher_detections(view, batch_index, img_size)
            assigned_tracks = set()
            for detection_index in detections["scores"].argsort(descending=True):
                detection_index = int(detection_index.item())
                label = int(detections["labels"][detection_index].item())
                box = detections["boxes"][detection_index]
                best_track = None
                best_overlap = self.query_track_iou
                for track_index, track in enumerate(tracks):
                    if track_index in assigned_tracks or track["label"] != label:
                        continue
                    overlap = float(_box_iou_matrix(
                        track["observations"][-1][1][None], box[None]
                    )[0, 0].item())
                    if overlap >= best_overlap:
                        best_overlap = overlap
                        best_track = track_index
                observation = (
                    endpoint,
                    box,
                    detections["classes"][detection_index],
                    detections["scores"][detection_index],
                )
                if best_track is None:
                    tracks.append({"label": label, "observations": [observation]})
                    assigned_tracks.add(len(tracks) - 1)
                else:
                    tracks[best_track]["observations"].append(observation)
                    assigned_tracks.add(best_track)
        tracks.sort(
            key=lambda track: max(
                float(observation[3].item())
                for observation in track["observations"]
            ),
            reverse=True,
        )
        return tracks[:self.query_max_detections]

    @staticmethod
    def _query_time_index(endpoint, query_frames, clip_length):
        return min(
            query_frames - 1,
            int(endpoint) * query_frames // max(int(clip_length), 1),
        )

    def _query_trajectory_loss(self, student_outputs, teacher_views,
                               img_size, clip_length):
        dense = _dense_outputs(student_outputs)
        zero = dense[0][0].new_zeros(())
        totals = {"class": zero, "box": zero, "visibility": zero}
        query_output = (
            student_outputs.get("tube_queries")
            if isinstance(student_outputs, dict) else None
        )
        enabled = any((
            self.query_class_weight, self.query_box_weight,
            self.query_visibility_weight,
        ))
        if not enabled:
            return totals, 0, 0
        if query_output is None:
            raise ValueError("query trajectory distillation requires tube queries")
        required = ("class_logits", "boxes", "visibility_logits")
        if any(name not in query_output for name in required):
            raise ValueError("student tube-query trajectory output is incomplete")
        if linear_sum_assignment is None:
            raise RuntimeError("scipy is required for query trajectory matching")

        class_logits = query_output["class_logits"]
        boxes = query_output["boxes"]
        visibility_logits = query_output["visibility_logits"]
        matched_weight = zero
        track_count = 0
        match_count = 0
        for batch_index in range(class_logits.shape[0]):
            tracks = self._teacher_tracks(
                teacher_views, batch_index, img_size
            )
            track_count += len(tracks)
            if not tracks:
                continue
            track_classes = []
            for track in tracks:
                weights = torch.stack([
                    observation[3] for observation in track["observations"]
                ])
                probabilities = torch.stack([
                    observation[2] for observation in track["observations"]
                ])
                track_classes.append(
                    (probabilities * weights[:, None]).sum(dim=0)
                    / weights.sum().clamp_min(1e-6)
                )
            track_classes = torch.stack(track_classes)
            student_probability = class_logits[batch_index].sigmoid()
            class_cost = (
                student_probability[:, None] - track_classes[None]
            ).square().mean(dim=-1)
            box_cost_columns = []
            visibility_cost_columns = []
            for track in tracks:
                observation_box_costs = []
                observation_visibility_costs = []
                observation_weights = []
                for endpoint, teacher_box, _, confidence in track["observations"]:
                    time_index = self._query_time_index(
                        endpoint, boxes.shape[2], clip_length
                    )
                    observation_box_costs.append(
                        (boxes[batch_index, :, time_index] - teacher_box).abs().mean(
                            dim=-1
                        )
                    )
                    observation_visibility_costs.append(
                        F.softplus(-visibility_logits[batch_index, :, time_index])
                    )
                    observation_weights.append(confidence)
                weights = torch.stack(observation_weights)
                box_cost_columns.append(
                    (torch.stack(observation_box_costs, dim=1) * weights[None]).sum(
                        dim=1
                    ) / weights.sum().clamp_min(1e-6)
                )
                visibility_cost_columns.append(
                    (
                        torch.stack(observation_visibility_costs, dim=1)
                        * weights[None]
                    ).sum(dim=1) / weights.sum().clamp_min(1e-6)
                )
            box_cost = torch.stack(box_cost_columns, dim=1)
            visibility_cost = torch.stack(visibility_cost_columns, dim=1)
            cost = (
                self.query_match_class_cost * class_cost
                + self.query_match_box_cost * box_cost
                + self.query_match_visibility_cost * visibility_cost
            )
            rows, columns = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
            for student_index, track_index in zip(rows, columns):
                track = tracks[int(track_index)]
                pair_weight = torch.stack([
                    observation[3] for observation in track["observations"]
                ]).mean().clamp_min(1e-4)
                matched_weight = matched_weight + pair_weight
                match_count += 1
                totals["class"] = totals["class"] + pair_weight * (
                    _bernoulli_kl_from_probability(
                        class_logits[batch_index, int(student_index)],
                        track_classes[int(track_index)], self.temperature,
                    ).mean()
                )
                observation_box_losses = []
                observation_visibility_losses = []
                observation_weights = []
                for endpoint, teacher_box, _, confidence in track["observations"]:
                    time_index = self._query_time_index(
                        endpoint, boxes.shape[2], clip_length
                    )
                    student_box = boxes[
                        batch_index, int(student_index), time_index
                    ].float()
                    teacher_box = teacher_box.float()
                    smooth_l1 = F.smooth_l1_loss(
                        student_box, teacher_box, reduction="mean", beta=0.05
                    )
                    ciou = bbox_iou_ciou(
                        student_box[None], teacher_box[None]
                    )[0]
                    observation_box_losses.append(
                        smooth_l1 + self.box_ciou_weight * (1.0 - ciou)
                    )
                    observation_visibility_losses.append(
                        F.softplus(-visibility_logits[
                            batch_index, int(student_index), time_index
                        ].float())
                    )
                    observation_weights.append(confidence.float())
                weights = torch.stack(observation_weights)
                totals["box"] = totals["box"] + pair_weight * (
                    (torch.stack(observation_box_losses) * weights).sum()
                    / weights.sum().clamp_min(1e-6)
                )
                totals["visibility"] = totals["visibility"] + pair_weight * (
                    (torch.stack(observation_visibility_losses) * weights).sum()
                    / weights.sum().clamp_min(1e-6)
                )
        denominator = matched_weight.clamp_min(1.0)
        return (
            {name: value / denominator for name, value in totals.items()},
            track_count,
            match_count,
        )

    @staticmethod
    def _anchors(size, spatial_stride, img_size, device, dtype):
        step = float(spatial_stride) / float(img_size)
        coordinates = (
            torch.arange(size, device=device, dtype=dtype) + 0.5
        ) * step
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)

    def _decode_student_boxes(self, regression, size, spatial_stride,
                              img_size):
        anchors = self._anchors(
            size, spatial_stride, img_size,
            regression.device, regression.dtype,
        )
        step = float(spatial_stride) / float(img_size)
        center_x = anchors[:, 0] + (regression[..., 0].sigmoid() - 0.5) * step
        center_y = anchors[:, 1] + (regression[..., 1].sigmoid() - 0.5) * step
        width = regression[..., 2].clamp(max=5.0).exp() * step
        height = regression[..., 3].clamp(max=5.0).exp() * step
        return torch.stack(
            (
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            ),
            dim=-1,
        )

    def forward(self, student_outputs, teacher_views, temporal_strides,
                spatial_strides, img_size, clip_length):
        dense_outputs = _dense_outputs(student_outputs)
        if not dense_outputs:
            raise ValueError("YOLO-ST student has no dense pyramid outputs")
        if len(dense_outputs) != len(temporal_strides):
            raise ValueError("temporal stride count does not match student pyramid")
        if len(dense_outputs) != len(spatial_strides):
            raise ValueError("spatial stride count does not match student pyramid")
        if not teacher_views:
            raise ValueError("At least one YOWOFormer teacher view is required")

        reference = dense_outputs[0][0]
        totals = {
            "class": reference.new_zeros(()),
            "object": reference.new_zeros(()),
            "box": reference.new_zeros(()),
        }
        scale_view_count = 0
        box_term_count = 0
        teacher_positive_count = 0
        teacher_invalid_count = 0
        expected_anchors = sum(scale[0].shape[-1] ** 2 for scale in dense_outputs)

        for view in teacher_views:
            endpoint = int(view["endpoint"])
            if endpoint < 0 or endpoint >= int(clip_length):
                raise ValueError(
                    f"Teacher endpoint {endpoint} is outside clip length {clip_length}"
                )
            teacher = view["outputs"].detach()
            if teacher.ndim != 3:
                raise ValueError(
                    "YOWOFormer output must have shape (B, 4 + classes, anchors)"
                )
            if teacher.shape[1] != 4 + self.num_classes:
                raise ValueError(
                    f"YOWOFormer output has {teacher.shape[1] - 4} classes; "
                    f"expected {self.num_classes}"
                )
            if teacher.shape[2] != expected_anchors:
                raise ValueError(
                    f"YOWOFormer output has {teacher.shape[2]} anchors; "
                    f"student pyramid requires {expected_anchors}"
                )
            if teacher.shape[0] != reference.shape[0]:
                raise ValueError("Teacher and student batch sizes differ")

            teacher_box_values = teacher[:, :4].transpose(1, 2)
            teacher_class_values = teacher[:, 4:].transpose(1, 2)
            teacher_box_finite = torch.isfinite(teacher_box_values).all(dim=-1)
            teacher_class_finite = torch.isfinite(teacher_class_values).all(
                dim=-1
            )
            teacher_invalid_count += int(
                (~(teacher_box_finite & teacher_class_finite)).sum().item()
            )
            teacher_boxes = _xywh_to_xyxy(
                torch.nan_to_num(
                    teacher_box_values,
                    nan=0.0,
                    posinf=float(img_size),
                    neginf=0.0,
                ) / float(img_size)
            ).clamp(0.0, 1.0)
            teacher_classes = torch.nan_to_num(
                teacher_class_values,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            offset = 0
            for scale_index, scale in enumerate(dense_outputs):
                if len(scale) < 3:
                    raise ValueError(
                        "Dense student outputs require class, box, and object tensors"
                    )
                class_logits, regression_logits, object_logits = scale[:3]
                batch_size, class_count, time_count, size, width = class_logits.shape
                if width != size or class_count != self.num_classes:
                    raise ValueError("Student dense pyramid has incompatible shape")
                anchor_count = size * size
                stop = offset + anchor_count
                teacher_scale_classes = teacher_classes[:, offset:stop]
                teacher_scale_boxes = teacher_boxes[:, offset:stop]
                teacher_scale_box_finite = teacher_box_finite[:, offset:stop]
                offset = stop

                time_index = min(
                    time_count - 1,
                    endpoint // int(temporal_strides[scale_index]),
                )
                student_classes = class_logits[:, :, time_index].permute(
                    0, 2, 3, 1
                ).reshape(batch_size, anchor_count, class_count)
                student_regression = regression_logits[:, :, time_index].permute(
                    0, 2, 3, 1
                ).reshape(batch_size, anchor_count, 4)
                student_object = object_logits[:, :, time_index].permute(
                    0, 2, 3, 1
                ).reshape(batch_size, anchor_count)
                student_boxes = self._decode_student_boxes(
                    student_regression, size,
                    spatial_strides[scale_index], img_size,
                )

                confidence = teacher_scale_classes.amax(dim=-1)
                class_weights = confidence.clamp_min(self.background_floor)
                class_kl = _bernoulli_kl_from_probability(
                    student_classes,
                    teacher_scale_classes,
                    self.temperature,
                ).mean(dim=-1)
                totals["class"] = totals["class"] + (
                    class_kl * class_weights
                ).sum() / class_weights.sum().clamp_min(1.0)
                totals["object"] = totals["object"] + (
                    F.binary_cross_entropy_with_logits(
                        student_object.float(), confidence.float(),
                        reduction="mean",
                    )
                )

                positive = (
                    (confidence >= self.min_teacher_confidence)
                    & teacher_scale_box_finite
                )
                positive_count = int(positive.sum().item())
                teacher_positive_count += positive_count
                if positive_count:
                    selected_student = student_boxes[positive]
                    selected_teacher = teacher_scale_boxes[positive]
                    selected_weight = confidence[positive]
                    # CIoU contains divisions by small box extents and is not
                    # numerically stable in fp16 for low-confidence cells.
                    student_geometry = selected_student.float()
                    teacher_geometry = selected_teacher.float()
                    geometry_weight = selected_weight.float()
                    smooth_l1 = F.smooth_l1_loss(
                        student_geometry,
                        teacher_geometry,
                        reduction="none",
                        beta=0.05,
                    ).mean(dim=-1)
                    ciou = bbox_iou_ciou(
                        student_geometry, teacher_geometry
                    )
                    box_loss = smooth_l1 + self.box_ciou_weight * (1.0 - ciou)
                    totals["box"] = totals["box"] + (
                        box_loss * geometry_weight
                    ).sum() / geometry_weight.sum().clamp_min(1e-6)
                    box_term_count += 1
                scale_view_count += 1

        denominator = max(scale_view_count, 1)
        totals["class"] = totals["class"] / denominator
        totals["object"] = totals["object"] / denominator
        if box_term_count:
            totals["box"] = totals["box"] / box_term_count

        query_totals, teacher_track_count, teacher_match_count = (
            self._query_trajectory_loss(
                student_outputs, teacher_views, img_size, clip_length
            )
        )

        loss = (
            self.class_weight * totals["class"]
            + self.object_weight * totals["object"]
            + self.box_weight * totals["box"]
            + self.query_class_weight * query_totals["class"]
            + self.query_box_weight * query_totals["box"]
            + self.query_visibility_weight * query_totals["visibility"]
        )
        metrics = {
            "yowo_distill_class": float(totals["class"].detach().item()),
            "yowo_distill_object": float(totals["object"].detach().item()),
            "yowo_distill_box": float(totals["box"].detach().item()),
            "yowo_distill_query_class": float(
                query_totals["class"].detach().item()
            ),
            "yowo_distill_query_box": float(
                query_totals["box"].detach().item()
            ),
            "yowo_distill_query_visibility": float(
                query_totals["visibility"].detach().item()
            ),
            "yowo_distill_loss": float(loss.detach().item()),
            "yowo_teacher_positives": teacher_positive_count,
            "yowo_teacher_invalid_cells": teacher_invalid_count,
            "yowo_teacher_tracks": teacher_track_count,
            "yowo_teacher_query_matches": teacher_match_count,
        }
        return loss, metrics
