"""Permutation-invariant distillation losses for YOLO-ST outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - training already requires scipy
    linear_sum_assignment = None


COMPONENTS = (
    "dense_cls",
    "dense_box",
    "dense_obj",
    "dense_boundary",
    "query_cls",
    "query_box",
    "query_visibility",
    "query_boundary",
    "query_start",
    "query_end",
)


def _bernoulli_kl(student_logits, teacher_logits, temperature=1.0):
    """Bernoulli KL with gradients only through the student logits."""
    teacher_logits = teacher_logits.detach()
    teacher_probability = (teacher_logits / temperature).sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        student_logits / temperature, teacher_probability, reduction="none"
    )
    teacher_entropy = F.binary_cross_entropy_with_logits(
        teacher_logits / temperature, teacher_probability, reduction="none"
    )
    return (cross_entropy - teacher_entropy) * (temperature ** 2)


def _categorical_kl(student_logits, teacher_logits, temperature=1.0):
    teacher_logits = teacher_logits.detach()
    teacher_probability = (teacher_logits / temperature).softmax(dim=-1)
    teacher_log_probability = teacher_probability.clamp_min(1e-8).log()
    student_log_probability = (student_logits / temperature).log_softmax(dim=-1)
    return (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1) * (temperature ** 2)


def _dense_outputs(outputs):
    return outputs["dense"] if isinstance(outputs, dict) else outputs


def _query_outputs(outputs):
    return outputs.get("tube_queries") if isinstance(outputs, dict) else None


def _selected_classes(logits, class_ids):
    if class_ids is None:
        return logits
    indices = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
    return logits.index_select(-1, indices)


class YOLOSTDistillationLoss(nn.Module):
    """Distill dense maps and unordered tube queries from frozen teachers."""

    def __init__(self, component_weights=None, temperature=1.0,
                 min_teacher_confidence=0.05, max_teacher_queries=16,
                 match_class_cost=2.0, match_box_cost=5.0,
                 match_visibility_cost=1.0, dense_positive_floor=0.05):
        super().__init__()
        weights = component_weights or {}
        self.component_weights = {
            name: float(weights.get(name, 0.0)) for name in COMPONENTS
        }
        self.temperature = float(temperature)
        self.min_teacher_confidence = float(min_teacher_confidence)
        self.max_teacher_queries = int(max_teacher_queries)
        self.match_class_cost = float(match_class_cost)
        self.match_box_cost = float(match_box_cost)
        self.match_visibility_cost = float(match_visibility_cost)
        self.dense_positive_floor = float(dense_positive_floor)

    @staticmethod
    def _zero(outputs):
        dense = _dense_outputs(outputs)
        return dense[0][0].new_zeros(())

    def _dense_loss(self, student_outputs, teacher_outputs, class_ids):
        student_dense = _dense_outputs(student_outputs)
        teacher_dense = _dense_outputs(teacher_outputs)
        if len(student_dense) != len(teacher_dense):
            raise ValueError("Student and teacher dense pyramid depths differ")

        totals = {name: self._zero(student_outputs) for name in COMPONENTS[:4]}
        for student_scale, teacher_scale in zip(student_dense, teacher_dense):
            if len(student_scale) < 3 or len(teacher_scale) < 3:
                raise ValueError("Dense outputs require class, box, and objectness tensors")
            student_cls, student_box, student_obj = student_scale[:3]
            teacher_cls, teacher_box, teacher_obj = teacher_scale[:3]
            if student_cls.shape != teacher_cls.shape:
                raise ValueError("Student and teacher dense class shapes differ")
            if student_box.shape != teacher_box.shape or student_obj.shape != teacher_obj.shape:
                raise ValueError("Student and teacher dense geometry shapes differ")

            selected_student_cls = _selected_classes(
                student_cls.movedim(1, -1), class_ids
            )
            selected_teacher_cls = _selected_classes(
                teacher_cls.movedim(1, -1), class_ids
            )
            teacher_objectness = teacher_obj.detach().sigmoid()
            positive_weight = teacher_objectness.clamp_min(self.dense_positive_floor)
            cls_kl = _bernoulli_kl(
                selected_student_cls, selected_teacher_cls, self.temperature
            ).mean(dim=-1, keepdim=True).movedim(-1, 1)
            totals["dense_cls"] = totals["dense_cls"] + (
                cls_kl * positive_weight
            ).sum() / positive_weight.sum().clamp_min(1.0)

            totals["dense_obj"] = totals["dense_obj"] + _bernoulli_kl(
                student_obj, teacher_obj, self.temperature
            ).mean()

            selected_teacher_probability = selected_teacher_cls.detach().sigmoid()
            class_confidence = selected_teacher_probability.amax(
                dim=-1, keepdim=True
            ).movedim(-1, 1)
            geometry_weight = teacher_objectness * class_confidence
            box_delta = F.smooth_l1_loss(
                student_box, teacher_box.detach(), reduction="none", beta=0.05
            ).mean(dim=1, keepdim=True)
            totals["dense_box"] = totals["dense_box"] + (
                box_delta * geometry_weight
            ).sum() / geometry_weight.sum().clamp_min(1.0)

            if len(student_scale) > 3 and len(teacher_scale) > 3:
                boundary_kl = _bernoulli_kl(
                    student_scale[3], teacher_scale[3], self.temperature
                )
                totals["dense_boundary"] = totals["dense_boundary"] + (
                    boundary_kl * positive_weight
                ).sum() / positive_weight.sum().clamp_min(1.0)

        scale_count = max(len(student_dense), 1)
        return {name: value / scale_count for name, value in totals.items()}

    def _match_queries(self, student, teacher, batch_index, class_ids):
        if linear_sum_assignment is None:
            raise RuntimeError("scipy is required for query distillation")

        student_cls = _selected_classes(
            student["class_logits"][batch_index], class_ids
        )
        teacher_cls = _selected_classes(
            teacher["class_logits"][batch_index], class_ids
        ).detach()
        student_visibility = student["visibility_logits"][batch_index]
        teacher_visibility = teacher["visibility_logits"][batch_index].detach()
        teacher_class_probability = teacher_cls.sigmoid()
        teacher_visibility_probability = teacher_visibility.sigmoid()
        confidence = (
            teacher_class_probability.amax(dim=-1)
            * teacher_visibility_probability.mean(dim=-1)
        )
        keep = torch.nonzero(
            confidence >= self.min_teacher_confidence, as_tuple=True
        )[0]
        if keep.numel() == 0:
            keep = confidence.topk(min(1, confidence.numel())).indices
        if keep.numel() > self.max_teacher_queries:
            relative = confidence.index_select(0, keep).topk(
                self.max_teacher_queries
            ).indices
            keep = keep.index_select(0, relative)

        teacher_cls = teacher_cls.index_select(0, keep)
        teacher_visibility = teacher_visibility.index_select(0, keep)
        teacher_visibility_probability = teacher_visibility.sigmoid()
        teacher_boxes = teacher["boxes"][batch_index].detach().index_select(0, keep)

        student_probability = student_cls.sigmoid()
        teacher_probability = teacher_cls.sigmoid()
        class_cost = (
            student_probability[:, None, :] - teacher_probability[None, :, :]
        ).square().mean(dim=-1)

        visibility_cost = F.binary_cross_entropy_with_logits(
            student_visibility[:, None, :].expand(
                -1, keep.numel(), -1
            ),
            teacher_visibility_probability[None, :, :].expand(
                student_visibility.shape[0], -1, -1
            ),
            reduction="none",
        ).mean(dim=-1)

        box_delta = (
            student["boxes"][batch_index][:, None, :, :]
            - teacher_boxes[None, :, :, :]
        ).abs().mean(dim=-1)
        box_cost = (
            box_delta * teacher_visibility_probability[None, :, :]
        ).sum(dim=-1) / teacher_visibility_probability.sum(
            dim=-1
        ).clamp_min(1e-4)[None, :]

        cost = (
            self.match_class_cost * class_cost
            + self.match_box_cost * box_cost
            + self.match_visibility_cost * visibility_cost
        )
        rows, columns = linear_sum_assignment(
            cost.detach().float().cpu().numpy()
        )
        student_indices = torch.as_tensor(
            rows, dtype=torch.long, device=cost.device
        )
        teacher_relative_indices = torch.as_tensor(
            columns, dtype=torch.long, device=cost.device
        )
        teacher_indices = keep.index_select(0, teacher_relative_indices)
        return student_indices, teacher_indices, confidence.index_select(
            0, teacher_indices
        ).detach()

    def _query_loss(self, student_outputs, teacher_outputs, class_ids):
        student = _query_outputs(student_outputs)
        teacher = _query_outputs(teacher_outputs)
        totals = {
            name: self._zero(student_outputs) for name in COMPONENTS[4:]
        }
        if student is None or teacher is None:
            return totals
        required = ("class_logits", "boxes", "visibility_logits", "boundary_logits")
        if any(name not in student or name not in teacher for name in required):
            raise ValueError("Tube-query distillation outputs are incomplete")

        matched_weight = self._zero(student_outputs)
        batch_size = student["class_logits"].shape[0]
        for batch_index in range(batch_size):
            student_indices, teacher_indices, confidence = self._match_queries(
                student, teacher, batch_index, class_ids
            )
            if student_indices.numel() == 0:
                continue
            pair_weight = confidence.clamp_min(1e-4)
            matched_weight = matched_weight + pair_weight.sum()

            student_cls = _selected_classes(
                student["class_logits"][batch_index].index_select(
                    0, student_indices
                ),
                class_ids,
            )
            teacher_cls = _selected_classes(
                teacher["class_logits"][batch_index].index_select(
                    0, teacher_indices
                ),
                class_ids,
            )
            cls_loss = _bernoulli_kl(
                student_cls, teacher_cls, self.temperature
            ).mean(dim=-1)
            totals["query_cls"] = totals["query_cls"] + (
                cls_loss * pair_weight
            ).sum()

            student_visibility = student["visibility_logits"][batch_index].index_select(
                0, student_indices
            )
            teacher_visibility = teacher["visibility_logits"][batch_index].index_select(
                0, teacher_indices
            )
            visibility_loss = _bernoulli_kl(
                student_visibility, teacher_visibility, self.temperature
            ).mean(dim=-1)
            totals["query_visibility"] = totals["query_visibility"] + (
                visibility_loss * pair_weight
            ).sum()

            student_boundary = student["boundary_logits"][batch_index].index_select(
                0, student_indices
            )
            teacher_boundary = teacher["boundary_logits"][batch_index].index_select(
                0, teacher_indices
            )
            boundary_loss = _bernoulli_kl(
                student_boundary, teacher_boundary, self.temperature
            ).mean(dim=-1)
            totals["query_boundary"] = totals["query_boundary"] + (
                boundary_loss * pair_weight
            ).sum()

            visibility_weight = teacher_visibility.detach().sigmoid()
            box_loss = F.smooth_l1_loss(
                student["boxes"][batch_index].index_select(0, student_indices),
                teacher["boxes"][batch_index].index_select(
                    0, teacher_indices
                ).detach(),
                reduction="none",
                beta=0.05,
            ).mean(dim=-1)
            box_loss = (
                box_loss * visibility_weight
            ).sum(dim=-1) / visibility_weight.sum(dim=-1).clamp_min(1e-4)
            totals["query_box"] = totals["query_box"] + (
                box_loss * pair_weight
            ).sum()

            for name, output_name in (
                ("query_start", "start_logits"),
                ("query_end", "end_logits"),
            ):
                if output_name not in student or output_name not in teacher:
                    continue
                endpoint_loss = _categorical_kl(
                    student[output_name][batch_index].index_select(
                        0, student_indices
                    ),
                    teacher[output_name][batch_index].index_select(
                        0, teacher_indices
                    ),
                    self.temperature,
                )
                totals[name] = totals[name] + (
                    endpoint_loss * pair_weight
                ).sum()

        denominator = matched_weight.clamp_min(1.0)
        return {name: value / denominator for name, value in totals.items()}

    def forward(self, student_outputs, teachers):
        """Return weighted distillation loss and detached component metrics.

        Each teacher entry must contain ``outputs`` and may contain ``weight``
        and ``class_ids``. A teacher may also define ``component_weights``;
        when present, omitted components are disabled for that teacher. This
        allows role-separated teachers (for example, semantics from one model
        and trajectory geometry from another) without mixing their errors.
        """
        if not teachers:
            raise ValueError("At least one teacher output is required")
        totals = {name: self._zero(student_outputs) for name in COMPONENTS}
        component_denominators = {name: 0.0 for name in COMPONENTS}
        for teacher in teachers:
            weight = float(teacher.get("weight", 1.0))
            if weight <= 0:
                continue
            role_weights = teacher.get("component_weights")
            if role_weights is None:
                role_weights = {name: 1.0 for name in COMPONENTS}
            else:
                unknown = set(role_weights) - set(COMPONENTS)
                if unknown:
                    raise ValueError(
                        "Unknown distillation components: "
                        + ", ".join(sorted(unknown))
                    )
                role_weights = {
                    name: float(role_weights.get(name, 0.0))
                    for name in COMPONENTS
                }
                if any(value < 0 for value in role_weights.values()):
                    raise ValueError(
                        "Teacher component weights must be non-negative"
                    )
            teacher_outputs = teacher["outputs"]
            class_ids = teacher.get("class_ids")
            components = self._dense_loss(
                student_outputs, teacher_outputs, class_ids
            )
            components.update(self._query_loss(
                student_outputs, teacher_outputs, class_ids
            ))
            for name in COMPONENTS:
                role_weight = role_weights[name]
                if role_weight <= 0:
                    continue
                effective_weight = weight * role_weight
                totals[name] = (
                    totals[name] + effective_weight * components[name]
                )
                component_denominators[name] += effective_weight
        if not any(component_denominators.values()):
            raise ValueError(
                "At least one teacher/component weight must be positive"
            )

        totals = {
            name: (
                value / component_denominators[name]
                if component_denominators[name] > 0
                else value
            )
            for name, value in totals.items()
        }
        loss = sum(
            self.component_weights[name] * totals[name] for name in COMPONENTS
        )
        metrics = {
            f"distill_{name}": float(value.detach().item())
            for name, value in totals.items()
        }
        metrics["distill_loss"] = float(loss.detach().item())
        return loss, metrics
