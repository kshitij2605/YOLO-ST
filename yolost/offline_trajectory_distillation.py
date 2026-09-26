"""Training-only full-video trajectory supervision for tube queries."""

import torch
import torch.nn as nn

from yolost.tube_query import tube_query_loss


class OfflineTrajectoryDistillationLoss(nn.Module):
    """Apply cached teacher tracks without replacing supervised GT losses."""

    COMPONENTS = {
        "class": "query_cls_loss",
        "box": "query_box_loss",
        "giou": "query_giou_loss",
        "visibility": "query_visibility_loss",
        "boundary": "query_boundary_loss",
        "velocity": "query_velocity_loss",
        "acceleration": "query_acceleration_loss",
        "start": "query_start_loss",
        "end": "query_end_loss",
        "interval_iou": "query_interval_iou_loss",
        "coverage": "query_coverage_loss",
        "fragmentation": "query_fragmentation_loss",
    }

    def __init__(self, component_weights=None, cost_class=2.0, cost_box=5.0,
                 cost_giou=2.0, cost_visibility=1.0,
                 cost_interval=0.0, cost_coverage=0.0,
                 cost_fragmentation=0.0, confidence_weighted=False,
                 confidence_floor=0.05):
        super().__init__()
        component_weights = component_weights or {}
        self.component_weights = {
            name: float(component_weights.get(name, 0.0))
            for name in self.COMPONENTS
        }
        if min(self.component_weights.values()) < 0:
            raise ValueError("offline trajectory weights must be non-negative")
        if not any(self.component_weights.values()):
            raise ValueError("at least one offline trajectory weight is required")
        self.costs = {
            "cost_cls": float(cost_class),
            "cost_box": float(cost_box),
            "cost_giou": float(cost_giou),
            "cost_visibility": float(cost_visibility),
            "cost_interval": float(cost_interval),
            "cost_coverage": float(cost_coverage),
            "cost_fragmentation": float(cost_fragmentation),
        }
        self.confidence_weighted = bool(confidence_weighted)
        self.confidence_floor = float(confidence_floor)
        if not 0.0 < self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in (0, 1]")

    @staticmethod
    def _slice_outputs(query_outputs, indices):
        batch_size = query_outputs["boxes"].shape[0]
        return {
            name: (
                value.index_select(0, indices)
                if torch.is_tensor(value) and value.ndim
                and value.shape[0] == batch_size else value
            )
            for name, value in query_outputs.items()
        }

    def forward(self, model_outputs, targets, clip_length):
        query_outputs = (
            model_outputs.get("tube_queries")
            if isinstance(model_outputs, dict) else None
        )
        if query_outputs is None:
            raise ValueError("offline trajectory distillation requires tube queries")
        required = (
            "offline_boxes", "offline_labels", "offline_track_ids"
        )
        if any(name not in targets for name in required):
            raise ValueError("offline trajectory targets are missing from the batch")

        valid_observations = (
            (targets["offline_track_ids"] >= 0)
            & (targets["offline_boxes"][..., 1:5].sum(dim=-1) > 0)
        )
        valid_batch = valid_observations.any(dim=1)
        indices = valid_batch.nonzero(as_tuple=True)[0]
        if not indices.numel():
            zero = query_outputs["boxes"].sum() * 0.0
            metrics = {
                f"offline_trajectory_{name}": 0.0
                for name in self.COMPONENTS
            }
            metrics.update({
                "offline_trajectory_loss": 0.0,
                "offline_trajectory_samples": 0,
                "offline_trajectory_tracks": 0,
                "offline_trajectory_matches": 0,
                "offline_trajectory_target_weight": 0.0,
            })
            return zero, metrics

        offline_targets = {
            "boxes": targets["offline_boxes"].index_select(0, indices),
            "labels": targets["offline_labels"].index_select(0, indices),
            "track_ids": targets["offline_track_ids"].index_select(0, indices),
        }
        if self.confidence_weighted:
            required_confidence = ("offline_scores", "offline_quality")
            if any(name not in targets for name in required_confidence):
                raise ValueError(
                    "confidence-weighted offline targets require scores and quality"
                )
            offline_targets.update({
                "scores": targets["offline_scores"].index_select(0, indices),
                "quality": targets["offline_quality"].index_select(0, indices),
            })
        selected_outputs = self._slice_outputs(query_outputs, indices)
        losses = tube_query_loss(
            selected_outputs, offline_targets, clip_length,
            target_confidence_weighting=self.confidence_weighted,
            target_confidence_floor=self.confidence_floor,
            **self.costs,
        )
        total = selected_outputs["boxes"].new_zeros(())
        metrics = {}
        for component, loss_name in self.COMPONENTS.items():
            value = losses[loss_name]
            total = total + self.component_weights[component] * value
            metrics[f"offline_trajectory_{component}"] = float(
                value.detach().item()
            )
        track_count = 0
        for batch_index in range(offline_targets["track_ids"].shape[0]):
            ids = offline_targets["track_ids"][batch_index]
            track_count += int(ids[ids >= 0].unique().numel())
        metrics.update({
            "offline_trajectory_loss": float(total.detach().item()),
            "offline_trajectory_samples": int(indices.numel()),
            "offline_trajectory_tracks": track_count,
            "offline_trajectory_matches": int(losses["query_matches"]),
            "offline_trajectory_target_weight": float(
                losses["query_target_weight"]
            ),
        })
        return total, metrics
