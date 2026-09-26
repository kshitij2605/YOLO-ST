"""Parent-preserving residual corrections for factorized tube queries."""

import torch
import torch.nn as nn


class TubeQueryTrajectoryResidualAdapter(nn.Module):
    """Refine a frozen tube-query trajectory with routed temporal filters.

    All terminal projections are zero initialized, so enabling the adapter is
    an exact identity before optimization. Inputs are detached to keep the
    parent detector fixed while the residual learns from supervised and
    trajectory-teacher targets.
    """

    def __init__(self, num_classes, hidden=64, temporal_kernels=(3, 7, 15, 31),
                 max_box_delta=0.05, class_residual=True, box_residual=True,
                 visibility_residual=True, boundary_residual=True,
                 endpoint_residual=True):
        super().__init__()
        hidden = int(hidden)
        kernels = tuple(int(kernel) for kernel in temporal_kernels)
        if hidden < 8:
            raise ValueError("hidden must be at least 8")
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("temporal kernels must be positive odd integers")
        if max_box_delta <= 0:
            raise ValueError("max_box_delta must be positive")
        if not any((
            class_residual, box_residual, visibility_residual,
            boundary_residual, endpoint_residual,
        )):
            raise ValueError("at least one trajectory residual must be enabled")

        self.num_classes = int(num_classes)
        self.max_box_delta = float(max_box_delta)
        self.input_projection = nn.Linear(self.num_classes + 6, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.temporal_filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden, hidden, kernel_size=kernel,
                    padding=kernel // 2, groups=hidden, bias=False,
                ),
                nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
            )
            for kernel in kernels
        ])
        self.route = nn.Linear(hidden + 1, len(kernels))
        self.output_norm = nn.LayerNorm(hidden)

        self.class_projection = (
            nn.Linear(hidden, self.num_classes) if class_residual else None
        )
        self.box_projection = nn.Linear(hidden, 4) if box_residual else None
        self.visibility_projection = (
            nn.Linear(hidden, 1) if visibility_residual else None
        )
        self.boundary_projection = (
            nn.Linear(hidden, 1) if boundary_residual else None
        )
        self.start_projection = (
            nn.Linear(hidden, 1) if endpoint_residual else None
        )
        self.end_projection = (
            nn.Linear(hidden, 1) if endpoint_residual else None
        )
        for projection in (
            self.class_projection, self.box_projection,
            self.visibility_projection, self.boundary_projection,
            self.start_projection, self.end_projection,
        ):
            if projection is not None:
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    @staticmethod
    def _required(query_output, name):
        if name not in query_output:
            raise ValueError(f"tube-query output is missing {name}")
        return query_output[name]

    def forward(self, query_output):
        class_logits = self._required(query_output, "class_logits")
        boxes = self._required(query_output, "boxes")
        visibility_logits = self._required(query_output, "visibility_logits")
        boundary_logits = self._required(query_output, "boundary_logits")
        if boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError("tube-query boxes must have shape (B,Q,T,4)")
        if visibility_logits.shape != boxes.shape[:-1]:
            raise ValueError("visibility and box trajectories are incompatible")
        if boundary_logits.shape != visibility_logits.shape:
            raise ValueError("boundary and visibility trajectories are incompatible")

        detached_class = class_logits.detach().sigmoid()
        detached_boxes = boxes.detach()
        detached_visibility = visibility_logits.detach().sigmoid()
        detached_boundary = boundary_logits.detach().sigmoid()
        time_count = boxes.shape[2]
        class_features = detached_class.unsqueeze(2).expand(-1, -1, time_count, -1)
        features = torch.cat((
            class_features,
            detached_boxes,
            detached_visibility.unsqueeze(-1),
            detached_boundary.unsqueeze(-1),
        ), dim=-1)
        features = self.input_norm(self.input_projection(features))

        batch, queries, _, hidden = features.shape
        sequence = features.reshape(batch * queries, time_count, hidden).transpose(1, 2)
        filtered = torch.stack([
            temporal_filter(sequence).transpose(1, 2).reshape(
                batch, queries, time_count, hidden
            )
            for temporal_filter in self.temporal_filters
        ], dim=2)
        duration = detached_visibility.mean(dim=-1, keepdim=True)
        route_input = torch.cat((features.mean(dim=2), duration), dim=-1)
        route_weights = self.route(route_input).softmax(dim=-1)
        routed = (
            filtered * route_weights[:, :, :, None, None]
        ).sum(dim=2)
        residual_features = self.output_norm(features + routed)

        adapted = dict(query_output)
        if self.class_projection is not None:
            pooled_weight = detached_visibility.unsqueeze(-1)
            pooled = (residual_features * pooled_weight).sum(dim=2)
            pooled = pooled / pooled_weight.sum(dim=2).clamp_min(1e-4)
            adapted["class_logits"] = (
                class_logits.detach() + self.class_projection(pooled)
            )
        if self.box_projection is not None:
            delta = self.max_box_delta * self.box_projection(
                residual_features
            ).tanh()
            candidate = boxes.detach() + delta
            lower = torch.minimum(candidate[..., :2], candidate[..., 2:])
            upper = torch.maximum(candidate[..., :2], candidate[..., 2:])
            adapted["boxes"] = torch.cat((lower, upper), dim=-1).clamp(0.0, 1.0)
        if self.visibility_projection is not None:
            adapted["visibility_logits"] = (
                visibility_logits.detach()
                + self.visibility_projection(residual_features).squeeze(-1)
            )
        if self.boundary_projection is not None:
            adapted["boundary_logits"] = (
                boundary_logits.detach()
                + self.boundary_projection(residual_features).squeeze(-1)
            )
        if self.start_projection is not None and "start_logits" in query_output:
            adapted["start_logits"] = (
                query_output["start_logits"].detach()
                + self.start_projection(residual_features).squeeze(-1)
            )
        if self.end_projection is not None and "end_logits" in query_output:
            adapted["end_logits"] = (
                query_output["end_logits"].detach()
                + self.end_projection(residual_features).squeeze(-1)
            )
        adapted["trajectory_residual_route_weights"] = route_weights
        return adapted
