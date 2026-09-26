"""Dense-frame to tube-query geometry contract.

The dense P3 detector remains the frame-localization owner. This module
matches its strongest per-frame proposals to each tube query and learns a
bounded temporal blend toward those boxes. The final gate is initialized at
zero, so enabling the contract does not perturb an existing checkpoint.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseTubeGeometryContract(nn.Module):
    """Correct tube boxes from detached dense detections without changing them."""

    context_channels = 14

    def __init__(
        self,
        num_classes,
        hidden=64,
        temporal_kernels=(3, 7, 15, 31),
        proposals=16,
        match_temperature=0.2,
        max_blend=0.5,
        smooth_corrections=False,
    ):
        super().__init__()
        kernels = tuple(int(kernel) for kernel in temporal_kernels)
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("temporal_kernels must contain positive odd values")
        if int(proposals) < 1:
            raise ValueError("proposals must be positive")
        if float(match_temperature) <= 0:
            raise ValueError("match_temperature must be positive")
        if not 0 < float(max_blend) <= 1:
            raise ValueError("max_blend must be in (0, 1]")

        self.num_classes = int(num_classes)
        self.proposals = int(proposals)
        self.match_temperature = float(match_temperature)
        self.max_blend = float(max_blend)
        self.temporal_kernels = kernels
        self.smooth_corrections = bool(smooth_corrections)

        self.input_projection = nn.Sequential(
            nn.Conv1d(self.context_channels, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=hidden,
                    bias=False,
                ),
                nn.Conv1d(hidden, hidden, kernel_size=1),
                nn.GELU(),
            )
            for kernel in kernels
        ])
        self.route_head = nn.Conv1d(hidden, len(kernels), kernel_size=1)
        self.gate_head = nn.Conv1d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    @staticmethod
    def _pairwise_iou(query_boxes, proposal_boxes):
        left_top = torch.maximum(
            query_boxes.unsqueeze(-2)[..., :2],
            proposal_boxes.unsqueeze(-3)[..., :2],
        )
        right_bottom = torch.minimum(
            query_boxes.unsqueeze(-2)[..., 2:],
            proposal_boxes.unsqueeze(-3)[..., 2:],
        )
        intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
        query_area = (
            query_boxes[..., 2:] - query_boxes[..., :2]
        ).clamp_min(0).prod(dim=-1).unsqueeze(-1)
        proposal_area = (
            proposal_boxes[..., 2:] - proposal_boxes[..., :2]
        ).clamp_min(0).prod(dim=-1).unsqueeze(-2)
        return intersection / (
            query_area + proposal_area - intersection
        ).clamp_min(1e-6)

    def _dense_proposals(self, dense_output, target_time):
        if dense_output is None or len(dense_output) < 3:
            raise ValueError("dense P3 class, box, and object logits are required")
        class_logits, box_logits, object_logits = (
            value.detach() for value in dense_output[:3]
        )
        if class_logits.shape[1] != self.num_classes:
            raise ValueError(
                "dense class count does not match the geometry contract"
            )
        batch, _, _, height, width = box_logits.shape
        target_size = (int(target_time), height, width)
        if box_logits.shape[2] != target_time:
            class_logits = F.interpolate(
                class_logits,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            box_logits = F.interpolate(
                box_logits,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            object_logits = F.interpolate(
                object_logits,
                size=target_size,
                mode="trilinear",
                align_corners=False,
            )

        class_probability = class_logits.sigmoid()
        object_probability = object_logits[:, 0].sigmoid()
        detection_score = (
            object_probability * class_probability.amax(dim=1)
        ).flatten(2)
        count = min(self.proposals, height * width)
        scores, indices = detection_score.topk(count, dim=-1)
        y_index = torch.div(indices, width, rounding_mode="floor")
        x_index = indices.remainder(width)

        regression = box_logits.permute(0, 2, 3, 4, 1).reshape(
            batch, target_time, height * width, 4
        )
        regression = regression.gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, 4)
        )
        center_x = (
            x_index.to(regression.dtype) + regression[..., 0].sigmoid()
        ) / width
        center_y = (
            y_index.to(regression.dtype) + regression[..., 1].sigmoid()
        ) / height
        box_width = regression[..., 2].clamp(max=5).exp() / width
        box_height = regression[..., 3].clamp(max=5).exp() / height
        boxes = torch.stack(
            [
                center_x - box_width * 0.5,
                center_y - box_height * 0.5,
                center_x + box_width * 0.5,
                center_y + box_height * 0.5,
            ],
            dim=-1,
        ).clamp(0, 1)

        dense_classes = class_probability.permute(0, 2, 3, 4, 1).reshape(
            batch, target_time, height * width, self.num_classes
        )
        dense_classes = dense_classes.gather(
            2,
            indices.unsqueeze(-1).expand(
                -1, -1, -1, self.num_classes
            ),
        )
        return boxes, scores, dense_classes

    @staticmethod
    def _centers_and_sizes(boxes):
        centers = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4)
        return centers, sizes

    @staticmethod
    def _velocity(values):
        return torch.cat(
            [torch.zeros_like(values[..., :1, :]), values[..., 1:, :] - values[..., :-1, :]],
            dim=-2,
        )

    def _route_temporal_values(self, values, route_weights):
        """Apply the pyramid kernels to values using the learned local route."""
        batch, queries, time, channels = values.shape
        flattened = values.reshape(
            batch * queries, time, channels
        ).transpose(1, 2)
        branches = []
        for kernel in self.temporal_kernels:
            if kernel == 1:
                branch = flattened
            else:
                branch = F.avg_pool1d(
                    flattened,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    count_include_pad=False,
                )
            branches.append(branch.transpose(1, 2))
        branches = torch.stack(branches, dim=2)
        routed = (
            branches * route_weights.reshape(
                batch * queries, time, len(self.temporal_kernels), 1
            )
        ).sum(dim=2)
        return routed.reshape(batch, queries, time, channels)

    def forward(self, query_output, dense_output):
        boxes = query_output.get("boxes")
        if boxes is None or boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError("tube-query boxes must have shape (B,Q,T,4)")
        batch, queries, time, _ = boxes.shape
        proposal_boxes, proposal_scores, proposal_classes = (
            self._dense_proposals(dense_output, time)
        )

        query_boxes = boxes.detach().permute(0, 2, 1, 3)
        overlap = self._pairwise_iou(query_boxes, proposal_boxes)
        query_centers, query_sizes = self._centers_and_sizes(query_boxes)
        proposal_centers, _ = self._centers_and_sizes(proposal_boxes)
        normalized_distance = (
            (
                query_centers.unsqueeze(-2)
                - proposal_centers.unsqueeze(-3)
            )
            / query_sizes.unsqueeze(-2)
        ).square().sum(dim=-1).sqrt()

        frame_class_logits = query_output.get("frame_class_logits")
        if frame_class_logits is None:
            class_logits = query_output["class_logits"]
            frame_class_logits = class_logits.unsqueeze(2).expand(
                -1, -1, time, -1
            )
        query_classes = F.normalize(
            frame_class_logits.detach().sigmoid().permute(0, 2, 1, 3),
            dim=-1,
        )
        dense_classes = F.normalize(proposal_classes, dim=-1)
        class_agreement = torch.einsum(
            "btqc,btkc->btqk", query_classes, dense_classes
        )

        match_logits = (
            4.0 * overlap
            - normalized_distance
            + proposal_scores.unsqueeze(-2).clamp_min(1e-6).log()
            + class_agreement
        )
        assignment = (
            match_logits / self.match_temperature
        ).softmax(dim=-1)
        candidate_boxes = torch.einsum(
            "btqk,btkd->btqd", assignment, proposal_boxes
        )
        candidate_score = torch.einsum(
            "btqk,btk->btq", assignment, proposal_scores
        )
        candidate_overlap = (assignment * overlap).sum(dim=-1)
        candidate_agreement = (assignment * class_agreement).sum(dim=-1)
        assignment_entropy = -(
            assignment * assignment.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(max(2, assignment.shape[-1]))

        candidate_centers, candidate_sizes = self._centers_and_sizes(
            candidate_boxes
        )
        center_delta = (
            (candidate_centers - query_centers) / query_sizes
        ).clamp(-4, 4)
        log_size_delta = (
            candidate_sizes.log() - query_sizes.log()
        ).clamp(-4, 4)
        center_distance = center_delta.square().sum(dim=-1).sqrt()
        query_velocity = self._velocity(query_centers.permute(0, 2, 1, 3))
        candidate_velocity = self._velocity(
            candidate_centers.permute(0, 2, 1, 3)
        )
        visibility = query_output["visibility_logits"].detach().sigmoid()

        context = torch.cat(
            [
                center_delta.permute(0, 2, 1, 3),
                log_size_delta.permute(0, 2, 1, 3),
                candidate_overlap.permute(0, 2, 1).unsqueeze(-1),
                center_distance.permute(0, 2, 1).unsqueeze(-1),
                candidate_score.permute(0, 2, 1).unsqueeze(-1),
                candidate_agreement.permute(0, 2, 1).unsqueeze(-1),
                assignment_entropy.permute(0, 2, 1).unsqueeze(-1),
                visibility.unsqueeze(-1),
                query_velocity,
                candidate_velocity,
            ],
            dim=-1,
        )
        context = context.reshape(
            batch * queries, time, self.context_channels
        ).transpose(1, 2)
        base = self.input_projection(context)
        branches = torch.stack(
            [branch(base) for branch in self.temporal_branches],
            dim=2,
        )
        route_weights = self.route_head(base).softmax(dim=1).unsqueeze(1)
        temporal_context = (branches * route_weights).sum(dim=2)
        raw_gate = self.gate_head(temporal_context)
        # This positive-part form is exactly zero at initialization while
        # retaining a nonzero derivative for the zero-initialized gate.
        signed_gate = raw_gate.tanh()
        gate = 0.5 * (signed_gate + signed_gate.abs())
        gate = self.max_blend * gate.transpose(1, 2).reshape(
            batch, queries, time, 1
        )

        query_centers = query_centers.permute(0, 2, 1, 3)
        query_sizes = query_sizes.permute(0, 2, 1, 3)
        candidate_centers = candidate_centers.permute(0, 2, 1, 3)
        candidate_sizes = candidate_sizes.permute(0, 2, 1, 3)
        exported_route_weights = (
            route_weights.squeeze(1)
            .reshape(batch, queries, len(self.temporal_kernels), time)
            .permute(0, 1, 3, 2)
        )
        center_correction = candidate_centers - query_centers
        log_size_correction = candidate_sizes.log() - query_sizes.log()
        if self.smooth_corrections:
            center_correction = self._route_temporal_values(
                center_correction, exported_route_weights
            )
            log_size_correction = self._route_temporal_values(
                log_size_correction, exported_route_weights
            )
        corrected_centers = query_centers + gate * center_correction
        corrected_sizes = (
            query_sizes.log() + gate * log_size_correction
        ).exp()
        corrected_boxes = torch.cat(
            [
                corrected_centers - corrected_sizes * 0.5,
                corrected_centers + corrected_sizes * 0.5,
            ],
            dim=-1,
        ).clamp(0, 1)

        adapted = dict(query_output)
        adapted["boxes"] = corrected_boxes
        adapted["geometry_contract_parent_boxes"] = boxes.detach()
        adapted["geometry_contract_gate"] = gate.squeeze(-1)
        adapted["geometry_contract_candidate_boxes"] = torch.cat(
            [
                candidate_centers - candidate_sizes * 0.5,
                candidate_centers + candidate_sizes * 0.5,
            ],
            dim=-1,
        ).clamp(0, 1)
        adapted["geometry_contract_route_weights"] = exported_route_weights
        return adapted
