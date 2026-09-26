"""Learned tube queries and sequence-level Hungarian supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is in requirements.txt.
    linear_sum_assignment = None


class TubeQueryHead(nn.Module):
    """Decode class, visibility, boundaries, and a box sequence per query."""

    def __init__(self, channels, num_classes, hidden=256, num_queries=8,
                 num_heads=8, depth=2, dropout=0.1, frames=None,
                 max_frames=128):
        super().__init__()
        self.num_queries = int(num_queries)
        self.max_frames = int(max_frames)
        self.frames = int(frames) if frames is not None else None
        self.spatial_score = nn.Conv3d(channels, 1, kernel_size=1)
        self.feature_proj = nn.Linear(channels, hidden)
        self.time_embed = nn.Parameter(torch.zeros(1, max_frames, hidden))
        self.query_embed = nn.Parameter(torch.empty(num_queries, hidden))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=depth)
        self.query_norm = nn.LayerNorm(hidden)
        self.frame_norm = nn.LayerNorm(hidden)
        self.class_head = nn.Linear(hidden, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )
        self.visibility_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    def forward(self, feature):
        if self.frames is not None and feature.shape[2] != self.frames:
            feature = F.adaptive_avg_pool3d(
                feature, (self.frames, feature.shape[3], feature.shape[4])
            )
        batch_size, _, time, _, _ = feature.shape
        if time > self.max_frames:
            raise ValueError(f"tube query time {time} exceeds max_frames={self.max_frames}")
        weights = self.spatial_score(feature).flatten(3).softmax(dim=-1)
        dense = feature.flatten(3).permute(0, 2, 3, 1)
        temporal = torch.einsum("btn,btnc->btc", weights[:, 0], dense)
        temporal = self.feature_proj(temporal) + self.time_embed[:, :time]
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.query_norm(self.decoder(queries, temporal))
        frame_tokens = self.frame_norm(decoded.unsqueeze(2) + temporal.unsqueeze(1))
        return {
            "class_logits": self.class_head(decoded),
            "boxes": self.box_head(frame_tokens).sigmoid(),
            "visibility_logits": self.visibility_head(frame_tokens).squeeze(-1),
            "boundary_logits": self.boundary_head(frame_tokens).squeeze(-1),
        }


class FactorizedPersonTubeletLayer(nn.Module):
    """Alternate person, spatial, and temporal attention for full tubelets."""

    def __init__(self, hidden, num_heads, dropout=0.1, boundary_gated=False):
        super().__init__()
        self.boundary_gated = bool(boundary_gated)
        self.person_attention = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_norm = nn.LayerNorm(hidden)
        self.spatial_attention = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_dropout = nn.Dropout(dropout)
        self.spatial_ffn_norm = nn.LayerNorm(hidden)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.temporal_attention = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if self.boundary_gated:
            self.transition_gate = nn.Linear(hidden * 2, 1)
            nn.init.zeros_(self.transition_gate.weight)
            nn.init.constant_(self.transition_gate.bias, 4.0)
        else:
            self.transition_gate = None

    def forward(self, tokens, memory):
        batch, queries, time, hidden = tokens.shape

        # Person attention separates simultaneous actors independently per frame.
        person_tokens = tokens.permute(0, 2, 1, 3).reshape(
            batch * time, queries, hidden
        )
        person_tokens = self.person_attention(person_tokens)

        # Each person/time query localizes directly against that frame's pixels.
        spatial_query = self.spatial_norm(person_tokens)
        spatial_memory = memory.reshape(batch * time, memory.shape[2], hidden)
        spatial_update = self.spatial_attention(
            spatial_query, spatial_memory, spatial_memory, need_weights=False
        )[0]
        person_tokens = person_tokens + self.spatial_dropout(spatial_update)
        person_tokens = person_tokens + self.spatial_ffn(
            self.spatial_ffn_norm(person_tokens)
        )

        # Temporal attention binds each person query into one complete tubelet.
        tokens = person_tokens.reshape(batch, time, queries, hidden).permute(0, 2, 1, 3)
        temporal_tokens = tokens.reshape(batch * queries, time, hidden)
        temporal_update = self.temporal_attention(temporal_tokens)
        if self.transition_gate is not None:
            gate = self.transition_gate(
                torch.cat([temporal_tokens, temporal_update], dim=-1)
            ).sigmoid()
            temporal_update = temporal_tokens + gate * (temporal_update - temporal_tokens)
        return temporal_update.reshape(batch, queries, time, hidden)


class ActorInstanceTubeletLayer(nn.Module):
    """Factor attention across actors, action instances, space, and time."""

    def __init__(self, hidden, num_heads, dropout=0.1, boundary_gated=False):
        super().__init__()
        encoder_args = dict(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.actor_attention = nn.TransformerEncoderLayer(**encoder_args)
        self.instance_attention = nn.TransformerEncoderLayer(**encoder_args)
        self.spatial_norm = nn.LayerNorm(hidden)
        self.spatial_attention = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_dropout = nn.Dropout(dropout)
        self.spatial_ffn_norm = nn.LayerNorm(hidden)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.temporal_attention = nn.TransformerEncoderLayer(**encoder_args)
        if boundary_gated:
            self.transition_gate = nn.Linear(hidden * 2, 1)
            nn.init.zeros_(self.transition_gate.weight)
            nn.init.constant_(self.transition_gate.bias, 4.0)
        else:
            self.transition_gate = None

    def forward(self, tokens, memory):
        batch, actors, instances, time, hidden = tokens.shape

        # Each actor first aggregates its candidate action instances. Actor
        # attention then separates simultaneous people without erasing the
        # instance-specific residuals.
        actor_summary = tokens.mean(dim=2)
        actor_tokens = actor_summary.permute(0, 2, 1, 3).reshape(
            batch * time, actors, hidden
        )
        actor_tokens = self.actor_attention(actor_tokens)
        actor_tokens = actor_tokens.reshape(batch, time, actors, hidden).permute(
            0, 2, 1, 3
        )
        tokens = tokens + (actor_tokens - actor_summary).unsqueeze(2)

        # Multiple intervals belonging to one actor exchange context before
        # localizing independently against the spatial pyramid.
        instance_tokens = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch * actors * time, instances, hidden
        )
        instance_tokens = self.instance_attention(instance_tokens)
        tokens = instance_tokens.reshape(
            batch, actors, time, instances, hidden
        ).permute(0, 1, 3, 2, 4)

        queries = actors * instances
        spatial_tokens = tokens.permute(0, 3, 1, 2, 4).reshape(
            batch * time, queries, hidden
        )
        spatial_query = self.spatial_norm(spatial_tokens)
        spatial_memory = memory.reshape(batch * time, memory.shape[2], hidden)
        spatial_update = self.spatial_attention(
            spatial_query, spatial_memory, spatial_memory, need_weights=False
        )[0]
        spatial_tokens = spatial_tokens + self.spatial_dropout(spatial_update)
        spatial_tokens = spatial_tokens + self.spatial_ffn(
            self.spatial_ffn_norm(spatial_tokens)
        )

        tokens = spatial_tokens.reshape(
            batch, time, actors, instances, hidden
        ).permute(0, 2, 3, 1, 4)
        temporal_tokens = tokens.reshape(batch * queries, time, hidden)
        temporal_update = self.temporal_attention(temporal_tokens)
        if self.transition_gate is not None:
            gate = self.transition_gate(
                torch.cat([temporal_tokens, temporal_update], dim=-1)
            ).sigmoid()
            temporal_update = temporal_tokens + gate * (
                temporal_update - temporal_tokens
            )
        return temporal_update.reshape(batch, actors, instances, time, hidden)


class FactorizedPersonTubeletHead(nn.Module):
    """Person-bound factorized tubelet decoder over a multi-scale video pyramid.

    Queries are learned person slots rather than action slots or dense-detector
    proposals. Every slot predicts one action distribution and a complete box,
    visibility, and boundary sequence.
    """

    def __init__(self, channels, num_classes, hidden=256, num_queries=16,
                 num_heads=8, depth=6, dropout=0.1, frames=32,
                 max_frames=128, memory_grid=14, boundary_gated=False,
                 iterative_refinement=False, trajectory_sampling=False,
                 trajectory_points=5, interval_queries=False,
                 instances_per_actor=1, interval_pyramid=False,
                 drop_path_rate=0.0, class_prior_probability=None,
                 predict_quality=False, predict_boundary_distance=False,
                 boundary_distance_temperature=0.08, duration_router=False,
                 duration_kernels=(3, 7, 15),
                 change_point_pyramid=False,
                 change_point_dilations=(1, 2, 4, 8),
                 change_point_router_mode="actor_duration",
                 change_point_shared_projection=False,
                 identity_transport=False,
                 transport_proposals=16,
                 transport_sinkhorn_iterations=4,
                 transport_temperature=0.2,
                 action_reset_state=False,
                 action_fork_state=False,
                 action_fork_mode="state_boundary",
                 action_fork_temperature=0.5,
                 decision_memory_dim=0,
                 decision_memory_target="class_boundary"):
        super().__init__()
        if frames > max_frames:
            raise ValueError(f"tubelet frames={frames} exceeds max_frames={max_frames}")
        self.interval_queries = bool(interval_queries)
        self.num_actors = int(num_queries)
        self.instances_per_actor = (
            max(1, int(instances_per_actor)) if self.interval_queries else 1
        )
        self.num_queries = self.num_actors * self.instances_per_actor
        self.frames = int(frames)
        self.memory_grid = int(memory_grid)
        self.iterative_refinement = bool(iterative_refinement)
        self.trajectory_sampling = bool(trajectory_sampling)
        self.interval_pyramid = bool(interval_pyramid)
        self.predict_quality = bool(predict_quality)
        self.predict_boundary_distance = bool(predict_boundary_distance)
        self.boundary_distance_temperature = float(
            boundary_distance_temperature
        )
        if self.predict_boundary_distance and not self.interval_queries:
            raise ValueError("boundary-distance prediction requires interval queries")
        if self.boundary_distance_temperature <= 0:
            raise ValueError("boundary_distance_temperature must be positive")
        self.duration_router_enabled = bool(duration_router)
        self.change_point_pyramid_enabled = bool(change_point_pyramid)
        self.change_point_shared_projection = bool(
            change_point_shared_projection
        )
        self.identity_transport_enabled = bool(identity_transport)
        self.action_reset_state_enabled = bool(action_reset_state)
        self.action_fork_state_enabled = bool(action_fork_state)
        if self.identity_transport_enabled and not self.interval_queries:
            raise ValueError("identity transport requires interval queries")
        if self.action_reset_state_enabled and not self.interval_queries:
            raise ValueError("action-reset state requires interval queries")
        if self.action_fork_state_enabled and not self.interval_queries:
            raise ValueError("action-fork state requires interval queries")
        if self.action_fork_state_enabled and self.instances_per_actor < 2:
            raise ValueError("action-fork state requires at least two instances per actor")
        if self.action_fork_state_enabled and self.action_reset_state_enabled:
            raise ValueError("action-fork and action-reset state are mutually exclusive")
        self.action_fork_mode = str(action_fork_mode).lower()
        valid_action_fork_modes = {"visibility", "state", "state_boundary"}
        if self.action_fork_mode not in valid_action_fork_modes:
            raise ValueError(
                "action_fork_mode must be one of "
                f"{sorted(valid_action_fork_modes)}"
            )
        self.action_fork_temperature = float(action_fork_temperature)
        if self.action_fork_temperature <= 0:
            raise ValueError("action_fork_temperature must be positive")
        self.decision_memory_target = str(decision_memory_target).lower()
        valid_decision_targets = {
            "class", "boundary", "class_boundary", "quality",
            "class_quality", "quality_residual", "class_quality_residual",
        }
        if self.decision_memory_target not in valid_decision_targets:
            raise ValueError(
                "decision_memory_target must be one of "
                f"{sorted(valid_decision_targets)}"
            )
        if int(decision_memory_dim) > 0:
            self.decision_memory_projection = nn.Sequential(
                nn.Linear(int(decision_memory_dim), hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            decision_scale = (
                1.0 if self.decision_memory_target in {
                    "quality_residual", "class_quality_residual",
                } else 0.0
            )
            self.decision_memory_scale = nn.Parameter(
                torch.tensor(decision_scale)
            )
        else:
            self.decision_memory_projection = None
            self.register_parameter("decision_memory_scale", None)
        self.transport_proposals = max(self.num_actors, int(transport_proposals))
        self.transport_sinkhorn_iterations = max(
            1, int(transport_sinkhorn_iterations)
        )
        self.transport_temperature = float(transport_temperature)
        if self.transport_temperature <= 0:
            raise ValueError("transport_temperature must be positive")
        self.change_point_router_mode = str(change_point_router_mode).lower()
        valid_change_point_modes = {"actor_duration", "actor", "equal"}
        if self.change_point_router_mode not in valid_change_point_modes:
            raise ValueError(
                "change_point_router_mode must be one of "
                f"{sorted(valid_change_point_modes)}"
            )
        if self.change_point_pyramid_enabled and not self.interval_queries:
            raise ValueError("change-point pyramid requires interval queries")
        self.drop_path_rate = float(drop_path_rate)
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.feature_proj = nn.ModuleList([
            nn.Conv3d(value, hidden, kernel_size=1) for value in channels
        ])
        self.scale_embed = nn.Parameter(torch.empty(len(channels), hidden))
        self.position_embed = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        if self.identity_transport_enabled:
            self.transport_feature_norm = nn.LayerNorm(hidden)
            self.transport_actor_seed = nn.Parameter(
                torch.empty(1, self.num_actors, hidden)
            )
            self.transport_box_embed = nn.Sequential(
                nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            self.transport_context_norm = nn.LayerNorm(hidden)
            self.transport_cost_weights = nn.Parameter(torch.tensor([
                2.0, 4.0, 2.0, 1.0,
            ]))
            self.transport_dustbin_logit = nn.Parameter(torch.tensor(-1.0))
            self.transport_update_logit = nn.Parameter(torch.tensor(0.0))
            self.transport_scale = nn.Parameter(torch.tensor(-2.944439))
        else:
            self.transport_feature_norm = None
            self.register_parameter("transport_actor_seed", None)
            self.transport_box_embed = None
            self.transport_context_norm = None
            self.register_parameter("transport_cost_weights", None)
            self.register_parameter("transport_dustbin_logit", None)
            self.register_parameter("transport_update_logit", None)
            self.register_parameter("transport_scale", None)
        if self.interval_queries:
            self.person_embed = nn.Parameter(
                torch.empty(1, self.num_actors, 1, 1, hidden)
            )
            self.instance_embed = nn.Parameter(
                torch.empty(1, 1, self.instances_per_actor, 1, hidden)
            )
            self.time_embed = nn.Parameter(torch.empty(1, 1, 1, max_frames, hidden))
            self.layers = nn.ModuleList([
                ActorInstanceTubeletLayer(
                    hidden, num_heads, dropout, boundary_gated=boundary_gated
                )
                for _ in range(depth)
            ])
        else:
            self.person_embed = nn.Parameter(
                torch.empty(1, self.num_actors, 1, hidden)
            )
            self.register_parameter("instance_embed", None)
            self.time_embed = nn.Parameter(torch.empty(1, 1, max_frames, hidden))
            self.layers = nn.ModuleList([
                FactorizedPersonTubeletLayer(
                    hidden, num_heads, dropout, boundary_gated=boundary_gated
                )
                for _ in range(depth)
            ])
        self.output_norm = nn.LayerNorm(hidden)
        self.person_norm = nn.LayerNorm(hidden)
        self.class_head = nn.Linear(hidden, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )
        if self.iterative_refinement:
            self.refinement_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, 4),
                ) for _ in range(depth)
            ])
        else:
            self.refinement_heads = None
        if self.iterative_refinement or self.trajectory_sampling:
            self.box_condition = nn.Sequential(
                nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden)
            )
            self.box_condition_scale = nn.Parameter(torch.zeros(()))
        else:
            self.box_condition = None
            self.register_parameter("box_condition_scale", None)
        if self.trajectory_sampling:
            patterns = torch.tensor([
                [0.0, 0.0], [-1.0, -1.0], [1.0, -1.0],
                [-1.0, 1.0], [1.0, 1.0],
            ])
            count = max(1, min(int(trajectory_points), patterns.shape[0]))
            self.register_buffer("trajectory_offsets", patterns[:count])
            self.trajectory_norm = nn.LayerNorm(hidden)
            self.trajectory_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("trajectory_offsets", None)
            self.trajectory_norm = None
            self.register_parameter("trajectory_scale", None)
        self.visibility_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)
        self.quality_head = nn.Linear(hidden, 1) if self.predict_quality else None
        self.quality_residual_enabled = self.decision_memory_target in {
            "quality_residual", "class_quality_residual",
        }
        self.quality_residual_base_logit = 4.59512
        if self.predict_quality and self.quality_residual_enabled:
            self.quality_residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("quality_residual_scale", None)
        if self.predict_boundary_distance:
            self.boundary_distance_head = nn.Linear(hidden, 2)
            self.boundary_distance_scale = nn.Parameter(torch.tensor(-4.59512))
        else:
            self.boundary_distance_head = None
            self.register_parameter("boundary_distance_scale", None)
        if self.action_reset_state_enabled:
            self.action_reset_norm = nn.LayerNorm(hidden)
            self.action_reset_head = nn.Linear(hidden, 1)
            self.action_state_cell = nn.GRUCell(hidden, hidden)
            self.action_state_proj = nn.Linear(hidden, hidden)
            self.action_state_scale = nn.Parameter(torch.tensor(-2.944439))
            self.action_reset_scale = nn.Parameter(torch.tensor(-2.944439))
        else:
            self.action_reset_norm = None
            self.action_reset_head = None
            self.action_state_cell = None
            self.action_state_proj = None
            self.register_parameter("action_state_scale", None)
            self.register_parameter("action_reset_scale", None)
        if self.action_fork_state_enabled:
            self.action_fork_norm = nn.LayerNorm(hidden)
            self.action_fork_head = nn.Linear(hidden, 1)
            self.action_fork_state_proj = nn.Linear(hidden, hidden)
            self.action_fork_state_scale = nn.Parameter(torch.tensor(-2.944439))
            self.action_fork_visibility_scale = nn.Parameter(
                torch.tensor(-2.944439)
            )
            self.action_fork_boundary_scale = nn.Parameter(
                torch.tensor(-2.944439)
            )
        else:
            self.action_fork_norm = None
            self.action_fork_head = None
            self.action_fork_state_proj = None
            self.register_parameter("action_fork_state_scale", None)
            self.register_parameter("action_fork_visibility_scale", None)
            self.register_parameter("action_fork_boundary_scale", None)
        if self.duration_router_enabled:
            kernels = tuple(int(kernel) for kernel in duration_kernels)
            if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
                raise ValueError("duration_kernels must contain positive odd values")
            self.duration_filters = nn.ModuleList([
                nn.Conv1d(
                    hidden, hidden, kernel_size=kernel,
                    padding=kernel // 2, groups=hidden, bias=False,
                )
                for kernel in kernels
            ])
            self.duration_router = nn.Sequential(
                nn.Linear(hidden + 1, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, len(kernels)),
            )
            self.duration_mix = nn.Linear(hidden, hidden)
            self.duration_router_scale = nn.Parameter(torch.zeros(()))
        else:
            self.duration_filters = nn.ModuleList()
            self.duration_router = None
            self.duration_mix = None
            self.register_parameter("duration_router_scale", None)
        if self.interval_queries:
            self.start_head = nn.Linear(hidden, 1)
            self.end_head = nn.Linear(hidden, 1)
            self.start_prior = nn.Parameter(torch.zeros(max_frames))
            self.end_prior = nn.Parameter(torch.zeros(max_frames))
            self.interval_visibility_gate = nn.Parameter(torch.tensor(-2.1972246))
            if self.change_point_pyramid_enabled:
                dilations = tuple(dict.fromkeys(
                    int(value) for value in change_point_dilations
                ))
                if not dilations or any(value < 1 for value in dilations):
                    raise ValueError(
                        "change_point_dilations must contain positive integers"
                    )
                self.change_point_dilations = dilations
                self.change_point_norm = nn.LayerNorm(hidden)
                if self.change_point_shared_projection:
                    self.change_point_shared = nn.ModuleList([
                        nn.Linear(hidden, 1, bias=False) for _ in dilations
                    ])
                    self.change_point_start = nn.ModuleList()
                    self.change_point_end = nn.ModuleList()
                else:
                    self.change_point_shared = nn.ModuleList()
                    self.change_point_start = nn.ModuleList([
                        nn.Linear(hidden, 1, bias=False) for _ in dilations
                    ])
                    self.change_point_end = nn.ModuleList([
                        nn.Linear(hidden, 1, bias=False) for _ in dilations
                    ])
                if self.change_point_router_mode == "equal":
                    self.change_point_router = None
                else:
                    router_hidden = max(16, hidden // 4)
                    router_input = hidden + int(
                        self.change_point_router_mode == "actor_duration"
                    )
                    self.change_point_router = nn.Sequential(
                        nn.Linear(router_input, router_hidden),
                        nn.GELU(),
                        nn.Linear(router_hidden, len(dilations)),
                    )
                # Start near the inherited interval head while keeping a
                # gradient path into every change projection from step one.
                self.change_point_scale = nn.Parameter(torch.tensor(-2.944439))
            else:
                self.change_point_dilations = ()
                self.change_point_norm = None
                self.change_point_shared = nn.ModuleList()
                self.change_point_start = nn.ModuleList()
                self.change_point_end = nn.ModuleList()
                self.change_point_router = None
                self.register_parameter("change_point_scale", None)
            if self.interval_pyramid:
                self.motion_proj = nn.Sequential(
                    nn.LayerNorm(hidden * 2),
                    nn.Linear(hidden * 2, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                self.motion_scale_logits = nn.Parameter(torch.zeros(len(channels)))
                self.motion_condition_scale = nn.Parameter(torch.tensor(0.1))
            else:
                self.motion_proj = None
                self.register_parameter("motion_scale_logits", None)
                self.register_parameter("motion_condition_scale", None)
        else:
            self.start_head = None
            self.end_head = None
            self.register_parameter("start_prior", None)
            self.register_parameter("end_prior", None)
            self.register_parameter("interval_visibility_gate", None)
            self.change_point_dilations = ()
            self.change_point_norm = None
            self.change_point_shared = nn.ModuleList()
            self.change_point_start = nn.ModuleList()
            self.change_point_end = nn.ModuleList()
            self.change_point_router = None
            self.register_parameter("change_point_scale", None)
            self.motion_proj = None
            self.register_parameter("motion_scale_logits", None)
            self.register_parameter("motion_condition_scale", None)

        nn.init.normal_(self.scale_embed, std=0.02)
        nn.init.normal_(self.person_embed, std=0.02)
        if self.transport_actor_seed is not None:
            nn.init.normal_(self.transport_actor_seed, std=0.02)
        if self.instance_embed is not None:
            nn.init.normal_(self.instance_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)
        if self.refinement_heads is not None:
            for head in self.refinement_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.box_head[-1].bias[2:].fill_(-1.3862944)
            self.visibility_head.bias.fill_(-1.0)
            self.boundary_head.bias.fill_(-2.0)
            if self.quality_head is not None:
                self.quality_head.bias.fill_(-1.0986123)
            if self.boundary_distance_head is not None:
                nn.init.zeros_(self.boundary_distance_head.weight)
                nn.init.zeros_(self.boundary_distance_head.bias)
            if class_prior_probability is not None:
                probability = float(class_prior_probability)
                if not 0.0 < probability < 1.0:
                    raise ValueError("class_prior_probability must be in (0, 1)")
                self.class_head.bias.fill_(
                    torch.logit(self.class_head.bias.new_tensor(probability))
                )
            if self.interval_queries:
                self.start_head.bias.zero_()
                self.end_head.bias.zero_()
            if self.action_reset_head is not None:
                nn.init.normal_(self.action_reset_head.weight, std=0.02)
                self.action_reset_head.bias.fill_(-3.0)
            if self.action_fork_head is not None:
                nn.init.normal_(self.action_fork_head.weight, std=0.02)
                self.action_fork_head.bias.fill_(-3.0)
                nn.init.zeros_(self.action_fork_state_proj.weight)
                nn.init.zeros_(self.action_fork_state_proj.bias)
            if self.change_point_pyramid_enabled:
                for projection in self.change_point_shared:
                    nn.init.normal_(projection.weight, std=0.02)
                for projection in self.change_point_start:
                    nn.init.normal_(projection.weight, std=0.02)
                for projection in self.change_point_end:
                    nn.init.normal_(projection.weight, std=0.02)

    def _memory(self, features):
        memories = []
        projected_features = []
        for scale, (feature, projection) in enumerate(zip(features, self.feature_proj)):
            projected = projection(feature)
            if projected.shape[2] != self.frames:
                projected = F.interpolate(
                    projected,
                    size=(self.frames, projected.shape[3], projected.shape[4]),
                    mode="trilinear",
                    align_corners=False,
                )
            projected_features.append(projected)
            height = min(projected.shape[-2], self.memory_grid)
            width = min(projected.shape[-1], self.memory_grid)
            projected = F.adaptive_avg_pool3d(
                projected, output_size=(self.frames, height, width)
            )
            batch, hidden, time, _, _ = projected.shape
            tokens = projected.permute(0, 2, 3, 4, 1).reshape(
                batch, time, height * width, hidden
            )
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, height, device=feature.device,
                               dtype=feature.dtype),
                torch.linspace(-1.0, 1.0, width, device=feature.device,
                               dtype=feature.dtype),
                indexing="ij",
            )
            scale_coordinate = torch.full_like(xx, float(scale) / max(len(features) - 1, 1))
            coordinates = torch.stack([xx, yy, scale_coordinate], dim=-1).reshape(-1, 3)
            position = self.position_embed(coordinates).view(1, 1, height * width, hidden)
            memories.append(tokens + position + self.scale_embed[scale].view(1, 1, 1, hidden))
        return torch.cat(memories, dim=2), projected_features

    @staticmethod
    def _pairwise_box_iou(actor_boxes, proposal_boxes):
        left_top = torch.maximum(
            actor_boxes[:, :, None, :2], proposal_boxes[:, None, :, :2]
        )
        right_bottom = torch.minimum(
            actor_boxes[:, :, None, 2:], proposal_boxes[:, None, :, 2:]
        )
        intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
        actor_area = (
            actor_boxes[..., 2:] - actor_boxes[..., :2]
        ).clamp_min(0).prod(dim=-1)[:, :, None]
        proposal_area = (
            proposal_boxes[..., 2:] - proposal_boxes[..., :2]
        ).clamp_min(0).prod(dim=-1)[:, None, :]
        return intersection / (actor_area + proposal_area - intersection).clamp_min(1e-6)

    def _dense_proposals(self, dense_output, projected_feature):
        if dense_output is None:
            raise ValueError("identity transport requires the dense P3 output")
        _, reg_logits, obj_logits = dense_output[:3]
        reg_logits = reg_logits.detach()
        obj_logits = obj_logits.detach()
        batch, _, time, height, width = reg_logits.shape
        if time != self.frames:
            target_size = (self.frames, height, width)
            reg_logits = F.interpolate(
                reg_logits, size=target_size, mode="trilinear",
                align_corners=False,
            )
            obj_logits = F.interpolate(
                obj_logits, size=target_size, mode="trilinear",
                align_corners=False,
            )
            time = self.frames
        if projected_feature.shape[2:] != (time, height, width):
            projected_feature = F.interpolate(
                projected_feature, size=(time, height, width),
                mode="trilinear", align_corners=False,
            )

        count = min(self.transport_proposals, height * width)
        scores = obj_logits[:, 0].flatten(2)
        indices = scores.topk(count, dim=-1).indices
        y_index = torch.div(indices, width, rounding_mode="floor")
        x_index = indices.remainder(width)
        regression = reg_logits.permute(0, 2, 3, 4, 1).reshape(
            batch, time, -1, 4
        )
        regression = regression.gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, 4)
        )
        center_x = (x_index.to(regression.dtype) + regression[..., 0].sigmoid()) / width
        center_y = (y_index.to(regression.dtype) + regression[..., 1].sigmoid()) / height
        box_width = regression[..., 2].clamp(max=5).exp() / width
        box_height = regression[..., 3].clamp(max=5).exp() / height
        boxes = torch.stack([
            center_x - box_width * 0.5,
            center_y - box_height * 0.5,
            center_x + box_width * 0.5,
            center_y + box_height * 0.5,
        ], dim=-1).clamp(0, 1)

        dense_features = projected_feature.permute(0, 2, 3, 4, 1).reshape(
            batch, time, -1, projected_feature.shape[1]
        )
        proposal_features = dense_features.gather(
            2,
            indices.unsqueeze(-1).expand(
                -1, -1, -1, projected_feature.shape[1]
            ),
        )
        proposal_scores = scores.gather(2, indices).sigmoid()
        return boxes, proposal_features, proposal_scores

    def _partial_sinkhorn(self, logits):
        log_transport = logits / self.transport_temperature
        for _ in range(self.transport_sinkhorn_iterations):
            log_transport = log_transport - torch.logsumexp(
                log_transport, dim=-1, keepdim=True
            )
            real = log_transport[..., :-1]
            column_mass = torch.logsumexp(real, dim=1, keepdim=True)
            real = real - column_mass.clamp_min(0.0)
            log_transport = torch.cat([
                real, log_transport[..., -1:]
            ], dim=-1)
        return log_transport.softmax(dim=-1)

    def _identity_transport(self, dense_output, projected_feature):
        proposal_boxes, proposal_features, proposal_scores = self._dense_proposals(
            dense_output, projected_feature
        )
        batch, time, proposals, hidden = proposal_features.shape
        proposal_features = self.transport_feature_norm(proposal_features)
        proposal_keys = F.normalize(proposal_features, dim=-1)
        state = self.transport_feature_norm(
            self.transport_actor_seed.expand(batch, -1, -1)
        )

        columns = max(1, int(self.num_actors ** 0.5))
        rows = (self.num_actors + columns - 1) // columns
        actor_index = torch.arange(
            self.num_actors, device=state.device, dtype=state.dtype
        )
        center_x = (actor_index.remainder(columns) + 0.5) / columns
        center_y = (torch.div(
            actor_index, columns, rounding_mode="floor"
        ) + 0.5) / rows
        centers = torch.stack([center_x, center_y], dim=-1)
        sizes = centers.new_full(centers.shape, 0.35)
        previous_boxes = torch.cat([
            centers - sizes * 0.5, centers + sizes * 0.5
        ], dim=-1).clamp(0, 1).unsqueeze(0).expand(batch, -1, -1)
        velocity = previous_boxes.new_zeros(batch, self.num_actors, 2)
        cost_weights = F.softplus(self.transport_cost_weights)
        update_rate = self.transport_update_logit.sigmoid()
        contexts, boxes, confidences, assignments = [], [], [], []

        for time_index in range(time):
            current_boxes = proposal_boxes[:, time_index]
            current_features = proposal_features[:, time_index]
            current_keys = proposal_keys[:, time_index]
            current_scores = proposal_scores[:, time_index]
            predicted_boxes = previous_boxes + torch.cat([
                velocity, velocity
            ], dim=-1)
            predicted_boxes = predicted_boxes.clamp(0, 1)
            predicted_centers = 0.5 * (
                predicted_boxes[..., :2] + predicted_boxes[..., 2:]
            )
            proposal_centers = 0.5 * (
                current_boxes[..., :2] + current_boxes[..., 2:]
            )
            appearance = torch.einsum(
                "bah,bkh->bak", F.normalize(state, dim=-1), current_keys
            )
            distance = (
                predicted_centers[:, :, None] - proposal_centers[:, None]
            ).pow(2).sum(dim=-1)
            overlap = self._pairwise_box_iou(predicted_boxes, current_boxes)
            real_logits = (
                cost_weights[0] * appearance
                - cost_weights[1] * distance
                + cost_weights[2] * overlap
                + cost_weights[3] * current_scores[:, None]
            )
            dustbin = self.transport_dustbin_logit + (
                1.0 - current_scores.max(dim=-1, keepdim=True).values
            )
            dustbin = dustbin[:, None].expand(-1, self.num_actors, -1)
            assignment = self._partial_sinkhorn(torch.cat([
                real_logits, dustbin
            ], dim=-1))
            real_assignment = assignment[..., :-1]
            confidence = real_assignment.sum(dim=-1).clamp(0, 1)
            normalizer = confidence.unsqueeze(-1).clamp_min(1e-6)
            transported_feature = torch.einsum(
                "bak,bkh->bah", real_assignment, current_features
            ) / normalizer
            transported_box = torch.einsum(
                "bak,bkd->bad", real_assignment, current_boxes
            ) / normalizer
            transported_box = (
                confidence.unsqueeze(-1) * transported_box
                + (1.0 - confidence.unsqueeze(-1)) * predicted_boxes
            ).clamp(0, 1)
            blend = (update_rate * confidence).unsqueeze(-1)
            state = self.transport_feature_norm(
                state + blend * (transported_feature - state)
            )
            new_centers = 0.5 * (
                transported_box[..., :2] + transported_box[..., 2:]
            )
            velocity = new_centers - predicted_centers
            previous_boxes = transported_box
            context = self.transport_context_norm(
                state + self.transport_box_embed(transported_box)
            )
            contexts.append(context)
            boxes.append(transported_box)
            confidences.append(confidence)
            assignments.append(assignment)

        return {
            "context": torch.stack(contexts, dim=2),
            "boxes": torch.stack(boxes, dim=2),
            "confidence": torch.stack(confidences, dim=2),
            "assignment": torch.stack(assignments, dim=2),
        }

    def _action_reset_context(self, tokens):
        previous = torch.cat([tokens[:, :, :1], tokens[:, :, :-1]], dim=2)
        reset_features = self.action_reset_norm(tokens - previous)
        reset_logits = self.action_reset_head(reset_features).squeeze(-1)
        batch, queries, time, hidden = tokens.shape
        state = tokens.new_zeros(batch * queries, hidden)
        states = []
        for time_index in range(time):
            reset = reset_logits[:, :, time_index].sigmoid().reshape(-1, 1)
            state = state * (1.0 - reset)
            state = self.action_state_cell(
                tokens[:, :, time_index].reshape(-1, hidden), state
            )
            states.append(state.reshape(batch, queries, hidden))
        action_state = self.action_state_proj(torch.stack(states, dim=2))
        tokens = tokens + self.action_state_scale.sigmoid() * action_state
        return tokens, reset_logits

    def _action_fork_context(self, tokens):
        """Route one persistent actor identity into ordered action instances."""
        batch, _, time, hidden = tokens.shape
        actor_tokens = tokens.reshape(
            batch, self.num_actors, self.instances_per_actor, time, hidden
        )
        actor_context = actor_tokens.mean(dim=2)
        previous = torch.cat([
            actor_context[:, :, :1], actor_context[:, :, :-1]
        ], dim=2)
        fork_features = self.action_fork_norm(actor_context - previous)
        fork_logits = self.action_fork_head(fork_features).squeeze(-1)

        hazards = fork_logits.sigmoid()
        hazards = torch.cat([torch.zeros_like(hazards[:, :, :1]), hazards[:, :, 1:]], dim=2)
        progress = hazards.cumsum(dim=2).clamp(max=self.instances_per_actor - 1)
        centers = torch.arange(
            self.instances_per_actor,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        assignment_logits = -(
            progress.unsqueeze(-1) - centers
        ).square() / self.action_fork_temperature
        assignment = assignment_logits.softmax(dim=-1).permute(0, 1, 3, 2)

        if self.action_fork_mode != "visibility":
            routed_state = self.action_fork_state_proj(actor_context)
            actor_tokens = actor_tokens + (
                self.action_fork_state_scale.sigmoid()
                * assignment.unsqueeze(-1)
                * routed_state[:, :, None]
            )
        return (
            actor_tokens.reshape(batch, self.num_queries, time, hidden),
            fork_logits,
            assignment.reshape(batch, self.num_queries, time),
        )

    @staticmethod
    def _boxes_from_raw(raw):
        center = raw[..., :2].sigmoid()
        size = raw[..., 2:].sigmoid()
        return torch.cat([center - 0.5 * size, center + 0.5 * size], dim=-1).clamp(0, 1)

    @staticmethod
    def _refine_boxes(boxes, delta):
        center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4)
        center = center + 0.25 * size * delta[..., :2].tanh()
        size = size * (0.5 * delta[..., 2:].tanh()).exp()
        return torch.cat([center - 0.5 * size, center + 0.5 * size], dim=-1).clamp(0, 1)

    def _motion_context(self, features):
        contexts = []
        for feature in features:
            appearance = feature.mean(dim=(-1, -2)).permute(0, 2, 1)
            delta = torch.zeros_like(appearance)
            delta[:, 1:] = appearance[:, 1:] - appearance[:, :-1]
            contexts.append(self.motion_proj(
                torch.cat([appearance, delta.abs()], dim=-1)
            ))
        weights = self.motion_scale_logits.softmax(dim=0)
        return sum(weight * context for weight, context in zip(weights, contexts))

    @staticmethod
    def _interval_support(start_logits, end_logits):
        start_cdf = start_logits.softmax(dim=-1).cumsum(dim=-1)
        end_survival = end_logits.softmax(dim=-1).flip(-1).cumsum(dim=-1).flip(-1)
        return (start_cdf * end_survival).clamp(1e-5, 1.0)

    def _change_point_residuals(self, tokens, raw_visibility_logits):
        actor_context = tokens.mean(dim=2)
        if self.change_point_router_mode == "equal":
            route_weights = actor_context.new_full(
                (*actor_context.shape[:2], len(self.change_point_dilations)),
                1.0 / len(self.change_point_dilations),
            )
        else:
            route_input = actor_context
            if self.change_point_router_mode == "actor_duration":
                estimated_duration = (
                    raw_visibility_logits.sigmoid()
                    .mean(dim=2, keepdim=True)
                    .detach()
                )
                route_input = torch.cat([route_input, estimated_duration], dim=-1)
            route_weights = self.change_point_router(route_input).softmax(dim=-1)
        start_evidence = []
        end_evidence = []
        time = tokens.shape[2]
        if self.change_point_shared_projection:
            projections = (
                (dilation, projection, projection)
                for dilation, projection in zip(
                    self.change_point_dilations, self.change_point_shared
                )
            )
        else:
            projections = zip(
                self.change_point_dilations,
                self.change_point_start,
                self.change_point_end,
            )
        for dilation, start_projection, end_projection in projections:
            offset = min(dilation, max(time - 1, 1))
            previous = torch.cat([
                tokens[:, :, :1].expand(-1, -1, offset, -1),
                tokens[:, :, :-offset],
            ], dim=2)
            following = torch.cat([
                tokens[:, :, offset:],
                tokens[:, :, -1:].expand(-1, -1, offset, -1),
            ], dim=2)
            onset = self.change_point_norm(tokens - previous)
            offset_features = self.change_point_norm(tokens - following)
            start_evidence.append(start_projection(onset).squeeze(-1))
            end_evidence.append(end_projection(offset_features).squeeze(-1))
        start_evidence = torch.stack(start_evidence, dim=-1)
        end_evidence = torch.stack(end_evidence, dim=-1)
        gate = self.change_point_scale.sigmoid()
        weights = route_weights.unsqueeze(2)
        return (
            gate * (start_evidence * weights).sum(dim=-1),
            gate * (end_evidence * weights).sum(dim=-1),
            route_weights,
        )

    def _drop_path(self, previous, updated, layer_index):
        probability = self.drop_path_rate * (layer_index + 1) / len(self.layers)
        if not self.training or probability <= 0:
            return updated
        keep_probability = 1.0 - probability
        shape = (updated.shape[0],) + (1,) * (updated.ndim - 1)
        keep = torch.empty(shape, dtype=updated.dtype, device=updated.device)
        keep.bernoulli_(keep_probability)
        return previous + keep * (updated - previous) / keep_probability

    def _trajectory_features(self, features, boxes):
        center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        size = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-4)
        points = center.unsqueeze(-2) + 0.5 * size.unsqueeze(-2) * self.trajectory_offsets
        points = points.clamp(0, 1)
        sampled_scales = []
        for feature in features:
            batch, hidden, time, height, width = feature.shape
            grid = (points * 2.0 - 1.0).permute(0, 2, 1, 3, 4).reshape(
                batch * time, self.num_queries * points.shape[-2], 1, 2
            )
            maps = feature.permute(0, 2, 1, 3, 4).reshape(
                batch * time, hidden, height, width
            )
            sampled = F.grid_sample(
                maps, grid, mode="bilinear", padding_mode="border",
                align_corners=False,
            )[:, :, :, 0]
            sampled = sampled.permute(0, 2, 1).reshape(
                batch, time, self.num_queries, points.shape[-2], hidden
            ).permute(0, 2, 1, 3, 4)
            sampled_scales.append(sampled.mean(dim=3))
        return torch.stack(sampled_scales, dim=0).mean(dim=0)

    def _decision_tokens(self, tokens, decision_context):
        if decision_context is None:
            return tokens
        if self.decision_memory_projection is None:
            raise ValueError(
                "decision_context was provided without decision memory"
            )
        context = decision_context
        if context.shape[1] != self.frames:
            context = F.interpolate(
                context.transpose(1, 2),
                size=self.frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        residual = self.decision_memory_projection(context).unsqueeze(1)
        return tokens + self.decision_memory_scale.tanh() * residual

    def forward(self, features, dense_output=None, decision_context=None):
        memory, projected_features = self._memory(features)
        batch = memory.shape[0]
        transport = None
        if self.interval_queries:
            tokens = self.person_embed.expand(
                batch, -1, self.instances_per_actor, self.frames, -1
            )
            tokens = tokens + self.instance_embed
            tokens = tokens + self.time_embed[:, :, :, :self.frames]
            if self.identity_transport_enabled:
                transport = self._identity_transport(
                    dense_output, projected_features[0]
                )
                tokens = tokens + self.transport_scale.sigmoid() * (
                    transport["context"][:, :, None]
                )
            if self.motion_proj is not None:
                motion_context = self._motion_context(projected_features)
                tokens = tokens + self.motion_condition_scale.tanh() * (
                    motion_context[:, None, None]
                )
        else:
            tokens = self.person_embed.expand(batch, -1, self.frames, -1)
            tokens = tokens + self.time_embed[:, :, :self.frames]
        boxes = None
        for layer_index, layer in enumerate(self.layers):
            if boxes is not None and self.box_condition is not None:
                condition = self.box_condition(boxes)
                if self.interval_queries:
                    condition = condition.reshape(
                        batch, self.num_actors, self.instances_per_actor,
                        self.frames, -1,
                    )
                tokens = tokens + self.box_condition_scale.tanh() * condition
            if boxes is not None and self.trajectory_sampling:
                trajectory = self._trajectory_features(projected_features, boxes)
                trajectory = self.trajectory_norm(trajectory)
                if self.interval_queries:
                    trajectory = trajectory.reshape(
                        batch, self.num_actors, self.instances_per_actor,
                        self.frames, -1,
                    )
                tokens = tokens + self.trajectory_scale.tanh() * trajectory
            previous_tokens = tokens
            tokens = self._drop_path(
                previous_tokens, layer(tokens, memory), layer_index
            )
            if layer_index + 1 < len(self.layers) and (
                    self.iterative_refinement or self.trajectory_sampling):
                intermediate = self.output_norm(tokens)
                if self.interval_queries:
                    intermediate = intermediate.reshape(
                        batch, self.num_queries, self.frames, -1
                    )
                boxes = self._boxes_from_raw(self.box_head(intermediate))
                if self.refinement_heads is not None:
                    boxes = self._refine_boxes(
                        boxes, self.refinement_heads[layer_index](intermediate)
                    )
        tokens = self.output_norm(tokens)
        if self.interval_queries:
            tokens = tokens.reshape(batch, self.num_queries, self.frames, -1)
        action_reset_logits = None
        if self.action_reset_state_enabled:
            tokens, action_reset_logits = self._action_reset_context(tokens)
        action_fork_logits = None
        action_fork_assignment = None
        if self.action_fork_state_enabled:
            tokens, action_fork_logits, action_fork_assignment = (
                self._action_fork_context(tokens)
            )
        duration_route_weights = None
        if self.duration_router is not None:
            provisional_visibility = self.visibility_head(tokens).squeeze(-1).sigmoid()
            estimated_duration = provisional_visibility.mean(dim=-1, keepdim=True)
            router_input = torch.cat([tokens.mean(dim=2), estimated_duration], dim=-1)
            duration_route_weights = self.duration_router(router_input).softmax(dim=-1)
            sequence = tokens.reshape(
                batch * self.num_queries, self.frames, -1
            ).transpose(1, 2)
            contexts = torch.stack([
                temporal_filter(sequence).transpose(1, 2).reshape_as(tokens)
                for temporal_filter in self.duration_filters
            ], dim=2)
            routed = (
                contexts * duration_route_weights[:, :, :, None, None]
            ).sum(dim=2)
            tokens = tokens + self.duration_router_scale.tanh() * self.duration_mix(routed)
        decision_tokens = self._decision_tokens(tokens, decision_context)
        class_tokens = (
            decision_tokens
            if self.decision_memory_target in {
                "class", "class_boundary", "class_quality",
                "class_quality_residual",
            }
            else tokens
        )
        boundary_tokens = (
            decision_tokens
            if self.decision_memory_target in {"boundary", "class_boundary"}
            else tokens
        )
        quality_tokens = (
            decision_tokens
            if self.decision_memory_target in {
                "quality", "class_quality", "quality_residual",
                "class_quality_residual",
            }
            else tokens
        )
        boxes = self._boxes_from_raw(self.box_head(tokens))
        if self.refinement_heads is not None:
            boxes = self._refine_boxes(boxes, self.refinement_heads[-1](tokens))
        raw_visibility_logits = self.visibility_head(
            boundary_tokens
        ).squeeze(-1)
        if action_fork_assignment is not None:
            ownership_bias = torch.log(
                (action_fork_assignment * self.instances_per_actor).clamp_min(1e-5)
            )
            raw_visibility_logits = raw_visibility_logits + (
                self.action_fork_visibility_scale.sigmoid() * ownership_bias
            )
        start_logits = None
        end_logits = None
        interval_support = None
        boundary_distances = None
        boundary_distance_support = None
        change_point_route_weights = None
        if self.interval_queries:
            start_logits = self.start_head(boundary_tokens).squeeze(-1)
            end_logits = self.end_head(boundary_tokens).squeeze(-1)
            if action_reset_logits is not None:
                start_logits = start_logits + (
                    self.action_reset_scale.sigmoid() * action_reset_logits
                )
            if (
                    action_fork_assignment is not None
                    and self.action_fork_mode == "state_boundary"):
                previous_ownership = torch.cat([
                    torch.zeros_like(action_fork_assignment[:, :, :1]),
                    action_fork_assignment[:, :, :-1],
                ], dim=2)
                next_ownership = torch.cat([
                    action_fork_assignment[:, :, 1:],
                    torch.zeros_like(action_fork_assignment[:, :, :1]),
                ], dim=2)
                boundary_gate = self.action_fork_boundary_scale.sigmoid()
                start_logits = start_logits + 4.0 * boundary_gate * (
                    action_fork_assignment - previous_ownership
                ).clamp_min(0)
                end_logits = end_logits + 4.0 * boundary_gate * (
                    action_fork_assignment - next_ownership
                ).clamp_min(0)
            if self.change_point_pyramid_enabled:
                start_residual, end_residual, change_point_route_weights = (
                    self._change_point_residuals(
                        boundary_tokens, raw_visibility_logits
                    )
                )
                start_logits = start_logits + start_residual
                end_logits = end_logits + end_residual
            if self.boundary_distance_head is not None:
                boundary_distances = torch.tanh(
                    self.boundary_distance_head(boundary_tokens)
                )
                distance_gate = self.boundary_distance_scale.sigmoid()
                temperature = self.boundary_distance_temperature
                start_logits = start_logits - distance_gate * (
                    boundary_distances[..., 0].abs() / temperature
                )
                end_logits = end_logits - distance_gate * (
                    boundary_distances[..., 1].abs() / temperature
                )
            start_logits = start_logits + self.start_prior[:self.frames]
            end_logits = end_logits + self.end_prior[:self.frames]
            interval_support = self._interval_support(start_logits, end_logits)
            if boundary_distances is not None:
                boundary_distance_support = (
                    (boundary_distances[..., 0] / temperature).sigmoid()
                    * (boundary_distances[..., 1] / temperature).sigmoid()
                )
                interval_support = (
                    (1.0 - distance_gate) * interval_support
                    + distance_gate * boundary_distance_support
                ).clamp(1e-5, 1.0)
            interval_gate = self.interval_visibility_gate.sigmoid()
            visibility_factor = 1.0 - interval_gate * (1.0 - interval_support)
            visibility_probability = (
                raw_visibility_logits.sigmoid() * visibility_factor
            ).clamp(1e-5, 1.0 - 1e-5)
            visibility_logits = torch.logit(visibility_probability)
        else:
            visibility_logits = raw_visibility_logits
        visibility_weights = visibility_logits.sigmoid().unsqueeze(-1)
        pooled = (class_tokens * visibility_weights).sum(dim=2)
        pooled = pooled / visibility_weights.sum(dim=2).clamp_min(1e-4)
        normalized_pooled = self.person_norm(pooled)
        output = {
            "class_logits": self.class_head(normalized_pooled),
            "frame_class_logits": self.class_head(class_tokens),
            "boxes": boxes,
            "visibility_logits": visibility_logits,
            "boundary_logits": self.boundary_head(
                boundary_tokens
            ).squeeze(-1),
        }
        if self.quality_head is not None:
            quality_pooled = (quality_tokens * visibility_weights).sum(dim=2)
            quality_pooled = (
                quality_pooled
                / visibility_weights.sum(dim=2).clamp_min(1e-4)
            )
            raw_quality_logits = self.quality_head(
                self.person_norm(quality_pooled)
            ).squeeze(-1)
            if self.quality_residual_scale is not None:
                output["quality_raw_logits"] = raw_quality_logits
                output["quality_logits"] = (
                    self.quality_residual_base_logit
                    + self.quality_residual_scale.tanh() * raw_quality_logits
                )
            else:
                output["quality_logits"] = raw_quality_logits
        if duration_route_weights is not None:
            output["duration_route_weights"] = duration_route_weights
        if change_point_route_weights is not None:
            output["change_point_route_weights"] = change_point_route_weights
        if transport is not None:
            output.update({
                "transport_boxes": transport["boxes"].repeat_interleave(
                    self.instances_per_actor, dim=1
                ),
                "transport_confidence": transport["confidence"].repeat_interleave(
                    self.instances_per_actor, dim=1
                ),
                "transport_assignment": transport["assignment"],
            })
        if action_reset_logits is not None:
            output["action_reset_logits"] = action_reset_logits
        if action_fork_logits is not None:
            output["action_fork_logits"] = action_fork_logits
            output["action_fork_assignment"] = action_fork_assignment
        if self.interval_queries:
            output.update({
                "raw_visibility_logits": raw_visibility_logits,
                "start_logits": start_logits,
                "end_logits": end_logits,
                "interval_support": interval_support,
            })
            if boundary_distances is not None:
                output.update({
                    "boundary_distances": boundary_distances,
                    "boundary_distance_support": boundary_distance_support,
                })
        return output


class ActorAlignedTubeQueryHead(nn.Module):
    """Iteratively sample multi-scale actor features around dense proposals."""

    def __init__(self, channels, num_classes, hidden=256, num_queries=8,
                 num_heads=8, depth=2, dropout=0.1, max_frames=128,
                 spatial_stride=8, img_size=224, feedback=True,
                 deformable_points=1, boundary_recurrent=False):
        super().__init__()
        self.num_queries = int(num_queries)
        self.max_frames = int(max_frames)
        self.spatial_stride = int(spatial_stride)
        self.img_size = int(img_size)
        self.deformable_points = max(1, int(deformable_points))
        self.boundary_recurrent = bool(boundary_recurrent)
        self.scale_proj = nn.ModuleList([nn.Linear(value, hidden) for value in channels])
        self.query_embed = nn.Parameter(torch.empty(num_queries, hidden))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, max_frames, hidden))
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(depth)
        ])
        self.box_updates = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4),
        ) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden)
        self.class_head = nn.Linear(hidden, num_classes)
        self.visibility_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Linear(hidden, 1)
        if self.deformable_points > 1:
            self.sample_offsets = nn.ModuleList([
                nn.Linear(hidden, self.deformable_points * 2) for _ in range(depth)
            ])
            self.sample_weights = nn.ModuleList([
                nn.Linear(hidden, len(channels) * self.deformable_points)
                for _ in range(depth)
            ])
        else:
            self.sample_offsets = None
            self.sample_weights = None
        if self.boundary_recurrent:
            self.recurrent_cells = nn.ModuleList([
                nn.GRUCell(hidden, hidden) for _ in range(depth)
            ])
            self.recurrent_scales = nn.Parameter(torch.zeros(depth))
        else:
            self.recurrent_cells = None
            self.recurrent_scales = None
        self.feedback_proj = nn.Linear(hidden, channels[0]) if feedback else None
        self.feedback_scale = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.query_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        for update in self.box_updates:
            nn.init.zeros_(update[-1].weight)
            nn.init.zeros_(update[-1].bias)
        if self.sample_offsets is not None:
            pattern = torch.tensor([
                [0.0, 0.0], [-0.5, -0.5], [0.5, -0.5],
                [-0.5, 0.5], [0.5, 0.5],
            ])
            for offset, weight in zip(self.sample_offsets, self.sample_weights):
                nn.init.zeros_(offset.weight)
                nn.init.zeros_(offset.bias)
                count = min(self.deformable_points, pattern.shape[0])
                with torch.no_grad():
                    offset.bias.view(self.deformable_points, 2)[:count].copy_(
                        torch.atanh(pattern[:count].clamp(-0.99, 0.99))
                    )
                nn.init.zeros_(weight.weight)
                nn.init.zeros_(weight.bias)

    def _proposal_boxes(self, dense_output):
        _, reg_logits, obj_logits = dense_output[:3]
        batch, _, time, height, width = reg_logits.shape
        count = min(self.num_queries, height * width)
        scores = obj_logits[:, 0].flatten(2)
        indices = scores.topk(count, dim=-1).indices
        y_index = torch.div(indices, width, rounding_mode="floor")
        x_index = indices.remainder(width)
        reg = reg_logits.permute(0, 2, 3, 4, 1).reshape(batch, time, -1, 4)
        reg = reg.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 4))
        step = self.spatial_stride / self.img_size
        center_x = (x_index.to(reg.dtype) + reg[..., 0].sigmoid()) * step
        center_y = (y_index.to(reg.dtype) + reg[..., 1].sigmoid()) * step
        width_box = reg[..., 2].clamp(max=5).exp() * step
        height_box = reg[..., 3].clamp(max=5).exp() * step
        boxes = torch.stack([
            center_x - width_box * 0.5, center_y - height_box * 0.5,
            center_x + width_box * 0.5, center_y + height_box * 0.5,
        ], dim=-1).clamp(0, 1)
        if count < self.num_queries:
            boxes = torch.cat([
                boxes,
                boxes.new_full((batch, time, self.num_queries - count, 4), 0.5),
            ], dim=2)
        return boxes.permute(0, 2, 1, 3).contiguous(), indices

    @staticmethod
    def _centers(boxes):
        return 0.5 * (boxes[..., :2] + boxes[..., 2:])

    def _sample(self, feature, boxes, projection, target_time):
        if feature.shape[2] != target_time:
            feature = F.interpolate(
                feature, size=(target_time, feature.shape[3], feature.shape[4]),
                mode="trilinear", align_corners=False,
            )
        batch, channels, time, height, width = feature.shape
        centers = self._centers(boxes).clamp(0, 1)
        grid = (centers * 2 - 1).permute(0, 2, 1, 3).reshape(
            batch * time, self.num_queries, 1, 2
        )
        maps = feature.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        sampled = F.grid_sample(
            maps, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[:, :, :, 0]
        sampled = sampled.permute(0, 2, 1).reshape(
            batch, time, self.num_queries, channels
        ).permute(0, 2, 1, 3)
        return projection(sampled)

    def _sample_points(self, feature, points, projection, target_time):
        """Sample proposal-relative points from one pyramid level."""
        if feature.shape[2] != target_time:
            feature = F.interpolate(
                feature, size=(target_time, feature.shape[3], feature.shape[4]),
                mode="trilinear", align_corners=False,
            )
        batch, channels, time, height, width = feature.shape
        points = points.clamp(0, 1)
        grid = (points * 2 - 1).permute(0, 2, 1, 3, 4).reshape(
            batch * time, self.num_queries * self.deformable_points, 1, 2
        )
        maps = feature.permute(0, 2, 1, 3, 4).reshape(
            batch * time, channels, height, width
        )
        sampled = F.grid_sample(
            maps, grid, mode="bilinear", padding_mode="border", align_corners=False
        )[:, :, :, 0]
        sampled = sampled.permute(0, 2, 1).reshape(
            batch, time, self.num_queries, self.deformable_points, channels
        ).permute(0, 2, 1, 3, 4)
        return projection(sampled)

    def _deformable_sample(self, features, boxes, tokens, layer_index, target_time):
        centers = self._centers(boxes).unsqueeze(-2)
        sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-3).unsqueeze(-2)
        offsets = self.sample_offsets[layer_index](tokens).reshape(
            *tokens.shape[:-1], self.deformable_points, 2
        ).tanh()
        points = centers + 0.5 * sizes * offsets
        sampled = torch.stack([
            self._sample_points(feature, points, projection, target_time)
            for feature, projection in zip(features, self.scale_proj)
        ], dim=3)
        batch, queries, time, scales, points_count, hidden = sampled.shape
        sampled = sampled.reshape(batch, queries, time, scales * points_count, hidden)
        weights = self.sample_weights[layer_index](tokens).softmax(dim=-1).unsqueeze(-1)
        return (sampled * weights).sum(dim=3)

    def _recurrent_update(self, tokens, layer_index):
        batch, queries, time, hidden = tokens.shape
        state = tokens.new_zeros(batch, queries, hidden)
        sequence = []
        cell = self.recurrent_cells[layer_index]
        for time_index in range(time):
            current = tokens[:, :, time_index]
            reset = self.boundary_head(current).sigmoid()
            state = state * (1.0 - reset)
            state = cell(
                current.reshape(batch * queries, hidden),
                state.reshape(batch * queries, hidden),
            ).reshape(batch, queries, hidden)
            sequence.append(state)
        recurrent = torch.stack(sequence, dim=2)
        scale = self.recurrent_scales[layer_index].tanh()
        return tokens + scale * recurrent

    @staticmethod
    def _refine_boxes(boxes, delta):
        centers = 0.5 * (boxes[..., :2] + boxes[..., 2:])
        sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-3)
        centers = centers + 0.25 * sizes * delta[..., :2].tanh()
        sizes = sizes * (0.5 * delta[..., 2:].tanh()).exp()
        return torch.cat([centers - sizes * 0.5, centers + sizes * 0.5], -1).clamp(0, 1)

    def _feedback(self, tokens, boxes, spatial_size):
        if self.feedback_proj is None:
            return None
        height, width = spatial_size
        centers = self._centers(boxes)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=tokens.device, dtype=tokens.dtype),
            torch.linspace(0, 1, width, device=tokens.device, dtype=tokens.dtype),
            indexing="ij",
        )
        coordinates = torch.stack([xx, yy], -1)
        distance = (centers.unsqueeze(-2).unsqueeze(-2) - coordinates).pow(2).sum(-1)
        weights = (-distance / 0.01).exp()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        values = self.feedback_proj(tokens)
        feedback = torch.einsum("bqthw,bqtc->bcthw", weights, values)
        return self.feedback_scale.tanh() * feedback

    def forward(self, features, dense_output):
        target_time = features[0].shape[2]
        if target_time > self.max_frames:
            raise ValueError(f"tube query time {target_time} exceeds {self.max_frames}")
        boxes, proposal_indices = self._proposal_boxes(dense_output)
        boxes = boxes.detach()
        tokens = self.query_embed.view(1, self.num_queries, 1, -1)
        tokens = tokens + self.time_embed[:, :, :target_time]
        for layer_index, (layer, box_update) in enumerate(
                zip(self.temporal_layers, self.box_updates)):
            if self.sample_offsets is not None:
                sampled = self._deformable_sample(
                    features, boxes, tokens, layer_index, target_time
                )
            else:
                sampled = sum(
                    self._sample(feature, boxes, projection, target_time)
                    for feature, projection in zip(features, self.scale_proj)
                ) / len(features)
            tokens = tokens + sampled
            if self.recurrent_cells is not None:
                tokens = self._recurrent_update(tokens, layer_index)
            batch, queries, time, hidden = tokens.shape
            tokens = layer(tokens.reshape(batch * queries, time, hidden)).reshape(
                batch, queries, time, hidden
            )
            boxes = self._refine_boxes(boxes, box_update(tokens))
        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=2)
        return {
            "class_logits": self.class_head(pooled),
            "frame_class_logits": self.class_head(tokens),
            "boxes": boxes,
            "visibility_logits": self.visibility_head(tokens).squeeze(-1),
            "boundary_logits": self.boundary_head(tokens).squeeze(-1),
            "feedback": self._feedback(tokens, boxes, features[0].shape[-2:]),
            "proposal_indices": proposal_indices,
        }


def _box_iou(box, boxes):
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    inter = (rb - lt).clamp_min(0).prod(dim=-1)
    area = (box[2:] - box[:2]).clamp_min(0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=-1)
    return inter / (area + areas - inter).clamp_min(1e-6)


def _aligned_giou(boxes1, boxes2):
    lt_i = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb_i = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter = (rb_i - lt_i).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    union = (area1 + area2 - inter).clamp_min(1e-7)
    iou = inter / union
    lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclosing = (rb - lt).clamp_min(0).prod(dim=-1).clamp_min(1e-7)
    return iou - (enclosing - union) / enclosing


def _aligned_iou(boxes1, boxes2):
    lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
    intersection = (rb - lt).clamp_min(0).prod(dim=-1)
    area1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp_min(0).prod(dim=-1)
    area2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-7)


def _soft_interval_iou(prediction, target):
    target = target.to(dtype=prediction.dtype)
    intersection = (prediction * target).sum(dim=-1)
    union = (prediction + target - prediction * target).sum(dim=-1)
    return intersection / union.clamp_min(1e-6)


def _interval_coverage_error(prediction, target):
    target = target.to(dtype=prediction.dtype)
    missed = (target * (1.0 - prediction)).sum(dim=-1)
    missed = missed / target.sum(dim=-1).clamp_min(1.0)
    outside = 1.0 - target
    spill = (outside * prediction).sum(dim=-1)
    spill = spill / outside.sum(dim=-1).clamp_min(1.0)
    return 0.5 * (missed + spill)


def _interval_fragmentation_error(prediction, target):
    target = target.to(dtype=prediction.dtype)
    if prediction.shape[-1] < 2:
        return prediction.new_zeros(prediction.shape[:-1])
    predicted_edges = (prediction[..., 1:] - prediction[..., :-1]).abs()
    target_edges = (target[..., 1:] - target[..., :-1]).abs()
    return (predicted_edges - target_edges).abs().mean(dim=-1)


def _sigmoid_focal_loss(logits, target, alpha, gamma):
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    target_probability = probability * target + (1.0 - probability) * (1.0 - target)
    loss = cross_entropy * (1.0 - target_probability).pow(float(gamma))
    if alpha is not None and float(alpha) >= 0:
        alpha_weight = float(alpha) * target + (1.0 - float(alpha)) * (1.0 - target)
        loss = alpha_weight * loss
    return loss


def _link_tubes(boxes, labels, max_queries, max_track_gap=3,
                track_ids=None, observation_scores=None,
                track_qualities=None):
    if labels.ndim > 1:
        labels = labels.argmax(dim=-1)
    valid = boxes[:, 1:5].sum(dim=-1) > 0
    if track_ids is not None:
        valid = valid & (track_ids >= 0)
        track_ids = track_ids[valid]
    if observation_scores is not None:
        observation_scores = observation_scores[valid]
    if track_qualities is not None:
        track_qualities = track_qualities[valid]
    boxes, labels = boxes[valid], labels[valid]
    if boxes.numel() == 0:
        return []
    order = boxes[:, 0].argsort()
    boxes, labels = boxes[order], labels[order]
    if track_ids is not None:
        track_ids = track_ids[order]
        if observation_scores is not None:
            observation_scores = observation_scores[order]
        if track_qualities is not None:
            track_qualities = track_qualities[order]
        tracks = []
        for track_id in track_ids.unique(sorted=True):
            mask = track_ids == track_id
            track_boxes = boxes[mask]
            track_labels = labels[mask]
            track = {
                "label": int(track_labels[0].item()),
                "last_frame": int(track_boxes[-1, 0].item()),
                "last_box": track_boxes[-1, 1:5],
                "frames": [int(value) for value in track_boxes[:, 0].tolist()],
                "boxes": list(track_boxes[:, 1:5]),
            }
            if observation_scores is not None:
                track["scores"] = list(observation_scores[mask])
            if track_qualities is not None:
                track["qualities"] = list(track_qualities[mask])
            tracks.append(track)
        tracks.sort(key=lambda track: len(track["frames"]), reverse=True)
        return tracks[:max_queries]
    tracks = []
    for frame in boxes[:, 0].long().unique(sorted=True):
        mask = boxes[:, 0].long() == frame
        used = set()
        for box, label in zip(boxes[mask, 1:5], labels[mask]):
            label = int(label.item())
            candidates = [
                i for i, track in enumerate(tracks)
                if track["label"] == label and i not in used
                and int(frame.item()) - track["last_frame"] <= max_track_gap
            ]
            selected = None
            if candidates:
                ious = _box_iou(box, torch.stack([tracks[i]["last_box"] for i in candidates]))
                best = int(ious.argmax().item())
                if float(ious[best]) >= 0.1:
                    selected = candidates[best]
            if selected is None:
                tracks.append({"label": label, "last_frame": int(frame.item()),
                               "last_box": box, "frames": [int(frame.item())],
                               "boxes": [box]})
                used.add(len(tracks) - 1)
            else:
                track = tracks[selected]
                track["last_frame"], track["last_box"] = int(frame.item()), box
                track["frames"].append(int(frame.item()))
                track["boxes"].append(box)
                used.add(selected)
    tracks.sort(key=lambda track: len(track["frames"]), reverse=True)
    return tracks[:max_queries]


def _tube_targets(targets, batch_index, time, clip_length, max_queries, dtype, device):
    track_ids = targets.get("track_ids")
    if track_ids is not None:
        track_ids = track_ids[batch_index]
    observation_scores = targets.get("scores")
    if observation_scores is not None:
        observation_scores = observation_scores[batch_index]
    track_qualities = targets.get("quality")
    if track_qualities is not None:
        track_qualities = track_qualities[batch_index]
    tracks = _link_tubes(
        targets["boxes"][batch_index], targets["labels"][batch_index],
        max_queries, track_ids=track_ids,
        observation_scores=observation_scores,
        track_qualities=track_qualities,
    )
    result = []
    for track in tracks:
        boxes = torch.zeros(time, 4, dtype=dtype, device=device)
        visible = torch.zeros(time, dtype=torch.bool, device=device)
        frames = torch.tensor(track["frames"], device=device)
        indices = torch.round(frames.float() * (time - 1) / max(clip_length - 1, 1))
        indices = indices.long().clamp(0, time - 1)
        boxes[indices] = torch.stack(track["boxes"]).to(device=device, dtype=dtype)
        visible[indices] = True
        observation_confidence = torch.ones(
            time, dtype=dtype, device=device
        )
        if "scores" in track:
            scores = torch.stack(track["scores"]).to(device=device, dtype=dtype)
            observation_confidence[indices] = scores
        track_quality = torch.ones((), dtype=dtype, device=device)
        if "qualities" in track:
            qualities = torch.stack(track["qualities"]).to(
                device=device, dtype=dtype
            )
            track_quality = qualities.mean()
        boundary = torch.zeros(time, dtype=dtype, device=device)
        valid_indices = visible.nonzero(as_tuple=True)[0]
        interval = torch.zeros(time, dtype=dtype, device=device)
        start = torch.zeros(time, dtype=dtype, device=device)
        end = torch.zeros(time, dtype=dtype, device=device)
        boundary_distance = torch.zeros(time, 2, dtype=dtype, device=device)
        if valid_indices.numel():
            start_index = valid_indices[0]
            end_index = valid_indices[-1]
            boundary[start_index] = 1.0
            boundary[end_index] = 1.0
            interval[start_index:end_index + 1] = 1.0
            start[start_index] = 1.0
            end[end_index] = 1.0
            position = torch.arange(time, dtype=dtype, device=device)
            denominator = max(time - 1, 1)
            boundary_distance[:, 0] = (
                position - start_index.to(dtype)
            ) / denominator
            boundary_distance[:, 1] = (
                end_index.to(dtype) - position
            ) / denominator
        result.append({"label": track["label"], "boxes": boxes,
                       "visible": visible, "boundary": boundary,
                       "interval": interval, "start": start, "end": end,
                       "boundary_distance": boundary_distance,
                       "observation_confidence": observation_confidence,
                       "track_quality": track_quality})
    return result


def _normalized_confidence_weights(values, floor):
    values = values.clamp(min=float(floor), max=1.0)
    values = values / values.mean().clamp_min(1e-6)
    return values.clamp(min=float(floor), max=1.0 / float(floor))


def _weighted_time_mean(values, weights):
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(0)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1e-6)


def _tube_quality_target(spatial_quality, temporal_quality,
                         mode="sqrt_product", strict_blend=0.5):
    """Return a detached tube-quality target aligned to the requested metric."""
    joint_quality = (spatial_quality * temporal_quality).clamp(0.0, 1.0)
    if mode == "sqrt_product":
        target = joint_quality.sqrt()
    elif mode == "product":
        target = joint_quality
    elif mode in {"ap50_95", "ap50_95_blend"}:
        thresholds = joint_quality.new_tensor(
            [0.50, 0.55, 0.60, 0.65, 0.70,
             0.75, 0.80, 0.85, 0.90, 0.95]
        )
        strict_target = (
            joint_quality.unsqueeze(-1) >= thresholds
        ).to(joint_quality.dtype).mean(dim=-1)
        if mode == "ap50_95":
            target = strict_target
        else:
            blend = float(strict_blend)
            if not 0.0 <= blend <= 1.0:
                raise ValueError(
                    "quality strict_blend must be in [0, 1], "
                    f"got {strict_blend}"
                )
            target = (
                blend * strict_target
                + (1.0 - blend) * joint_quality.sqrt()
            )
    else:
        raise ValueError(f"unsupported tube quality target mode: {mode!r}")
    return target.detach()


def tube_query_loss(outputs, targets, clip_length, cost_cls=2.0, cost_box=5.0,
                    cost_visibility=1.0, cost_giou=2.0,
                    boundary_pos_weight=1.0, boundary_focal_gamma=0.0,
                    cost_interval=0.0, cost_coverage=0.0,
                    cost_fragmentation=0.0, cost_transport=0.0,
                    class_focal_alpha=None,
                    class_focal_gamma=2.0,
                    target_confidence_weighting=False,
                    target_confidence_floor=0.05,
                    quality_target_mode="sqrt_product",
                    quality_strict_blend=0.5):
    """Match complete predicted/GT tubes, then return normalized component losses."""
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for tube-query Hungarian matching")
    cls_logits = outputs["class_logits"]
    pred_boxes = outputs["boxes"]
    visibility_logits = outputs["visibility_logits"]
    boundary_logits = outputs["boundary_logits"]
    quality_logits = outputs.get("quality_logits")
    geometry_parent_boxes = outputs.get("geometry_contract_parent_boxes")
    interval_support = outputs.get("interval_support")
    start_logits = outputs.get("start_logits")
    end_logits = outputs.get("end_logits")
    boundary_distances = outputs.get("boundary_distances")
    transport_boxes = outputs.get("transport_boxes")
    has_intervals = all(value is not None for value in (
        interval_support, start_logits, end_logits
    ))
    batch_size, num_queries, time, _ = pred_boxes.shape
    device, dtype = pred_boxes.device, pred_boxes.dtype
    totals = {
        name: pred_boxes.new_zeros(()) for name in (
            "cls", "box", "visibility", "boundary", "giou", "velocity",
            "acceleration", "start", "end", "interval_iou", "coverage",
            "fragmentation", "boundary_distance",
            "boundary_distance_slope", "quality", "transport",
            "geometry_preservation",
        )
    }
    matched_count = 0
    matched_weight = 0.0

    for batch_index in range(batch_size):
        tubes = _tube_targets(
            targets, batch_index, time, clip_length, num_queries, dtype, device
        )
        if target_confidence_weighting and tubes:
            qualities = torch.stack([
                tube["track_quality"] for tube in tubes
            ])
            quality_weights = _normalized_confidence_weights(
                qualities, target_confidence_floor
            )
            for tube, quality_weight in zip(tubes, quality_weights):
                tube["loss_weight"] = quality_weight
                mask = tube["visible"]
                tube["frame_weights"] = _normalized_confidence_weights(
                    tube["observation_confidence"][mask],
                    target_confidence_floor,
                )
                tube["temporal_frame_weights"] = pred_boxes.new_ones(time)
                tube["temporal_frame_weights"][mask] = tube["frame_weights"]
        else:
            for tube in tubes:
                tube["loss_weight"] = pred_boxes.new_ones(())
                tube["frame_weights"] = pred_boxes.new_ones(
                    int(tube["visible"].sum().item())
                )
                tube["temporal_frame_weights"] = pred_boxes.new_ones(time)
        visibility_target = torch.zeros_like(visibility_logits[batch_index])
        boundary_target = torch.zeros_like(boundary_logits[batch_index])
        class_target = torch.zeros_like(cls_logits[batch_index])
        quality_target = (
            torch.zeros_like(quality_logits[batch_index])
            if quality_logits is not None else None
        )
        if not tubes:
            totals["visibility"] = totals["visibility"] + F.binary_cross_entropy_with_logits(
                visibility_logits[batch_index], visibility_target
            )
            if class_focal_alpha is not None:
                totals["cls"] = totals["cls"] + _sigmoid_focal_loss(
                    cls_logits[batch_index], class_target,
                    class_focal_alpha, class_focal_gamma,
                ).sum()
            if quality_logits is not None:
                totals["quality"] = totals["quality"] + F.binary_cross_entropy_with_logits(
                    quality_logits[batch_index], quality_target
                )
            continue

        labels = torch.tensor([tube["label"] for tube in tubes], device=device)
        class_probability = cls_logits[batch_index].sigmoid()
        if class_focal_alpha is None:
            class_cost = -class_probability[:, labels]
        else:
            alpha = float(class_focal_alpha)
            gamma = float(class_focal_gamma)
            positive_cost = -alpha * (1.0 - class_probability).pow(gamma) * (
                class_probability.clamp_min(1e-8).log()
            )
            negative_cost = -(1.0 - alpha) * class_probability.pow(gamma) * (
                (1.0 - class_probability).clamp_min(1e-8).log()
            )
            class_cost = positive_cost[:, labels] - negative_cost[:, labels]
        box_costs, giou_costs, visibility_costs = [], [], []
        interval_costs, coverage_costs, fragmentation_costs = [], [], []
        transport_costs = []
        for tube in tubes:
            mask = tube["visible"]
            frame_weights = tube["frame_weights"]
            box_costs.append(
                _weighted_time_mean(
                    (
                        pred_boxes[batch_index, :, mask]
                        - tube["boxes"][mask]
                    ).abs().mean(dim=-1),
                    frame_weights,
                )
            )
            target_boxes = tube["boxes"][mask].unsqueeze(0).expand(num_queries, -1, -1)
            giou_costs.append(
                _weighted_time_mean(
                    1.0 - _aligned_giou(
                        pred_boxes[batch_index, :, mask], target_boxes
                    ),
                    frame_weights,
                )
            )
            target_vis = tube["visible"].to(dtype).expand(num_queries, -1)
            visibility_costs.append(F.binary_cross_entropy_with_logits(
                visibility_logits[batch_index], target_vis, reduction="none"
            ).mean(1))
            if transport_boxes is not None:
                transport_costs.append(
                    _weighted_time_mean(
                        (
                            transport_boxes[batch_index, :, mask]
                            - tube["boxes"][mask]
                        ).abs().mean(dim=-1),
                        frame_weights,
                    )
                )
            if has_intervals:
                target_interval = tube["interval"].expand(num_queries, -1)
                interval_costs.append(
                    1.0 - _soft_interval_iou(
                        interval_support[batch_index], target_interval
                    )
                )
                coverage_costs.append(_interval_coverage_error(
                    visibility_logits[batch_index].sigmoid(), target_interval
                ))
                fragmentation_costs.append(_interval_fragmentation_error(
                    visibility_logits[batch_index].sigmoid(), target_interval
                ))
        cost = (cost_cls * class_cost + cost_box * torch.stack(box_costs, 1)
                + cost_giou * torch.stack(giou_costs, 1)
                + cost_visibility * torch.stack(visibility_costs, 1))
        if has_intervals:
            cost = (cost + cost_interval * torch.stack(interval_costs, 1)
                    + cost_coverage * torch.stack(coverage_costs, 1)
                    + cost_fragmentation * torch.stack(fragmentation_costs, 1))
        if transport_boxes is not None:
            cost = cost + cost_transport * torch.stack(transport_costs, 1)
        rows, cols = linear_sum_assignment(cost.detach().cpu().float().numpy())
        query_indices = torch.as_tensor(rows, dtype=torch.long, device=device)
        tube_indices = torch.as_tensor(cols, dtype=torch.long, device=device)
        matched_count += len(rows)

        for query_index, tube_index in zip(query_indices, tube_indices):
            tube = tubes[int(tube_index)]
            mask = tube["visible"]
            frame_weights = tube["frame_weights"]
            tube_weight = tube["loss_weight"]
            matched_weight += float(tube_weight.detach().item())
            visibility_target[query_index] = mask.to(dtype)
            boundary_target[query_index] = tube["boundary"]
            if class_focal_alpha is None:
                totals["cls"] = totals["cls"] + F.cross_entropy(
                    cls_logits[batch_index, query_index].unsqueeze(0),
                    labels[tube_index].unsqueeze(0),
                ) * tube_weight
            else:
                class_target[query_index, labels[tube_index]] = 1.0
            box_error = F.smooth_l1_loss(
                pred_boxes[batch_index, query_index, mask],
                tube["boxes"][mask],
                reduction="none",
                beta=0.05,
            ).mean(dim=-1)
            totals["box"] = totals["box"] + tube_weight * _weighted_time_mean(
                box_error, frame_weights
            )
            totals["giou"] = totals["giou"] + tube_weight * _weighted_time_mean(
                1.0 - _aligned_giou(
                    pred_boxes[batch_index, query_index, mask],
                    tube["boxes"][mask],
                ),
                frame_weights,
            )
            if geometry_parent_boxes is not None:
                parent_iou = _aligned_iou(
                    geometry_parent_boxes[
                        batch_index, query_index, mask
                    ],
                    tube["boxes"][mask],
                ).detach()
                corrected_iou = _aligned_iou(
                    pred_boxes[batch_index, query_index, mask],
                    tube["boxes"][mask],
                )
                totals["geometry_preservation"] = (
                    totals["geometry_preservation"]
                    + tube_weight * _weighted_time_mean(
                        (parent_iou - corrected_iou).clamp_min(0),
                        frame_weights,
                    )
                )
            if transport_boxes is not None:
                transport_error = F.smooth_l1_loss(
                    transport_boxes[batch_index, query_index, mask],
                    tube["boxes"][mask],
                    reduction="none",
                    beta=0.05,
                ).mean(dim=-1)
                totals["transport"] = (
                    totals["transport"]
                    + tube_weight * _weighted_time_mean(
                        transport_error, frame_weights
                    )
                )
            if quality_logits is not None:
                spatial_quality = _aligned_iou(
                    pred_boxes[batch_index, query_index, mask], tube["boxes"][mask]
                ).mean()
                if has_intervals:
                    temporal_quality = _soft_interval_iou(
                        interval_support[batch_index, query_index], tube["interval"]
                    )
                else:
                    temporal_quality = _soft_interval_iou(
                        visibility_logits[batch_index, query_index].sigmoid(),
                        tube["visible"],
                    )
                quality_target[query_index] = _tube_quality_target(
                    spatial_quality,
                    temporal_quality,
                    mode=quality_target_mode,
                    strict_blend=quality_strict_blend,
                )
            if has_intervals:
                start_index = tube["start"].argmax().view(1)
                end_index = tube["end"].argmax().view(1)
                totals["start"] = totals["start"] + F.cross_entropy(
                    start_logits[batch_index, query_index].unsqueeze(0), start_index
                ) * tube_weight
                totals["end"] = totals["end"] + F.cross_entropy(
                    end_logits[batch_index, query_index].unsqueeze(0), end_index
                ) * tube_weight
                target_interval = tube["interval"]
                totals["interval_iou"] = totals["interval_iou"] + tube_weight * (
                    1.0 - _soft_interval_iou(
                        interval_support[batch_index, query_index], target_interval
                    )
                )
                totals["coverage"] = (
                    totals["coverage"]
                    + tube_weight * _interval_coverage_error(
                        visibility_logits[batch_index, query_index].sigmoid(),
                        target_interval,
                    )
                )
                totals["fragmentation"] = totals["fragmentation"] + (
                    tube_weight * _interval_fragmentation_error(
                        visibility_logits[batch_index, query_index].sigmoid(),
                        target_interval,
                    )
                )
            if boundary_distances is not None:
                predicted_distance = boundary_distances[
                    batch_index, query_index
                ]
                target_distance = tube["boundary_distance"]
                totals["boundary_distance"] = totals[
                    "boundary_distance"
                ] + tube_weight * F.smooth_l1_loss(
                    predicted_distance, target_distance,
                    reduction="mean", beta=0.05,
                )
                if time > 1:
                    totals["boundary_distance_slope"] = totals[
                        "boundary_distance_slope"
                    ] + tube_weight * F.smooth_l1_loss(
                        predicted_distance[1:] - predicted_distance[:-1],
                        target_distance[1:] - target_distance[:-1],
                        reduction="mean", beta=0.02,
                    )
            pair_mask = mask[1:] & mask[:-1]
            if pair_mask.any():
                pred_velocity = (pred_boxes[batch_index, query_index, 1:]
                                 - pred_boxes[batch_index, query_index, :-1])
                target_velocity = tube["boxes"][1:] - tube["boxes"][:-1]
                velocity_error = F.smooth_l1_loss(
                    pred_velocity[pair_mask], target_velocity[pair_mask],
                    reduction="none", beta=0.02,
                ).mean(dim=-1)
                pair_weights = (
                    tube["temporal_frame_weights"][1:]
                    * tube["temporal_frame_weights"][:-1]
                ).sqrt()[pair_mask]
                totals["velocity"] = (
                    totals["velocity"]
                    + tube_weight * _weighted_time_mean(
                        velocity_error, pair_weights
                    )
                )
            triplet_mask = mask[2:] & mask[1:-1] & mask[:-2]
            if triplet_mask.any():
                pred_acceleration = (
                    pred_boxes[batch_index, query_index, 2:]
                    - 2.0 * pred_boxes[batch_index, query_index, 1:-1]
                    + pred_boxes[batch_index, query_index, :-2]
                )
                target_acceleration = (
                    tube["boxes"][2:] - 2.0 * tube["boxes"][1:-1]
                    + tube["boxes"][:-2]
                )
                acceleration_error = F.smooth_l1_loss(
                    pred_acceleration[triplet_mask],
                    target_acceleration[triplet_mask],
                    reduction="none", beta=0.02,
                ).mean(dim=-1)
                triplet_weights = (
                    tube["temporal_frame_weights"][2:]
                    * tube["temporal_frame_weights"][1:-1]
                    * tube["temporal_frame_weights"][:-2]
                ).pow(1.0 / 3.0)[triplet_mask]
                totals["acceleration"] = (
                    totals["acceleration"]
                    + tube_weight * _weighted_time_mean(
                        acceleration_error, triplet_weights
                    )
                )
        if class_focal_alpha is not None:
            totals["cls"] = totals["cls"] + _sigmoid_focal_loss(
                cls_logits[batch_index], class_target,
                class_focal_alpha, class_focal_gamma,
            ).sum()
        totals["visibility"] = totals["visibility"] + F.binary_cross_entropy_with_logits(
            visibility_logits[batch_index], visibility_target
        )
        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_logits[batch_index], boundary_target,
            reduction="none",
            pos_weight=boundary_logits.new_tensor(float(boundary_pos_weight)),
        )
        if boundary_focal_gamma > 0:
            probability = boundary_logits[batch_index].sigmoid()
            target_probability = torch.where(
                boundary_target > 0, probability, 1.0 - probability
            )
            boundary_bce = boundary_bce * (
                1.0 - target_probability
            ).pow(float(boundary_focal_gamma))
        totals["boundary"] = totals["boundary"] + boundary_bce.mean()
        if quality_logits is not None:
            totals["quality"] = totals["quality"] + F.binary_cross_entropy_with_logits(
                quality_logits[batch_index], quality_target
            )

    normalizer = max(matched_weight, 1.0)
    return {
        "query_cls_loss": totals["cls"] / normalizer,
        "query_box_loss": totals["box"] / normalizer,
        "query_giou_loss": totals["giou"] / normalizer,
        "query_visibility_loss": totals["visibility"] / batch_size,
        "query_boundary_loss": totals["boundary"] / batch_size,
        "query_velocity_loss": totals["velocity"] / normalizer,
        "query_acceleration_loss": totals["acceleration"] / normalizer,
        "query_start_loss": totals["start"] / normalizer,
        "query_end_loss": totals["end"] / normalizer,
        "query_interval_iou_loss": totals["interval_iou"] / normalizer,
        "query_coverage_loss": totals["coverage"] / normalizer,
        "query_fragmentation_loss": totals["fragmentation"] / normalizer,
        "query_boundary_distance_loss": totals["boundary_distance"] / normalizer,
        "query_boundary_distance_slope_loss": (
            totals["boundary_distance_slope"] / normalizer
        ),
        "query_quality_loss": totals["quality"] / batch_size,
        "query_transport_loss": totals["transport"] / normalizer,
        "query_geometry_preservation_loss": (
            totals["geometry_preservation"] / normalizer
        ),
        "query_matches": matched_count,
        "query_target_weight": matched_weight,
    }
