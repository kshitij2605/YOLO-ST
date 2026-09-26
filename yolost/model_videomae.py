"""YOLO-ST variant with a VideoMAE video-pretrained backbone.

VideoMAE returns tubelet tokens instead of frame-level maps. This wrapper
reshapes tokens into a dense (T, H, W) map, then projects them to the existing
YOLO-ST boundary head pyramid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification

from .backbone import YOLOST_Backbone
from .dense_tube_geometry_contract import DenseTubeGeometryContract
from .head_boundary import DecoupledHeadBoundary
from .model_dinov3 import ConvBNAct3D, TemporalAdapterBlock
from .neck_pyramid import TemporalPyramidNeck
from .pyramid_residual_adapter import PyramidAlignedResidualAdapters
from .query_trajectory_residual import TubeQueryTrajectoryResidualAdapter
from .tube_query import (ActorAlignedTubeQueryHead, FactorizedPersonTubeletHead,
                         TubeQueryHead)


def _load_videomae_pretrained(model_class, model_id, dtype):
    """Support both current and legacy Transformers dtype keywords."""
    try:
        return model_class.from_pretrained(model_id, dtype=dtype)
    except TypeError as error:
        if "unexpected keyword argument 'dtype'" not in str(error):
            raise
        return model_class.from_pretrained(model_id, torch_dtype=dtype)


class FactorizedConvBNAct3D(nn.Module):
    """(2+1)D conv: spatial mixing followed by temporal mixing."""

    def __init__(self, channels, kernel_size=3, temporal_kernel=3):
        super().__init__()
        spatial_pad = kernel_size // 2
        temporal_pad = temporal_kernel // 2
        self.spatial = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, kernel_size, kernel_size),
                padding=(0, spatial_pad, spatial_pad),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(temporal_kernel, 1, 1),
                padding=(temporal_pad, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm3d(channels),
            nn.SiLU(inplace=True),
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.temporal(self.spatial(x)))


class LearnedTemporalAntiAlias(nn.Module):
    """Learn temporal low-pass weights with a small factorized 3D scorer."""

    def __init__(self, hidden=8, spatial=False, center_blend=False):
        super().__init__()
        self.spatial_weights = bool(spatial)
        self.spatial = nn.Conv3d(3, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(
            hidden, hidden, kernel_size=(3, 1, 1), padding=(1, 0, 0)
        )
        self.score = nn.Conv3d(hidden, 1, kernel_size=1)
        self.activation = nn.GELU()
        if center_blend:
            self.center_logit = nn.Parameter(torch.tensor(-2.1972246))
        else:
            self.register_parameter("center_logit", None)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x, output_frames):
        _, _, time, height, width = x.shape
        reduced = F.avg_pool3d(x.float(), kernel_size=(1, 4, 4), stride=(1, 4, 4))
        logits = self.score(
            self.activation(self.temporal(self.activation(self.spatial(reduced))))
        )
        if self.spatial_weights:
            logits = F.interpolate(
                logits, size=(time, height, width), mode="trilinear",
                align_corners=False,
            )
        else:
            logits = logits.mean(dim=(3, 4), keepdim=True)
        edges = torch.linspace(0, time, output_frames + 1, device=x.device).round().long()
        sampled = []
        for index in range(output_frames):
            start, end = int(edges[index]), int(edges[index + 1])
            end = max(end, start + 1)
            frames = x[:, :, start:end]
            weights = logits[:, :, start:end].softmax(dim=2).to(dtype=x.dtype)
            pooled = (frames * weights).sum(dim=2)
            if self.center_logit is not None:
                center = frames[:, :, (end - start - 1) // 2]
                center_weight = self.center_logit.sigmoid().to(dtype=x.dtype)
                pooled = (1.0 - center_weight) * pooled + center_weight * center
            sampled.append(pooled)
        return torch.stack(sampled, dim=2)


class APTPyramidAdapter(nn.Module):
    """Lightweight factorized adapter for foundation-token pyramid features."""

    def __init__(self, channels, depth=1, kernel_size=3, temporal_kernel=3, dropout=0.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                FactorizedConvBNAct3D(
                    channels,
                    kernel_size=kernel_size,
                    temporal_kernel=temporal_kernel,
                ),
                nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class SpatialQueryAdapter(nn.Module):
    """Reconstruct time-indexed maps by querying the complete VideoMAE token set."""

    def __init__(self, hidden, out_channels=256, spatial_size=7, num_heads=8,
                 dropout=0.1, temporal_steps=1):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError("SQA hidden size must be divisible by num_heads")
        self.spatial_size = int(spatial_size)
        self.spatial_queries = nn.Parameter(torch.randn(
            1, self.spatial_size * self.spatial_size, hidden
        ))
        self.temporal_queries = nn.Parameter(torch.randn(
            1, int(temporal_steps), 1, hidden
        ))
        self.query_norm = nn.LayerNorm(hidden)
        self.token_norm = nn.LayerNorm(hidden)
        self.refine_norm = nn.LayerNorm(hidden)
        self.cross_attention = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self._reset_parameters()
        nn.init.normal_(self.spatial_queries, std=0.02)
        nn.init.normal_(self.temporal_queries, std=0.02)

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        batch = tokens.shape[0]
        temporal_steps = self.temporal_queries.shape[1]
        spatial_tokens = self.spatial_size * self.spatial_size
        queries = (
            self.spatial_queries.unsqueeze(1) + self.temporal_queries
        ).expand(batch, -1, -1, -1).reshape(
            batch, temporal_steps * spatial_tokens, -1
        )
        normalized_tokens = self.token_norm(tokens)
        update, _ = self.cross_attention(
            self.query_norm(queries), normalized_tokens, normalized_tokens,
            need_weights=False,
        )
        queries = queries + update
        queries = queries.reshape(batch * temporal_steps, spatial_tokens, -1)
        normalized_queries = self.refine_norm(queries)
        update, _ = self.self_attention(
            normalized_queries, normalized_queries, normalized_queries,
            need_weights=False,
        )
        queries = queries + update
        spatial = self.projection(queries).transpose(1, 2).reshape(
            batch * temporal_steps, -1, self.spatial_size, self.spatial_size
        )
        spatial = spatial + self.spatial_refine(spatial)
        return spatial.reshape(
            batch, temporal_steps, -1, self.spatial_size, self.spatial_size
        ).permute(0, 2, 1, 3, 4).contiguous()


class PyramidContextGate(nn.Module):
    """Global temporal context gate over a pyramid level.

    This is the first APT context hook. It is intentionally lightweight so it
    can be ablated against the stronger actor-conditioned attention module.
    """

    def __init__(self, channels, hidden=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden, channels * 2, kernel_size=1),
        )

    def forward(self, x):
        b, c, t, _, _ = x.shape
        ctx = x.mean(dim=(3, 4))
        gate_bias = self.net(ctx).view(b, 2, c, t, 1, 1)
        gate = torch.sigmoid(gate_bias[:, 0])
        bias = gate_bias[:, 1]
        return x * (1.0 + gate) + bias


class ActorContextGate(nn.Module):
    """Actor-token temporal context gate for dense action features.

    A learned saliency map extracts one actor-like token per frame. Temporal
    self-attention then lets each frame condition its dense map on surrounding
    actor context without introducing a separate query decoder.
    """

    def __init__(self, channels, hidden=128, num_heads=4, depth=1, dropout=0.0):
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1)
        self.proj_in = nn.Linear(channels, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj_out = nn.Linear(hidden, channels * 2)

    def forward(self, x):
        b, c, t, h, w = x.shape
        score = self.score(x).flatten(3).softmax(dim=-1)
        feat = x.flatten(3).permute(0, 2, 3, 1)
        actor = (feat * score.permute(0, 2, 3, 1)).sum(dim=2)
        ctx = self.temporal(self.proj_in(actor))
        gate_bias = self.proj_out(ctx).view(b, t, 2, c).permute(0, 2, 3, 1)
        gate = torch.sigmoid(gate_bias[:, 0]).unsqueeze(-1).unsqueeze(-1)
        bias = gate_bias[:, 1].unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gate) + bias


class MultiActorContextGate(nn.Module):
    """Route multiple temporally coherent actor slots back to dense features."""

    def __init__(self, channels, hidden=128, num_heads=4, depth=1, slots=2, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.slots = slots
        self.key = nn.Conv3d(channels, hidden, kernel_size=1)
        self.query = nn.Conv3d(channels, hidden, kernel_size=1)
        self.value = nn.Conv3d(channels, hidden, kernel_size=1)
        self.slot_queries = nn.Parameter(torch.empty(slots, hidden))
        nn.init.normal_(self.slot_queries, std=hidden ** -0.5)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=depth)
        self.slot_norm = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, channels * 2)

    def forward(self, x):
        b, _, t, h, w = x.shape
        key = self.key(x).permute(0, 2, 3, 4, 1).reshape(b, t, h * w, self.hidden)
        value = self.value(x).permute(0, 2, 3, 4, 1).reshape(b, t, h * w, self.hidden)

        score = torch.einsum("btnc,kc->btnk", key, self.slot_queries)
        score = score.permute(0, 1, 3, 2).softmax(dim=-1)
        slots = torch.einsum("btkn,btnc->btkc", score, value)
        slots = slots + self.slot_queries.view(1, 1, self.slots, self.hidden)

        slots = slots.permute(0, 2, 1, 3).reshape(b * self.slots, t, self.hidden)
        slots = self.temporal(slots)
        slots = self.slot_norm(slots)
        slots = slots.reshape(b, self.slots, t, self.hidden).permute(0, 2, 1, 3)

        dense_query = self.query(x).permute(0, 2, 3, 4, 1).reshape(b, t, h * w, self.hidden)
        route = torch.einsum("btnc,btkc->btnk", dense_query, slots)
        route = (route * (self.hidden ** -0.5)).softmax(dim=-1)
        context = torch.einsum("btnk,btkc->btnc", route, slots)
        gate_bias = self.proj_out(context).view(b, t, h, w, 2, -1)
        gate = torch.sigmoid(gate_bias[..., 0, :]).permute(0, 4, 1, 2, 3)
        bias = gate_bias[..., 1, :].permute(0, 4, 1, 2, 3)
        return x * (1.0 + gate) + bias


class ClassSpecificContextRefiner(nn.Module):
    """Refine dense class logits with actor-positioned class queries."""

    def __init__(self, channels, num_classes, hidden=128, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.num_classes = num_classes
        self.class_queries = nn.Parameter(torch.empty(num_classes, hidden))
        nn.init.normal_(self.class_queries, std=hidden ** -0.5)
        self.actor_proj = nn.Linear(channels, hidden)
        self.position_proj = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.key = nn.Conv3d(channels, hidden, kernel_size=1)
        self.value = nn.Conv3d(channels, hidden, kernel_size=1)
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.score_scale = nn.Parameter(torch.zeros(()))
        self.spatial_scale = nn.Parameter(torch.zeros(()))

    def forward(self, feature, cls_logits, obj_logits):
        b, channels, t, h, w = feature.shape
        n = h * w
        dense = feature.permute(0, 2, 3, 4, 1).reshape(b, t, n, channels)

        actor_weight = obj_logits[:, 0].detach().flatten(2).softmax(dim=-1)
        actor = torch.einsum("btn,btnc->btc", actor_weight, dense)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feature.device, dtype=feature.dtype),
            torch.linspace(-1.0, 1.0, w, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        coordinates = torch.stack([xx, yy], dim=-1).reshape(n, 2)
        actor_xy = torch.einsum("btn,nc->btc", actor_weight, coordinates)

        query = self.actor_proj(actor) + self.position_proj(actor_xy)
        query = query.unsqueeze(2) + self.class_queries.view(1, 1, self.num_classes, self.hidden)
        key = self.key(feature).permute(0, 2, 3, 4, 1).reshape(b, t, n, self.hidden)
        value = self.value(feature).permute(0, 2, 3, 4, 1).reshape(b, t, n, self.hidden)

        attention_logits = torch.einsum("btch,btnh->btcn", query, key) * (self.hidden ** -0.5)
        attention = attention_logits.softmax(dim=-1)
        context = torch.einsum("btcn,btnh->btch", attention, value)
        score_delta = self.score(context + query).squeeze(-1).permute(0, 2, 1)
        score_delta = score_delta.unsqueeze(-1).unsqueeze(-1)

        spatial_prior = (attention.clamp_min(1e-7) * n).log()
        spatial_prior = spatial_prior.reshape(b, t, self.num_classes, h, w).permute(0, 2, 1, 3, 4)
        return (
            cls_logits
            + self.score_scale.tanh() * score_delta
            + self.spatial_scale.tanh() * spatial_prior
        )


class NativeMotionFusion(nn.Module):
    """Fuse a native-rate (2+1)D pyramid into foundation-token features.

    The residual scale starts at zero, preserving the resumed VideoMAE model at
    initialization while still allowing the fusion path to learn immediately.
    """

    def __init__(self, channels):
        super().__init__()
        self.motion_proj = FactorizedConvBNAct3D(channels)
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, semantic, motion):
        if motion.shape[2:] != semantic.shape[2:]:
            motion = F.interpolate(
                motion,
                size=semantic.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        motion = self.motion_proj(motion)
        gate = torch.sigmoid(self.gate(torch.cat([semantic, motion], dim=1)))
        return semantic + self.scale.tanh() * gate * motion


class KeyframePyramidFusion(nn.Module):
    """Inject pretrained per-frame spatial features through a preserved residual."""

    def __init__(self, semantic_channels, keyframe_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv3d(keyframe_channels, semantic_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(semantic_channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Conv3d(semantic_channels * 2, semantic_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, semantic, keyframe):
        if keyframe.shape[2:] != semantic.shape[2:]:
            keyframe = F.interpolate(
                keyframe,
                size=semantic.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        keyframe = self.projection(keyframe)
        gate = torch.sigmoid(self.gate(torch.cat([semantic, keyframe], dim=1)))
        return semantic + self.scale.tanh() * gate * keyframe


class ActorLocalSignedMotionFusion(nn.Module):
    """Fuse signed native-rate motion only around provisional actor evidence."""

    def __init__(self, channels, actor_floor=0.1, channel_ratio=1.0):
        super().__init__()
        self.actor_floor = float(actor_floor)
        if not 0.0 < float(channel_ratio) <= 1.0:
            raise ValueError("native motion channel_ratio must be in (0, 1]")
        self.channels = int(channels)
        self.active_channels = max(1, int(round(channels * float(channel_ratio))))
        if self.active_channels == self.channels:
            self.motion_proj = FactorizedConvBNAct3D(channels)
            self.delta_proj = FactorizedConvBNAct3D(channels)
        else:
            self.motion_proj = nn.Sequential(
                nn.Conv3d(channels, self.active_channels, kernel_size=1, bias=False),
                FactorizedConvBNAct3D(self.active_channels),
            )
            self.delta_proj = nn.Sequential(
                nn.Conv3d(channels, self.active_channels, kernel_size=1, bias=False),
                FactorizedConvBNAct3D(self.active_channels),
            )
        self.gate = nn.Conv3d(
            self.active_channels * 3 + 1, self.active_channels, kernel_size=1
        )
        self.delta_scale = nn.Parameter(torch.zeros(()))
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, semantic, motion, actor_mask):
        if motion.shape[2:] != semantic.shape[2:]:
            motion = F.interpolate(
                motion,
                size=semantic.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if actor_mask.shape[2:] != semantic.shape[2:]:
            actor_mask = F.interpolate(
                actor_mask,
                size=semantic.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if motion.shape[2] > 1:
            difference = motion[:, :, 1:] - motion[:, :, :-1]
            signed_delta = torch.cat([difference[:, :, :1], difference], dim=2)
        else:
            signed_delta = torch.zeros_like(motion)

        semantic_active = semantic[:, -self.active_channels:]
        motion_feature = self.motion_proj(motion)
        delta_feature = self.delta_proj(signed_delta)
        actor_mask = actor_mask.detach().clamp(0.0, 1.0)
        actor_weight = self.actor_floor + (1.0 - self.actor_floor) * actor_mask
        gate = torch.sigmoid(self.gate(torch.cat(
            [semantic_active, motion_feature, delta_feature, actor_mask], dim=1
        )))
        motion_residual = (
            motion_feature + self.delta_scale.tanh() * delta_feature
        )
        updated = semantic_active + (
            self.scale.tanh() * actor_weight * gate * motion_residual
        )
        if self.active_channels == self.channels:
            return updated
        return torch.cat(
            [semantic[:, :-self.active_channels], updated], dim=1
        )


class DilatedBidirectionalStateAdapter(nn.Module):
    """Selected-level long temporal context with protected spatial residuals."""

    def __init__(self, channels, dilations=(1, 2, 4, 8), dropout=0.0):
        super().__init__()
        self.temporal_filters = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size=3, padding=int(dilation),
                dilation=int(dilation), groups=channels, bias=False,
            )
            for dilation in dilations
        ])
        self.mix = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, feature):
        state = feature.mean(dim=(-1, -2))
        contexts = [temporal_filter(state) for temporal_filter in self.temporal_filters]
        context = self.mix(torch.stack(contexts, dim=0).mean(dim=0))
        context = context.unsqueeze(-1).unsqueeze(-1).expand_as(feature)
        gate = torch.sigmoid(self.gate(torch.cat([feature, context], dim=1)))
        return feature + self.scale.tanh() * gate * context


class TubeTaskAdapter(nn.Module):
    """A tube-only residual path that cannot perturb dense frame features."""

    def __init__(self, channels, bottleneck_ratio=0.25, temporal_kernel=3):
        super().__init__()
        if not 0.0 < float(bottleneck_ratio) <= 1.0:
            raise ValueError("tube task adapter ratio must be in (0, 1]")
        temporal_kernel = int(temporal_kernel)
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("tube task adapter temporal kernel must be odd")
        hidden = max(16, int(round(channels * float(bottleneck_ratio))))
        self.input = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv3d(
                hidden, hidden, kernel_size=(temporal_kernel, 3, 3),
                padding=(temporal_kernel // 2, 1, 1), groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv3d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature):
        return self.output(self.context(self.input(feature)))


class LocalTrajectoryAligner(nn.Module):
    """Align adjacent-frame context with a local feature cost volume."""

    def __init__(self, channels, hidden=32, radius=2, temperature=0.07):
        super().__init__()
        if radius < 1:
            raise ValueError("trajectory radius must be at least 1")
        self.radius = int(radius)
        self.temperature = float(temperature)
        self.query = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.key = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.value = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Conv3d(channels * 2, channels, kernel_size=1)
        self.scale = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _shift_spatial(x, dy, dx, radius):
        height, width = x.shape[-2:]
        padded = F.pad(x, (radius, radius, radius, radius, 0, 0))
        y0 = radius + dy
        x0 = radius + dx
        return padded[..., y0:y0 + height, x0:x0 + width]

    def forward(self, x):
        if x.shape[2] < 2:
            return x

        query = F.normalize(self.query(x)[:, :, :-1], dim=1)
        key_next = F.normalize(self.key(x)[:, :, 1:], dim=1)
        value_next = self.value(x)[:, :, 1:]
        radius = self.radius

        scores = []
        shifted_values = []
        valid_template = torch.ones_like(key_next[:, :1])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shifted_key = self._shift_spatial(key_next, dy, dx, radius)
                score = (query * shifted_key).sum(dim=1) / self.temperature
                valid = self._shift_spatial(valid_template, dy, dx, radius)[:, 0] > 0
                scores.append(score.masked_fill(~valid, -1e4))
                shifted_values.append(self._shift_spatial(value_next, dy, dx, radius))

        weights = torch.stack(scores, dim=2).softmax(dim=2)
        aligned_next = torch.zeros_like(value_next)
        for index, shifted_value in enumerate(shifted_values):
            aligned_next = aligned_next + weights[:, :, index].unsqueeze(1) * shifted_value

        aligned = torch.cat([aligned_next, self.value(x[:, :, -1:])], dim=2)
        gate = torch.sigmoid(self.gate(torch.cat([x, aligned], dim=1)))
        return x + self.scale.tanh() * gate * (aligned - x)


class NoisyTubeDenoiser(nn.Module):
    """Reconstruct coherent GT tubelets from spatial and label perturbations."""

    def __init__(self, channels, num_classes, hidden=128, num_heads=4, depth=2,
                 max_tubes=8, box_noise=0.1, label_noise=0.2, max_track_gap=3):
        super().__init__()
        self.num_classes = num_classes
        self.max_tubes = max_tubes
        self.box_noise = box_noise
        self.label_noise = label_noise
        self.max_track_gap = max_track_gap
        self.feature_proj = nn.Linear(channels, hidden)
        self.box_embed = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(inplace=True))
        self.label_embed = nn.Embedding(num_classes, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=depth)
        self.box_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4)
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, num_classes),
        )
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    @staticmethod
    def _box_iou(box, boxes):
        left_top = torch.maximum(box[:2], boxes[:, :2])
        right_bottom = torch.minimum(box[2:], boxes[:, 2:])
        intersection = (right_bottom - left_top).clamp_min(0).prod(dim=-1)
        box_area = (box[2:] - box[:2]).clamp_min(0).prod()
        boxes_area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=-1)
        return intersection / (box_area + boxes_area - intersection).clamp_min(1e-6)

    def _link_tubes(self, boxes, labels):
        if labels.ndim > 1:
            labels = labels.argmax(dim=-1)
        valid = boxes[:, 1:5].sum(dim=-1) > 0
        boxes = boxes[valid]
        labels = labels[valid]
        if boxes.numel() == 0:
            return []

        order = boxes[:, 0].argsort()
        boxes = boxes[order]
        labels = labels[order]
        tracks = []
        for frame in boxes[:, 0].long().unique(sorted=True):
            frame_mask = boxes[:, 0].long() == frame
            frame_boxes = boxes[frame_mask, 1:5]
            frame_labels = labels[frame_mask]
            used_tracks = set()
            for box, label in zip(frame_boxes, frame_labels):
                label_value = int(label.item())
                candidates = [
                    index for index, track in enumerate(tracks)
                    if track["label"] == label_value
                    and index not in used_tracks
                    and int(frame.item()) - track["last_frame"] <= self.max_track_gap
                ]
                best_track = None
                if candidates:
                    last_boxes = torch.stack([tracks[index]["last_box"] for index in candidates])
                    ious = self._box_iou(box, last_boxes)
                    best = int(ious.argmax().item())
                    if float(ious[best].item()) >= 0.1:
                        best_track = candidates[best]
                if best_track is None:
                    tracks.append({
                        "label": label_value,
                        "last_frame": int(frame.item()),
                        "last_box": box,
                        "frames": [int(frame.item())],
                        "boxes": [box],
                    })
                    used_tracks.add(len(tracks) - 1)
                else:
                    track = tracks[best_track]
                    track["last_frame"] = int(frame.item())
                    track["last_box"] = box
                    track["frames"].append(int(frame.item()))
                    track["boxes"].append(box)
                    used_tracks.add(best_track)
        tracks.sort(key=lambda track: len(track["frames"]), reverse=True)
        return tracks[:self.max_tubes]

    def _perturb_boxes(self, boxes, mask):
        centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-3)
        valid_sizes = sizes[mask]
        reference_size = valid_sizes.mean(dim=0) if valid_sizes.numel() else sizes.new_ones(2)
        shared_shift = torch.randn(2, device=boxes.device) * self.box_noise * reference_size
        shared_scale = torch.exp(torch.randn(2, device=boxes.device) * self.box_noise)
        frame_shift = torch.randn_like(centers) * (0.25 * self.box_noise) * sizes
        centers = centers + shared_shift + frame_shift
        sizes = sizes * shared_scale
        noisy = torch.cat([centers - sizes * 0.5, centers + sizes * 0.5], dim=-1)
        return noisy.clamp(0.0, 1.0)

    def forward(self, feature, targets, clip_length):
        batch_size, _, feature_time, _, _ = feature.shape
        all_noisy = []
        all_target = []
        all_mask = []
        all_labels = []
        all_features = []

        for batch_index in range(batch_size):
            tracks = self._link_tubes(targets["boxes"][batch_index], targets["labels"][batch_index])
            frame_features = feature[batch_index].permute(1, 0, 2, 3)
            for track in tracks:
                target_boxes = feature.new_zeros(feature_time, 4)
                mask = torch.zeros(feature_time, dtype=torch.bool, device=feature.device)
                frame_indices = torch.tensor(track["frames"], device=feature.device)
                feature_indices = torch.round(
                    frame_indices.float() * (feature_time - 1) / max(clip_length - 1, 1)
                ).long().clamp(0, feature_time - 1)
                linked_boxes = torch.stack(track["boxes"]).to(
                    device=feature.device, dtype=feature.dtype
                )
                target_boxes[feature_indices] = linked_boxes
                mask[feature_indices] = True
                if not mask.any():
                    continue
                noisy_boxes = self._perturb_boxes(target_boxes, mask)
                centers = (noisy_boxes[:, :2] + noisy_boxes[:, 2:]) * 0.5
                grid = (centers * 2.0 - 1.0).view(feature_time, 1, 1, 2)
                sampled = F.grid_sample(
                    frame_features, grid, mode="bilinear", padding_mode="border",
                    align_corners=False,
                )[:, :, 0, 0]
                all_features.append(sampled)
                all_noisy.append(noisy_boxes)
                all_target.append(target_boxes)
                all_mask.append(mask)
                all_labels.append(track["label"])

        if not all_features:
            return None

        sampled_features = torch.stack(all_features)
        noisy_boxes = torch.stack(all_noisy)
        target_boxes = torch.stack(all_target)
        tube_mask = torch.stack(all_mask)
        target_labels = torch.tensor(all_labels, dtype=torch.long, device=feature.device)
        noisy_labels = target_labels.clone()
        corrupt = torch.rand_like(noisy_labels.float()) < self.label_noise
        random_labels = torch.randint(self.num_classes, noisy_labels.shape, device=feature.device)
        noisy_labels = torch.where(corrupt, random_labels, noisy_labels)

        tokens = (
            self.feature_proj(sampled_features)
            + self.box_embed(noisy_boxes)
            + self.label_embed(noisy_labels).unsqueeze(1)
        )
        tokens = self.temporal(tokens, src_key_padding_mask=~tube_mask)
        box_delta = 0.25 * self.box_head(tokens).tanh()
        predicted_boxes = (noisy_boxes + box_delta).clamp(0.0, 1.0)
        pooled = (tokens * tube_mask.unsqueeze(-1)).sum(dim=1) / tube_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return {
            "pred_boxes": predicted_boxes,
            "target_boxes": target_boxes,
            "mask": tube_mask,
            "pred_logits": self.class_head(pooled),
            "target_labels": target_labels,
        }


class CrossClipActorMemory(nn.Module):
    """Carry compact actor state across temporal chunks and evaluation clips."""

    def __init__(self, channels, hidden=128, chunk_size=16, bidirectional=True,
                 dropout=0.0):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("memory chunk size must be positive")
        self.hidden = hidden
        self.chunk_size = int(chunk_size)
        self.bidirectional = bidirectional
        self.score = nn.Conv3d(channels, 1, kernel_size=1)
        self.actor_proj = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.update = nn.GRUCell(hidden, hidden)
        self.initial_state = nn.Parameter(torch.zeros(1, hidden))
        self.output = nn.Linear(hidden, channels * 2)
        self.scale = nn.Parameter(torch.zeros(()))
        self.persistent_eval = False
        self._persistent_state = None

    def reset_memory(self):
        self._persistent_state = None

    def enable_persistent_eval(self, enabled=True):
        self.persistent_eval = bool(enabled)
        if not enabled:
            self.reset_memory()

    def _run_chunks(self, actor, initial_state, reverse=False):
        batch_size, time, _ = actor.shape
        boundaries = list(range(0, time, self.chunk_size))
        order = list(reversed(boundaries)) if reverse else boundaries
        state = initial_state
        context = actor.new_zeros(batch_size, time, self.hidden)
        chunk_states = {}
        for start in order:
            end = min(start + self.chunk_size, time)
            token = actor[:, start:end].mean(dim=1)
            state = self.update(token, state)
            context[:, start:end] = state.unsqueeze(1)
            chunk_states[start] = state
        ordered_states = torch.stack([chunk_states[start] for start in boundaries], dim=1)
        return context, state, ordered_states

    def forward(self, feature, return_context=False):
        batch_size, channels, time, _, _ = feature.shape
        weights = self.score(feature).flatten(3).softmax(dim=-1)
        dense = feature.flatten(3).permute(0, 2, 3, 1)
        actor = torch.einsum("btn,btnc->btc", weights[:, 0], dense)
        actor = self.actor_proj(actor)

        if (not self.training and self.persistent_eval and
                self._persistent_state is not None and
                self._persistent_state.shape[0] == batch_size):
            initial = self._persistent_state.to(device=feature.device, dtype=actor.dtype)
        else:
            initial = self.initial_state.expand(batch_size, -1).to(dtype=actor.dtype)

        forward_context, final_state, forward_states = self._run_chunks(actor, initial)
        consistency = feature.new_zeros(())
        context = forward_context
        if self.training and self.bidirectional:
            backward_initial = self.initial_state.expand(batch_size, -1).to(dtype=actor.dtype)
            backward_context, _, backward_states = self._run_chunks(
                actor, backward_initial, reverse=True
            )
            context = 0.5 * (forward_context + backward_context)
            consistency = (
                F.normalize(forward_states, dim=-1)
                - F.normalize(backward_states, dim=-1)
            ).pow(2).mean()

        if not self.training and self.persistent_eval:
            self._persistent_state = final_state.detach()

        gate_bias = self.output(context).view(batch_size, time, 2, channels)
        gate = torch.sigmoid(gate_bias[:, :, 0]).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        bias = gate_bias[:, :, 1].permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        refined = feature + self.scale.tanh() * (gate * feature + bias)
        if return_context:
            return refined, consistency, context
        return refined, consistency


class PyramidActorMemory(nn.Module):
    """Route one persistent actor state through a multi-rate feature pyramid."""

    def __init__(self, channels, hidden=128, chunk_size=8, bidirectional=True,
                 dropout=0.0, learned_scale_routing=True):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("memory chunk size must be positive")
        if isinstance(channels, int):
            channels = (channels, channels, channels)
        if len(channels) != 3:
            raise ValueError("pyramid actor memory requires three channel widths")
        channels = tuple(int(width) for width in channels)
        self.hidden = int(hidden)
        self.chunk_size = int(chunk_size)
        self.bidirectional = bool(bidirectional)
        self.learned_scale_routing = bool(learned_scale_routing)
        self.spatial_scores = nn.ModuleList([
            nn.Conv3d(width, 1, kernel_size=1) for width in channels
        ])
        self.actor_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            for width in channels
        ])
        self.route_scores = (
            nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
            if self.learned_scale_routing else None
        )
        self.update = nn.GRUCell(hidden, hidden)
        self.initial_state = nn.Parameter(torch.zeros(1, hidden))
        self.outputs = nn.ModuleList([
            nn.Linear(hidden, width * 2) for width in channels
        ])
        self.scales = nn.Parameter(torch.zeros(3))
        self.persistent_eval = False
        self._persistent_state = None

    def reset_memory(self):
        self._persistent_state = None

    def enable_persistent_eval(self, enabled=True):
        self.persistent_eval = bool(enabled)
        if not enabled:
            self.reset_memory()

    @staticmethod
    def _resize_tokens(tokens, time):
        if tokens.shape[1] == time:
            return tokens
        return F.interpolate(
            tokens.transpose(1, 2),
            size=time,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _pool_actor(self, feature, score, projection):
        weights = score(feature).flatten(3).softmax(dim=-1)
        dense = feature.flatten(3).permute(0, 2, 3, 1)
        actor = torch.einsum("btn,btnc->btc", weights[:, 0], dense)
        return projection(actor)

    def _route_actor_tokens(self, features):
        native_tokens = [
            self._pool_actor(feature, score, projection)
            for feature, score, projection in zip(
                features, self.spatial_scores, self.actor_projections
            )
        ]
        base_time = features[0].shape[2]
        aligned_tokens = [
            self._resize_tokens(tokens, base_time) for tokens in native_tokens
        ]
        if self.route_scores is None:
            routed = torch.stack(aligned_tokens, dim=0).mean(dim=0)
        else:
            route_logits = torch.cat([
                route_score(tokens)
                for route_score, tokens in zip(
                    self.route_scores, aligned_tokens
                )
            ], dim=-1)
            route_weights = route_logits.softmax(dim=-1)
            routed = sum(
                route_weights[..., index:index + 1] * tokens
                for index, tokens in enumerate(aligned_tokens)
            )
        return routed

    def _run_chunks(self, actor, initial_state, reverse=False):
        batch_size, time, _ = actor.shape
        boundaries = list(range(0, time, self.chunk_size))
        order = list(reversed(boundaries)) if reverse else boundaries
        state = initial_state
        context = actor.new_zeros(batch_size, time, self.hidden)
        chunk_states = {}
        for start in order:
            end = min(start + self.chunk_size, time)
            token = actor[:, start:end].mean(dim=1)
            state = self.update(token, state)
            context[:, start:end] = state.unsqueeze(1)
            chunk_states[start] = state
        ordered_states = torch.stack(
            [chunk_states[start] for start in boundaries], dim=1
        )
        return context, state, ordered_states

    def forward(self, features):
        if len(features) != 3:
            raise ValueError("pyramid actor memory requires P3/P4/P5 features")
        batch_size = features[0].shape[0]
        actor = self._route_actor_tokens(features)
        if (not self.training and self.persistent_eval and
                self._persistent_state is not None and
                self._persistent_state.shape[0] == batch_size):
            initial = self._persistent_state.to(
                device=actor.device, dtype=actor.dtype
            )
        else:
            initial = self.initial_state.expand(batch_size, -1).to(
                dtype=actor.dtype
            )

        context, final_state, forward_states = self._run_chunks(actor, initial)
        consistency = actor.new_zeros(())
        if self.training and self.bidirectional:
            backward_initial = self.initial_state.expand(
                batch_size, -1
            ).to(dtype=actor.dtype)
            backward_context, _, backward_states = self._run_chunks(
                actor, backward_initial, reverse=True
            )
            context = 0.5 * (context + backward_context)
            consistency = (
                F.normalize(forward_states, dim=-1)
                - F.normalize(backward_states, dim=-1)
            ).pow(2).mean()

        if not self.training and self.persistent_eval:
            self._persistent_state = final_state.detach()

        refined = []
        for index, (feature, output) in enumerate(
                zip(features, self.outputs)):
            level_context = self._resize_tokens(context, feature.shape[2])
            gate_bias = output(level_context).view(
                batch_size, feature.shape[2], 2, feature.shape[1]
            )
            gate = torch.sigmoid(gate_bias[:, :, 0]).permute(
                0, 2, 1
            ).unsqueeze(-1).unsqueeze(-1)
            bias = gate_bias[:, :, 1].permute(
                0, 2, 1
            ).unsqueeze(-1).unsqueeze(-1)
            refined.append(
                feature
                + self.scales[index].tanh() * (gate * feature + bias)
            )
        return refined, consistency


def _regenerate_sinusoid_table(num_positions, hidden_size):
    """Evaluate VideoMAE's analytic sinusoid table at a new token count.

    VideoMAE stores a fixed 1D sinusoid encoding over the flattened token
    index, so changing the clip length only changes how many rows are needed.
    Evaluating the closed form is exact, where interpolating between rows
    averages out of phase and shrinks every token's norm by about 11%.
    """
    try:
        from transformers.models.videomae.modeling_videomae import (
            get_sinusoid_encoding_table,
        )
        return get_sinusoid_encoding_table(num_positions, hidden_size)
    except Exception:
        # Same closed form, kept so a transformers refactor cannot break this.
        position = torch.arange(num_positions, dtype=torch.float64)
        channel = torch.arange(hidden_size, dtype=torch.float64)
        angle = position.unsqueeze(1) / torch.pow(
            10000.0, 2 * torch.div(channel, 2, rounding_mode="floor")
            / hidden_size
        )
        table = torch.zeros(num_positions, hidden_size, dtype=torch.float64)
        table[:, 0::2] = torch.sin(angle[:, 0::2])
        table[:, 1::2] = torch.cos(angle[:, 1::2])
        return table.float().unsqueeze(0)


class YOLOST_VideoMAE(nn.Module):
    """VideoMAE dense tubelet features + YOLO-ST temporal detection head."""

    def __init__(
        self,
        num_classes=24,
        img_size=224,
        clip_length=64,
        model_id="MCG-NJU/videomae-base-finetuned-kinetics",
        position_embedding_mode="interpolate",
        reg_max=0,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        backbone_gradient_checkpointing=False,
        backbone_frames=16,
        backbone_frame_sampling="uniform",
        backbone_sampling_temperature=0.5,
        backbone_sampling_blend=0.5,
        backbone_sampling_hidden=8,
        dtype="float16",
        temporal_adapter=True,
        temporal_adapter_depth=2,
        temporal_adapter_kernel=5,
        temporal_adapter_expansion=2,
        temporal_adapter_dropout=0.0,
        spatial_query_adapter=False,
        spatial_query_grid=7,
        spatial_query_heads=8,
        spatial_query_dropout=0.1,
        keyframe_pyramid=False,
        keyframe_version="n",
        keyframe_pretrained_path=None,
        keyframe_freeze=True,
        keyframe_bn_eval=True,
        keyframe_grad_checkpoint=False,
        class_ctx_grad_checkpoint=False,
        keyframe_frames=16,
        keyframe_micro_batch=16,
        keyframe_scales=(True, True, True),
        apt_pyramid_adapter=False,
        apt_adapter_depth=1,
        apt_adapter_kernel=3,
        apt_adapter_temporal_kernel=3,
        apt_adapter_dropout=0.0,
        apt_dilated_state=False,
        apt_dilated_state_scales=(False, True, True),
        apt_dilated_state_dilations=(1, 2, 4, 8),
        apt_dilated_state_dropout=0.0,
        apt_context_gate=False,
        apt_context_dim=128,
        apt_context_dropout=0.0,
        apt_actor_context=False,
        apt_actor_context_dim=128,
        apt_actor_context_heads=4,
        apt_actor_context_depth=1,
        apt_actor_context_slots=1,
        apt_actor_context_dropout=0.0,
        apt_class_context=False,
        apt_class_context_dim=128,
        apt_class_context_dropout=0.0,
        apt_trajectory_align=False,
        apt_trajectory_hidden=32,
        apt_trajectory_radius=2,
        apt_trajectory_temperature=0.07,
        apt_trajectory_scales=(True, False, False),
        apt_tube_denoising=False,
        apt_tube_denoising_dim=128,
        apt_tube_denoising_heads=4,
        apt_tube_denoising_depth=2,
        apt_tube_denoising_max_tubes=8,
        apt_tube_denoising_box_noise=0.1,
        apt_tube_denoising_label_noise=0.2,
        apt_cross_clip_memory=False,
        apt_cross_clip_memory_dim=128,
        apt_cross_clip_memory_chunk=16,
        apt_cross_clip_memory_bidirectional=True,
        apt_cross_clip_memory_dropout=0.0,
        apt_cross_clip_memory_target="features",
        apt_cross_clip_decision_target="class_boundary",
        apt_pyramid_actor_memory=False,
        apt_pyramid_actor_memory_dim=128,
        apt_pyramid_actor_memory_chunk=8,
        apt_pyramid_actor_memory_bidirectional=True,
        apt_pyramid_actor_memory_dropout=0.0,
        apt_pyramid_actor_memory_learned_routing=True,
        apt_tube_queries=False,
        apt_tube_query_dim=256,
        apt_tube_query_count=8,
        apt_tube_query_heads=8,
        apt_tube_query_depth=2,
        apt_tube_query_dropout=0.1,
        apt_tube_query_actor_aligned=False,
        apt_tube_query_factorized=False,
        apt_tube_query_frames=32,
        apt_tube_query_memory_grid=14,
        apt_tube_query_boundary_gate=False,
        apt_tube_query_iterative_refinement=False,
        apt_tube_query_trajectory_sampling=False,
        apt_tube_query_trajectory_points=5,
        apt_tube_query_intervals=False,
        apt_tube_query_instances_per_actor=1,
        apt_tube_query_interval_pyramid=False,
        apt_tube_query_drop_path_rate=0.0,
        apt_tube_query_class_prior_probability=None,
        apt_tube_query_quality=False,
        apt_tube_query_boundary_distance=False,
        apt_tube_query_boundary_distance_temperature=0.08,
        apt_tube_query_duration_router=False,
        apt_tube_query_duration_kernels=(3, 7, 15),
        apt_tube_query_change_point_pyramid=False,
        apt_tube_query_change_point_dilations=(1, 2, 4, 8),
        apt_tube_query_change_point_router_mode="actor_duration",
        apt_tube_query_change_point_shared_projection=False,
        apt_tube_query_identity_transport=False,
        apt_tube_query_transport_proposals=16,
        apt_tube_query_transport_sinkhorn_iterations=4,
        apt_tube_query_transport_temperature=0.2,
        apt_tube_query_action_reset_state=False,
        apt_tube_query_action_fork_state=False,
        apt_tube_query_action_fork_mode="state_boundary",
        apt_tube_query_action_fork_temperature=0.5,
        apt_tube_query_shared_grad_scale=1.0,
        apt_tube_query_shared_grad_scales=None,
        apt_tube_query_task_adapters=False,
        apt_tube_query_task_adapter_scales=(True, False, True),
        apt_tube_query_task_adapter_ratio=0.25,
        apt_tube_query_task_adapter_temporal_kernel=3,
        apt_tube_query_feedback=True,
        apt_tube_query_deformable_points=1,
        apt_tube_query_boundary_recurrent=False,
        apt_tube_query_sparse_refine=False,
        apt_sparse_refine_cls=True,
        apt_sparse_refine_box=True,
        apt_sparse_refine_obj=True,
        apt_dense_residual_adapter=False,
        apt_dense_residual_hidden_ratio=0.125,
        apt_dense_residual_temporal_kernels=(3, 3, 3),
        apt_dense_residual_scales=(True, True, True),
        apt_dense_residual_class=True,
        apt_dense_residual_box=True,
        apt_dense_residual_object=True,
        apt_query_trajectory_residual_adapter=False,
        apt_query_trajectory_residual_hidden=64,
        apt_query_trajectory_residual_kernels=(3, 7, 15, 31),
        apt_query_trajectory_residual_max_box_delta=0.05,
        apt_query_trajectory_residual_class=True,
        apt_query_trajectory_residual_box=True,
        apt_query_trajectory_residual_visibility=True,
        apt_query_trajectory_residual_boundary=True,
        apt_query_trajectory_residual_endpoints=True,
        apt_dense_tube_geometry_contract=False,
        apt_dense_tube_geometry_hidden=64,
        apt_dense_tube_geometry_kernels=(3, 7, 15, 31),
        apt_dense_tube_geometry_proposals=16,
        apt_dense_tube_geometry_match_temperature=0.2,
        apt_dense_tube_geometry_max_blend=0.5,
        apt_dense_tube_geometry_smooth_corrections=False,
        native_motion_pyramid=False,
        native_motion_init=None,
        native_motion_freeze=False,
        native_motion_actor_local=False,
        native_motion_actor_floor=0.1,
        native_motion_channel_ratio=1.0,
        native_motion_ablation_disabled=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.clip_length = clip_length
        self.model_id = model_id
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.backbone_gradient_checkpointing = bool(backbone_gradient_checkpointing)
        self.class_ctx_grad_checkpoint = bool(class_ctx_grad_checkpoint)
        self.backbone_frames = backbone_frames
        self.backbone_frame_sampling = str(backbone_frame_sampling)
        self.backbone_sampling_temperature = float(backbone_sampling_temperature)
        self.backbone_sampling_blend = float(backbone_sampling_blend)
        learned_sampling = {
            "learned_global", "learned_spatial", "learned_spatial_blend"
        }
        if self.backbone_frame_sampling in learned_sampling:
            self.learned_temporal_sampler = LearnedTemporalAntiAlias(
                hidden=int(backbone_sampling_hidden),
                spatial=self.backbone_frame_sampling != "learned_global",
                center_blend=self.backbone_frame_sampling == "learned_spatial_blend",
            )
        else:
            self.learned_temporal_sampler = None
        self.return_tube_queries = False
        self.apt_pyramid_adapter_enabled = apt_pyramid_adapter
        self.apt_context_gate_enabled = apt_context_gate
        self.apt_actor_context_enabled = apt_actor_context
        self.native_motion_pyramid_enabled = native_motion_pyramid
        self.native_motion_freeze = native_motion_freeze
        self.native_motion_actor_local_enabled = bool(native_motion_actor_local)
        self.native_motion_ablation_disabled = bool(native_motion_ablation_disabled)
        self.spatial_query_adapter_enabled = bool(spatial_query_adapter)
        self.keyframe_pyramid_enabled = bool(keyframe_pyramid)
        self.keyframe_freeze = bool(keyframe_freeze)
        self.backbone_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)

        hf_model = _load_videomae_pretrained(
            VideoMAEForVideoClassification,
            model_id,
            self.backbone_dtype,
        )
        self.backbone = hf_model.videomae
        cfg = self.backbone.config
        hidden = int(cfg.hidden_size)
        patch = int(getattr(cfg, "patch_size", 16))
        tubelet = int(getattr(cfg, "tubelet_size", 2))
        self.backbone_image_size = int(getattr(cfg, "image_size", img_size))
        self.tubelet_size = tubelet
        self.patch_size = patch

        if backbone_frames % tubelet != 0:
            raise ValueError(
                f"backbone_frames={backbone_frames} must be divisible by tubelet_size={tubelet}"
            )
        from .geometry import image_hw
        self.img_h, self.img_w = image_hw(img_size)
        if self.img_h % patch != 0 or self.img_w % patch != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch}")

        self.token_t = backbone_frames // tubelet
        self.grid_h = self.img_h // patch
        self.grid_w = self.img_w // patch
        self.grid_size = self.grid_h if self.grid_h == self.grid_w else None
        position = self.backbone.embeddings.position_embeddings
        expected_tokens = self.token_t * self.grid_h * self.grid_w
        self.position_embedding_mode = str(position_embedding_mode).lower()
        if self.position_embedding_mode not in ("interpolate", "regenerate"):
            raise ValueError(
                "position_embedding_mode must be 'interpolate' or "
                f"'regenerate', got {position_embedding_mode}"
            )
        if (self.position_embedding_mode == "regenerate"
                and position.shape[1] != expected_tokens):
            regenerated = _regenerate_sinusoid_table(
                expected_tokens, hidden
            ).to(dtype=position.dtype, device=position.device)
            if isinstance(position, nn.Parameter):
                regenerated = nn.Parameter(
                    regenerated, requires_grad=position.requires_grad
                )
            self.backbone.embeddings.position_embeddings = regenerated
        elif position.shape[1] != expected_tokens:
            source_grid = self.backbone_image_size // patch
            source_spatial_tokens = source_grid * source_grid
            if position.shape[1] % source_spatial_tokens != 0:
                raise ValueError(
                    "VideoMAE position tokens cannot be reshaped for interpolation: "
                    f"tokens={position.shape[1]}, source_grid={source_grid}"
                )
            source_time = position.shape[1] // source_spatial_tokens
            resized = position.detach().reshape(
                1, source_time, source_grid, source_grid, hidden
            ).permute(0, 4, 1, 2, 3)
            resized = F.interpolate(
                resized.float(),
                size=(self.token_t, self.grid_h, self.grid_w),
                mode="trilinear",
                align_corners=False,
            ).to(dtype=position.dtype)
            resized = resized.permute(0, 2, 3, 4, 1).reshape(
                1, expected_tokens, hidden
            )
            if isinstance(position, nn.Parameter):
                resized = nn.Parameter(resized, requires_grad=position.requires_grad)
            self.backbone.embeddings.position_embeddings = resized

        patch_embeddings = self.backbone.embeddings.patch_embeddings
        patch_embeddings.image_size = (self.img_h, self.img_w)
        patch_embeddings.num_patches = expected_tokens
        self.backbone.embeddings.num_patches = expected_tokens
        cfg.image_size = img_size
        cfg.num_frames = backbone_frames

        self._configure_backbone_training()

        self.reduce = ConvBNAct3D(hidden, 256, k=1)
        if self.spatial_query_adapter_enabled:
            self.spatial_query_adapter = SpatialQueryAdapter(
                hidden,
                out_channels=256,
                spatial_size=int(spatial_query_grid),
                num_heads=int(spatial_query_heads),
                dropout=float(spatial_query_dropout),
                temporal_steps=self.token_t,
            )
        else:
            self.spatial_query_adapter = None
        if temporal_adapter:
            self.temporal_adapter = nn.Sequential(*[
                TemporalAdapterBlock(
                    256,
                    expansion=temporal_adapter_expansion,
                    kernel_size=temporal_adapter_kernel,
                    dropout=temporal_adapter_dropout,
                )
                for _ in range(temporal_adapter_depth)
            ])
        else:
            self.temporal_adapter = nn.Identity()

        keyframe_scales = tuple(bool(value) for value in keyframe_scales)
        if len(keyframe_scales) != 3:
            raise ValueError("keyframe_scales must contain three booleans")
        self.keyframe_scales = keyframe_scales
        if self.keyframe_pyramid_enabled:
            from .keyframe_yolo11 import YOLO11KeyframePyramid
            self.keyframe_backbone = YOLO11KeyframePyramid(
                version=keyframe_version,
                pretrained_path=keyframe_pretrained_path,
                frames=keyframe_frames,
                micro_batch=keyframe_micro_batch,
                freeze=self.keyframe_freeze,
                bn_eval=keyframe_bn_eval,
                grad_checkpoint=keyframe_grad_checkpoint,
            )
            semantic_channels = (256, 512, 512)
            self.keyframe_fusions = nn.ModuleList([
                KeyframePyramidFusion(semantic, keyframe)
                for semantic, keyframe in zip(
                    semantic_channels, self.keyframe_backbone.out_channels
                )
            ])
        else:
            self.keyframe_backbone = None
            self.keyframe_fusions = nn.ModuleList()

        self.p3_refine = ConvBNAct3D(256, 256, k=3)
        self.p4_refine = ConvBNAct3D(256, 512, k=3)
        self.p5 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            ConvBNAct3D(512, 512, k=3),
        )
        if apt_pyramid_adapter:
            self.apt_p3 = APTPyramidAdapter(
                256,
                depth=apt_adapter_depth,
                kernel_size=apt_adapter_kernel,
                temporal_kernel=apt_adapter_temporal_kernel,
                dropout=apt_adapter_dropout,
            )
            self.apt_p4 = APTPyramidAdapter(
                512,
                depth=apt_adapter_depth,
                kernel_size=apt_adapter_kernel,
                temporal_kernel=apt_adapter_temporal_kernel,
                dropout=apt_adapter_dropout,
            )
            self.apt_p5 = APTPyramidAdapter(
                512,
                depth=apt_adapter_depth,
                kernel_size=apt_adapter_kernel,
                temporal_kernel=apt_adapter_temporal_kernel,
                dropout=apt_adapter_dropout,
            )
        else:
            self.apt_p3 = nn.Identity()
            self.apt_p4 = nn.Identity()
            self.apt_p5 = nn.Identity()

        state_scales = tuple(bool(enabled) for enabled in apt_dilated_state_scales)
        if len(state_scales) != 3:
            raise ValueError("apt_dilated_state_scales must contain three booleans")
        state_modules = []
        for channels, enabled in zip((256, 512, 512), state_scales):
            if apt_dilated_state and enabled:
                state_modules.append(DilatedBidirectionalStateAdapter(
                    channels,
                    dilations=tuple(apt_dilated_state_dilations),
                    dropout=apt_dilated_state_dropout,
                ))
            else:
                state_modules.append(nn.Identity())
        self.state_p3, self.state_p4, self.state_p5 = state_modules

        if apt_context_gate:
            self.ctx_p3 = PyramidContextGate(256, hidden=apt_context_dim, dropout=apt_context_dropout)
            self.ctx_p4 = PyramidContextGate(512, hidden=apt_context_dim, dropout=apt_context_dropout)
            self.ctx_p5 = PyramidContextGate(512, hidden=apt_context_dim, dropout=apt_context_dropout)
        else:
            self.ctx_p3 = nn.Identity()
            self.ctx_p4 = nn.Identity()
            self.ctx_p5 = nn.Identity()

        if apt_actor_context:
            actor_gate = MultiActorContextGate if apt_actor_context_slots > 1 else ActorContextGate
            actor_kwargs = {"slots": apt_actor_context_slots} if apt_actor_context_slots > 1 else {}
            self.actor_ctx_p3 = actor_gate(
                256,
                hidden=apt_actor_context_dim,
                num_heads=apt_actor_context_heads,
                depth=apt_actor_context_depth,
                dropout=apt_actor_context_dropout,
                **actor_kwargs,
            )
            self.actor_ctx_p4 = actor_gate(
                512,
                hidden=apt_actor_context_dim,
                num_heads=apt_actor_context_heads,
                depth=apt_actor_context_depth,
                dropout=apt_actor_context_dropout,
                **actor_kwargs,
            )
            self.actor_ctx_p5 = actor_gate(
                512,
                hidden=apt_actor_context_dim,
                num_heads=apt_actor_context_heads,
                depth=apt_actor_context_depth,
                dropout=apt_actor_context_dropout,
                **actor_kwargs,
            )
        else:
            self.actor_ctx_p3 = nn.Identity()
            self.actor_ctx_p4 = nn.Identity()
            self.actor_ctx_p5 = nn.Identity()

        if apt_class_context:
            self.class_ctx_p3 = ClassSpecificContextRefiner(
                256, num_classes, hidden=apt_class_context_dim, dropout=apt_class_context_dropout
            )
            self.class_ctx_p4 = ClassSpecificContextRefiner(
                512, num_classes, hidden=apt_class_context_dim, dropout=apt_class_context_dropout
            )
            self.class_ctx_p5 = ClassSpecificContextRefiner(
                512, num_classes, hidden=apt_class_context_dim, dropout=apt_class_context_dropout
            )
        else:
            self.class_ctx_p3 = None
            self.class_ctx_p4 = None
            self.class_ctx_p5 = None

        trajectory_channels = (256, 512, 512)
        trajectory_scales = tuple(bool(enabled) for enabled in apt_trajectory_scales)
        if len(trajectory_scales) != 3:
            raise ValueError("apt_trajectory_scales must contain three booleans")
        trajectory_modules = []
        for channels, enabled in zip(trajectory_channels, trajectory_scales):
            if apt_trajectory_align and enabled:
                trajectory_modules.append(LocalTrajectoryAligner(
                    channels,
                    hidden=apt_trajectory_hidden,
                    radius=apt_trajectory_radius,
                    temperature=apt_trajectory_temperature,
                ))
            else:
                trajectory_modules.append(nn.Identity())
        self.trajectory_p3, self.trajectory_p4, self.trajectory_p5 = trajectory_modules

        if apt_tube_denoising:
            self.tube_denoiser = NoisyTubeDenoiser(
                256,
                num_classes,
                hidden=apt_tube_denoising_dim,
                num_heads=apt_tube_denoising_heads,
                depth=apt_tube_denoising_depth,
                max_tubes=apt_tube_denoising_max_tubes,
                box_noise=apt_tube_denoising_box_noise,
                label_noise=apt_tube_denoising_label_noise,
            )
        else:
            self.tube_denoiser = None

        if apt_cross_clip_memory:
            self.cross_clip_memory = CrossClipActorMemory(
                256,
                hidden=apt_cross_clip_memory_dim,
                chunk_size=apt_cross_clip_memory_chunk,
                bidirectional=apt_cross_clip_memory_bidirectional,
                dropout=apt_cross_clip_memory_dropout,
            )
        else:
            self.cross_clip_memory = None
        self.cross_clip_memory_target = str(
            apt_cross_clip_memory_target
        ).lower()
        if self.cross_clip_memory_target not in {"features", "query_decisions"}:
            raise ValueError(
                "apt_cross_clip_memory_target must be 'features' or "
                "'query_decisions'"
            )
        self.cross_clip_decision_target = str(
            apt_cross_clip_decision_target
        ).lower()
        if apt_cross_clip_memory and apt_pyramid_actor_memory:
            raise ValueError(
                "cross-clip and pyramid actor memory are mutually exclusive"
            )
        if apt_pyramid_actor_memory:
            self.pyramid_actor_memory = PyramidActorMemory(
                (256, 512, 512),
                hidden=apt_pyramid_actor_memory_dim,
                chunk_size=apt_pyramid_actor_memory_chunk,
                bidirectional=apt_pyramid_actor_memory_bidirectional,
                dropout=apt_pyramid_actor_memory_dropout,
                learned_scale_routing=(
                    apt_pyramid_actor_memory_learned_routing
                ),
            )
        else:
            self.pyramid_actor_memory = None

        self.tube_query_actor_aligned = bool(apt_tube_query_actor_aligned)
        self.tube_query_factorized = bool(apt_tube_query_factorized)
        self.tube_query_identity_transport = bool(
            apt_tube_query_identity_transport
        )
        if self.tube_query_identity_transport and not self.tube_query_factorized:
            raise ValueError("identity transport requires factorized tube queries")
        if (self.cross_clip_memory_target == "query_decisions" and
                (self.cross_clip_memory is None or
                 not self.tube_query_factorized)):
            raise ValueError(
                "query-decision memory requires cross-clip memory and "
                "factorized tube queries"
            )
        self.tube_query_shared_grad_scale = float(apt_tube_query_shared_grad_scale)
        if not 0.0 <= self.tube_query_shared_grad_scale <= 1.0:
            raise ValueError("apt_tube_query_shared_grad_scale must be in [0, 1]")
        if apt_tube_query_shared_grad_scales is None:
            self.tube_query_shared_grad_scales = (
                self.tube_query_shared_grad_scale,
                self.tube_query_shared_grad_scale,
                self.tube_query_shared_grad_scale,
            )
        else:
            self.tube_query_shared_grad_scales = tuple(
                float(scale) for scale in apt_tube_query_shared_grad_scales
            )
            if len(self.tube_query_shared_grad_scales) != 3:
                raise ValueError(
                    "apt_tube_query_shared_grad_scales must contain three values"
                )
            if any(not 0.0 <= scale <= 1.0
                   for scale in self.tube_query_shared_grad_scales):
                raise ValueError(
                    "apt_tube_query_shared_grad_scales values must be in [0, 1]"
                )
        self.tube_query_task_adapter_scales = tuple(
            bool(enabled) for enabled in apt_tube_query_task_adapter_scales
        )
        if len(self.tube_query_task_adapter_scales) != 3:
            raise ValueError(
                "apt_tube_query_task_adapter_scales must contain three booleans"
            )
        task_adapters_enabled = bool(apt_tube_query_task_adapters)
        if ((any(scale != 1.0 for scale in self.tube_query_shared_grad_scales) or
             task_adapters_enabled) and not self.tube_query_factorized):
            raise ValueError(
                "tube-query gradient routing currently requires factorized queries"
            )
        if task_adapters_enabled:
            channels = (256, 512, 512)
            self.tube_task_adapters = nn.ModuleList([
                TubeTaskAdapter(
                    channel,
                    bottleneck_ratio=apt_tube_query_task_adapter_ratio,
                    temporal_kernel=apt_tube_query_task_adapter_temporal_kernel,
                ) if enabled else nn.Identity()
                for channel, enabled in zip(
                    channels, self.tube_query_task_adapter_scales
                )
            ])
        else:
            self.tube_task_adapters = nn.ModuleList()
        self.tube_query_sparse_refine = bool(apt_tube_query_sparse_refine)
        self.sparse_refine_cls = bool(apt_sparse_refine_cls)
        self.sparse_refine_box = bool(apt_sparse_refine_box)
        self.sparse_refine_obj = bool(apt_sparse_refine_obj)
        if self.tube_query_sparse_refine:
            self.sparse_cls_scale = nn.Parameter(torch.zeros(()))
            self.sparse_box_scale = nn.Parameter(torch.zeros(()))
            self.sparse_obj_scale = nn.Parameter(torch.zeros(()))
        if apt_tube_queries and self.tube_query_actor_aligned and self.tube_query_factorized:
            raise ValueError("tube queries cannot be both actor-aligned and factorized")
        if apt_tube_queries and self.tube_query_factorized:
            self.tube_query_head = FactorizedPersonTubeletHead(
                (256, 512, 512), num_classes, hidden=apt_tube_query_dim,
                num_queries=apt_tube_query_count, num_heads=apt_tube_query_heads,
                depth=apt_tube_query_depth, dropout=apt_tube_query_dropout,
                frames=apt_tube_query_frames, max_frames=clip_length,
                memory_grid=apt_tube_query_memory_grid,
                boundary_gated=apt_tube_query_boundary_gate,
                iterative_refinement=apt_tube_query_iterative_refinement,
                trajectory_sampling=apt_tube_query_trajectory_sampling,
                trajectory_points=apt_tube_query_trajectory_points,
                interval_queries=apt_tube_query_intervals,
                instances_per_actor=apt_tube_query_instances_per_actor,
                interval_pyramid=apt_tube_query_interval_pyramid,
                drop_path_rate=apt_tube_query_drop_path_rate,
                class_prior_probability=apt_tube_query_class_prior_probability,
                predict_quality=apt_tube_query_quality,
                predict_boundary_distance=apt_tube_query_boundary_distance,
                boundary_distance_temperature=(
                    apt_tube_query_boundary_distance_temperature
                ),
                duration_router=apt_tube_query_duration_router,
                duration_kernels=apt_tube_query_duration_kernels,
                change_point_pyramid=apt_tube_query_change_point_pyramid,
                change_point_dilations=apt_tube_query_change_point_dilations,
                change_point_router_mode=apt_tube_query_change_point_router_mode,
                change_point_shared_projection=(
                    apt_tube_query_change_point_shared_projection
                ),
                identity_transport=apt_tube_query_identity_transport,
                transport_proposals=apt_tube_query_transport_proposals,
                transport_sinkhorn_iterations=(
                    apt_tube_query_transport_sinkhorn_iterations
                ),
                transport_temperature=apt_tube_query_transport_temperature,
                action_reset_state=apt_tube_query_action_reset_state,
                action_fork_state=apt_tube_query_action_fork_state,
                action_fork_mode=apt_tube_query_action_fork_mode,
                action_fork_temperature=apt_tube_query_action_fork_temperature,
                decision_memory_dim=(
                    apt_cross_clip_memory_dim
                    if self.cross_clip_memory_target == "query_decisions"
                    else 0
                ),
                decision_memory_target=self.cross_clip_decision_target,
            )
        elif apt_tube_queries and self.tube_query_actor_aligned:
            self.tube_query_head = ActorAlignedTubeQueryHead(
                (256, 512, 512), num_classes, hidden=apt_tube_query_dim,
                num_queries=apt_tube_query_count, num_heads=apt_tube_query_heads,
                depth=apt_tube_query_depth, dropout=apt_tube_query_dropout,
                max_frames=clip_length, spatial_stride=self.spatial_strides[0]
                if hasattr(self, "spatial_strides") else 8,
                img_size=img_size, feedback=apt_tube_query_feedback,
                deformable_points=apt_tube_query_deformable_points,
                boundary_recurrent=apt_tube_query_boundary_recurrent,
            )
        elif apt_tube_queries:
            self.tube_query_head = TubeQueryHead(
                256, num_classes, hidden=apt_tube_query_dim,
                num_queries=apt_tube_query_count, num_heads=apt_tube_query_heads,
                depth=apt_tube_query_depth, dropout=apt_tube_query_dropout,
                frames=apt_tube_query_frames,
                max_frames=clip_length,
            )
        else:
            self.tube_query_head = None

        if apt_query_trajectory_residual_adapter:
            if self.tube_query_head is None:
                raise ValueError(
                    "query trajectory residuals require tube queries"
                )
            self.query_trajectory_residual_adapter = (
                TubeQueryTrajectoryResidualAdapter(
                    num_classes=num_classes,
                    hidden=apt_query_trajectory_residual_hidden,
                    temporal_kernels=tuple(
                        apt_query_trajectory_residual_kernels
                    ),
                    max_box_delta=(
                        apt_query_trajectory_residual_max_box_delta
                    ),
                    class_residual=apt_query_trajectory_residual_class,
                    box_residual=apt_query_trajectory_residual_box,
                    visibility_residual=(
                        apt_query_trajectory_residual_visibility
                    ),
                    boundary_residual=(
                        apt_query_trajectory_residual_boundary
                    ),
                    endpoint_residual=(
                        apt_query_trajectory_residual_endpoints
                    ),
                )
            )
        else:
            self.query_trajectory_residual_adapter = None

        if apt_dense_tube_geometry_contract:
            if self.tube_query_head is None:
                raise ValueError(
                    "dense-to-tube geometry requires tube queries"
                )
            self.dense_tube_geometry_contract = DenseTubeGeometryContract(
                num_classes=num_classes,
                hidden=apt_dense_tube_geometry_hidden,
                temporal_kernels=tuple(
                    apt_dense_tube_geometry_kernels
                ),
                proposals=apt_dense_tube_geometry_proposals,
                match_temperature=(
                    apt_dense_tube_geometry_match_temperature
                ),
                max_blend=apt_dense_tube_geometry_max_blend,
                smooth_corrections=(
                    apt_dense_tube_geometry_smooth_corrections
                ),
            )
        else:
            self.dense_tube_geometry_contract = None

        if native_motion_pyramid:
            self.motion_backbone = YOLOST_Backbone(use_checkpoint=True)
            self.motion_neck = TemporalPyramidNeck(target_T=[64, 32, 16])
            motion_fusion = (
                ActorLocalSignedMotionFusion
                if self.native_motion_actor_local_enabled
                else NativeMotionFusion
            )
            motion_kwargs = (
                {
                    "actor_floor": native_motion_actor_floor,
                    "channel_ratio": native_motion_channel_ratio,
                }
                if self.native_motion_actor_local_enabled
                else {}
            )
            self.motion_fuse_p3 = motion_fusion(256, **motion_kwargs)
            self.motion_fuse_p4 = motion_fusion(512, **motion_kwargs)
            self.motion_fuse_p5 = motion_fusion(512, **motion_kwargs)
            if native_motion_init:
                self._load_native_motion_init(native_motion_init)
            if native_motion_freeze:
                for module in (self.motion_backbone, self.motion_neck):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        else:
            self.motion_backbone = None
            self.motion_neck = None

        self.spatial_strides = [8, 16, 32]
        self.temporal_strides = [1, 2, 4]
        self.reg_max = int(reg_max)
        self.head = DecoupledHeadBoundary(
            in_channels_list=[256, 512, 512],
            num_classes=num_classes,
            strides=self.spatial_strides,
            img_size=img_size,
            reg_max=self.reg_max,
        )
        if apt_dense_residual_adapter:
            self.dense_residual_adapters = PyramidAlignedResidualAdapters(
                in_channels=(256, 512, 512),
                num_classes=num_classes,
                hidden_ratio=apt_dense_residual_hidden_ratio,
                temporal_kernels=tuple(apt_dense_residual_temporal_kernels),
                scales=tuple(apt_dense_residual_scales),
                class_residual=apt_dense_residual_class,
                box_residual=apt_dense_residual_box,
                object_residual=apt_dense_residual_object,
            )
        else:
            self.dense_residual_adapters = None

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone or self.unfreeze_last_n_blocks <= 0:
            self.backbone.eval()
        if self.native_motion_freeze and self.motion_backbone is not None:
            self.motion_backbone.eval()
            self.motion_neck.eval()
        if self.keyframe_freeze and self.keyframe_backbone is not None:
            self.keyframe_backbone.eval()
        return self

    def _load_native_motion_init(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        backbone_state = {
            key[len("backbone."):]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        neck_state = {
            key[len("neck."):]: value
            for key, value in state.items()
            if key.startswith("neck.")
        }
        backbone_result = self.motion_backbone.load_state_dict(backbone_state, strict=False)
        neck_result = self.motion_neck.load_state_dict(neck_state, strict=False)
        if backbone_result.unexpected_keys or neck_result.unexpected_keys:
            raise ValueError(f"Unexpected native motion checkpoint keys in {checkpoint_path}")

    def enable_persistent_memory(self, enabled=True):
        if self.cross_clip_memory is not None:
            self.cross_clip_memory.enable_persistent_eval(enabled)
        if self.pyramid_actor_memory is not None:
            self.pyramid_actor_memory.enable_persistent_eval(enabled)

    def reset_memory(self):
        if self.cross_clip_memory is not None:
            self.cross_clip_memory.reset_memory()
        if self.pyramid_actor_memory is not None:
            self.pyramid_actor_memory.reset_memory()

    def enable_tube_query_output(self, enabled=True):
        self.return_tube_queries = bool(enabled)

    def _configure_backbone_training(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        if self.freeze_backbone or self.unfreeze_last_n_blocks <= 0:
            self.backbone.eval()
            return

        layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
        if layers is None:
            raise ValueError("Could not find VideoMAE transformer layers at backbone.encoder.layer")

        n = min(int(self.unfreeze_last_n_blocks), len(layers))
        for block in layers[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)

        self.backbone.train()
        if self.backbone_gradient_checkpointing:
            # Memory only: recomputes encoder activations in backward.
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )

    def _apply_sparse_query_refinement(self, dense_output, query_output):
        """Correct only proposal-aligned dense predictions with tube states."""
        cls_logits, reg_logits, obj_logits, *rest = dense_output
        batch, _, time, height, width = reg_logits.shape
        indices = query_output["proposal_indices"]
        queries = indices.shape[-1]
        flat_size = height * width

        def scatter(values, channels):
            values = values.permute(0, 2, 1, 3)
            target = values.new_zeros(batch, time, flat_size, channels)
            expanded = indices.unsqueeze(-1).expand(-1, -1, -1, channels)
            target = target.scatter_add(2, expanded, values)
            counts = values.new_zeros(batch, time, flat_size, 1)
            counts = counts.scatter_add(2, indices.unsqueeze(-1), values.new_ones(
                batch, time, queries, 1
            ))
            return target / counts.clamp_min(1)

        if self.sparse_refine_cls:
            cls_delta = scatter(query_output["frame_class_logits"], self.num_classes)
            cls_flat = cls_logits.permute(0, 2, 3, 4, 1).reshape(
                batch, time, flat_size, self.num_classes
            )
            cls_flat = cls_flat + self.sparse_cls_scale.tanh() * cls_delta
            cls_logits = cls_flat.reshape(batch, time, height, width, self.num_classes)
            cls_logits = cls_logits.permute(0, 4, 1, 2, 3).contiguous()

        if self.sparse_refine_obj:
            obj_delta = scatter(query_output["visibility_logits"].unsqueeze(-1), 1)
            obj_flat = obj_logits.permute(0, 2, 3, 4, 1).reshape(batch, time, flat_size, 1)
            obj_flat = obj_flat + self.sparse_obj_scale.tanh() * obj_delta
            obj_logits = obj_flat.reshape(batch, time, height, width, 1)
            obj_logits = obj_logits.permute(0, 4, 1, 2, 3).contiguous()

        if self.sparse_refine_box:
            boxes = query_output["boxes"].permute(0, 2, 1, 3)
            centers = 0.5 * (boxes[..., :2] + boxes[..., 2:])
            sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1e-5)
            y_index = torch.div(indices, width, rounding_mode="floor")
            x_index = indices.remainder(width)
            if self.img_h != self.img_w:
                raise NotImplementedError('sparse_refine_box assumes square inputs')
            step = self.spatial_strides[0] / self.img_h
            offsets = torch.stack([
                centers[..., 0] / step - x_index.to(boxes.dtype),
                centers[..., 1] / step - y_index.to(boxes.dtype),
            ], -1).clamp(1e-4, 1 - 1e-4)
            desired = torch.cat([
                torch.logit(offsets), (sizes / step).clamp_min(1e-5).log()
            ], -1)
            base = reg_logits.permute(0, 2, 3, 4, 1).reshape(batch, time, flat_size, 4)
            gathered = base.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 4))
            box_delta = scatter((desired - gathered).permute(0, 2, 1, 3), 4)
            base = base + self.sparse_box_scale.tanh() * box_delta
            reg_logits = base.reshape(batch, time, height, width, 4)
            reg_logits = reg_logits.permute(0, 4, 1, 2, 3).contiguous()

        return (cls_logits, reg_logits, obj_logits, *rest)

    def _sample_backbone_frames(self, x):
        """Build the fixed-size VideoMAE input from the full-rate clip."""
        b, c, t, h, w = x.shape
        if t == self.backbone_frames:
            return x
        mode = self.backbone_frame_sampling
        if self.learned_temporal_sampler is not None:
            return self.learned_temporal_sampler(x, self.backbone_frames)
        if mode == "uniform":
            idx = torch.linspace(
                0, t - 1, self.backbone_frames, device=x.device
            ).round().long()
            return x.index_select(2, idx)
        valid_modes = {
            "mean", "center_blend", "motion_max", "motion_soft", "boundary_soft"
        }
        if mode not in valid_modes:
            raise ValueError(f"unknown backbone_frame_sampling={mode!r}")

        signal = x.float().mean(dim=1)
        motion = signal.new_zeros(b, t)
        motion[:, 1:] = (signal[:, 1:] - signal[:, :-1]).abs().mean(dim=(2, 3))
        boundary = signal.new_zeros(b, t)
        boundary[:, 2:] = (
            signal[:, 2:] - 2.0 * signal[:, 1:-1] + signal[:, :-2]
        ).abs().mean(dim=(2, 3))
        score = boundary if mode == "boundary_soft" else motion
        edges = torch.linspace(
            0, t, self.backbone_frames + 1, device=x.device
        ).round().long()
        sampled = []
        for index in range(self.backbone_frames):
            start, end = int(edges[index]), int(edges[index + 1])
            end = max(end, start + 1)
            frames = x[:, :, start:end]
            if mode == "mean":
                pooled = frames.mean(dim=2)
            elif mode == "center_blend":
                mean_frame = frames.mean(dim=2)
                center_frame = frames[:, :, (end - start - 1) // 2]
                blend = min(max(self.backbone_sampling_blend, 0.0), 1.0)
                pooled = blend * mean_frame + (1.0 - blend) * center_frame
            elif mode == "motion_max":
                selected = score[:, start:end].argmax(dim=1) + start
                gather_index = selected.view(b, 1, 1, 1, 1).expand(
                    -1, c, 1, h, w
                )
                pooled = x.gather(2, gather_index).squeeze(2)
            else:
                local_score = score[:, start:end]
                local_score = local_score - local_score.mean(dim=1, keepdim=True)
                scale = local_score.std(dim=1, keepdim=True).clamp_min(1e-6)
                weights = (
                    local_score / scale / max(self.backbone_sampling_temperature, 1e-3)
                ).softmax(dim=1)
                pooled = (frames * weights[:, None, :, None, None]).sum(dim=2)
            sampled.append(pooled)
        return torch.stack(sampled, dim=2)

    def _extract_videomae_maps(self, x):
        """Return VideoMAE dense map as (B, C, Ttok, Gh, Gw)."""
        x = self._sample_backbone_frames(x)
        video = x.permute(0, 2, 1, 3, 4).to(dtype=self.backbone_dtype)
        grad_enabled = (not self.freeze_backbone) and self.training

        context = torch.enable_grad() if grad_enabled else torch.no_grad()
        with context:
            out = self.backbone(pixel_values=video)
            tokens = out.last_hidden_state

        b = tokens.shape[0]
        dense = tokens.reshape(
            b,
            self.token_t,
            self.grid_h,
            self.grid_w,
            -1,
        )
        return dense.permute(0, 4, 1, 2, 3).contiguous().float()

    def forward(self, x, targets=None):
        dense = self._extract_videomae_maps(x)
        if self.spatial_query_adapter is not None:
            tokens = dense.permute(0, 2, 3, 4, 1).reshape(
                dense.shape[0], -1, dense.shape[1]
            )
            base = self.spatial_query_adapter(tokens)
            base = self.temporal_adapter(base)
        else:
            base = self.temporal_adapter(self.reduce(dense))
        motion_pyramid = None
        if self.motion_backbone is not None and not self.native_motion_ablation_disabled:
            motion_context = torch.no_grad() if self.native_motion_freeze else torch.enable_grad()
            with motion_context:
                motion_features = self.motion_backbone(x)
                motion_pyramid = self.motion_neck(*motion_features)
        keyframe_pyramid = (
            self.keyframe_backbone(x) if self.keyframe_backbone is not None else None
        )
        p3 = F.interpolate(
            base,
            size=(self.clip_length, self.grid_h * 2, self.grid_w * 2),
            mode="trilinear",
            align_corners=False,
        )
        p3 = self.p3_refine(p3)
        p3 = self.apt_p3(p3)
        p3 = self.state_p3(p3)
        if keyframe_pyramid is not None and self.keyframe_scales[0]:
            p3 = self.keyframe_fusions[0](p3, keyframe_pyramid[0])
        if motion_pyramid is not None and not self.native_motion_actor_local_enabled:
            p3 = self.motion_fuse_p3(p3, motion_pyramid[0])
        p3 = self.trajectory_p3(self.actor_ctx_p3(self.ctx_p3(p3)))
        memory_consistency = None
        decision_memory_context = None
        if self.cross_clip_memory is not None:
            if self.cross_clip_memory_target == "features":
                p3, memory_consistency = self.cross_clip_memory(p3)
            else:
                _, memory_consistency, decision_memory_context = (
                    self.cross_clip_memory(p3, return_context=True)
                )
        p4 = F.interpolate(
            base,
            size=(self.clip_length // 2, self.grid_h, self.grid_w),
            mode="trilinear",
            align_corners=False,
        )
        p4 = self.p4_refine(p4)
        p4 = self.apt_p4(p4)
        p4 = self.state_p4(p4)
        if keyframe_pyramid is not None and self.keyframe_scales[1]:
            p4 = self.keyframe_fusions[1](p4, keyframe_pyramid[1])
        if motion_pyramid is not None and not self.native_motion_actor_local_enabled:
            p4 = self.motion_fuse_p4(p4, motion_pyramid[1])
        p4 = self.trajectory_p4(self.actor_ctx_p4(self.ctx_p4(p4)))
        p5 = self.p5(p4)
        p5 = self.apt_p5(p5)
        p5 = self.state_p5(p5)
        if keyframe_pyramid is not None and self.keyframe_scales[2]:
            p5 = self.keyframe_fusions[2](p5, keyframe_pyramid[2])
        if motion_pyramid is not None and not self.native_motion_actor_local_enabled:
            p5 = self.motion_fuse_p5(p5, motion_pyramid[2])
        p5 = self.trajectory_p5(self.actor_ctx_p5(self.ctx_p5(p5)))
        features = [p3, p4, p5]
        if self.pyramid_actor_memory is not None:
            features, memory_consistency = self.pyramid_actor_memory(features)
        if motion_pyramid is not None and self.native_motion_actor_local_enabled:
            provisional_outputs = self.head(features)
            features = [
                fusion(feature, motion, torch.sigmoid(output[2]))
                for fusion, feature, motion, output in zip(
                    (self.motion_fuse_p3, self.motion_fuse_p4, self.motion_fuse_p5),
                    features,
                    motion_pyramid,
                    provisional_outputs,
                )
            ]
            p3, p4, p5 = features
        outputs = self.head(features)
        tube_query_output = None
        if self.tube_query_head is not None and self.tube_query_factorized:
            query_features = features
            if (self.tube_query_shared_grad_scales != (1.0, 1.0, 1.0) or
                    len(self.tube_task_adapters) > 0):
                query_features = []
                for index, (feature, scale) in enumerate(zip(
                        features, self.tube_query_shared_grad_scales)):
                    detached = feature.detach()
                    routed = detached + scale * (feature - detached)
                    if (len(self.tube_task_adapters) > 0 and
                            self.tube_query_task_adapter_scales[index]):
                        routed = routed + self.tube_task_adapters[index](detached)
                    query_features.append(routed)
            tube_query_output = self.tube_query_head(
                query_features,
                outputs[0] if self.tube_query_identity_transport else None,
                decision_context=decision_memory_context,
            )
        elif self.tube_query_head is not None and self.tube_query_actor_aligned:
            tube_query_output = self.tube_query_head(features, outputs[0])
            feedback = tube_query_output.pop("feedback")
            if feedback is not None:
                p3 = p3 + feedback
                features = [p3, p4, p5]
                outputs = self.head(features)
            if self.tube_query_sparse_refine:
                outputs[0] = self._apply_sparse_query_refinement(outputs[0], tube_query_output)
        if (tube_query_output is not None
                and self.query_trajectory_residual_adapter is not None):
            tube_query_output = self.query_trajectory_residual_adapter(
                tube_query_output
            )
        if (tube_query_output is not None
                and self.dense_tube_geometry_contract is not None):
            tube_query_output = self.dense_tube_geometry_contract(
                tube_query_output, outputs[0]
            )
        refiners = [self.class_ctx_p3, self.class_ctx_p4, self.class_ctx_p5]
        if any(refiner is not None for refiner in refiners):
            refined = []
            for feature, output, refiner in zip(features, outputs, refiners):
                cls_logits, reg_logits, obj_logits, bnd_logits = output
                if (self.class_ctx_grad_checkpoint and self.training
                        and torch.is_grad_enabled()):
                    # Memory only: the refiner is recomputed in backward.
                    from torch.utils.checkpoint import checkpoint as _checkpoint
                    cls_logits = _checkpoint(
                        refiner, feature, cls_logits, obj_logits, use_reentrant=False
                    )
                else:
                    cls_logits = refiner(feature, cls_logits, obj_logits)
                refined.append((cls_logits, reg_logits, obj_logits, bnd_logits))
            outputs = refined
        if self.dense_residual_adapters is not None:
            outputs = self.dense_residual_adapters(features, outputs)
        if ((self.training and (self.tube_denoiser is not None or
                                memory_consistency is not None or
                                self.tube_query_head is not None)) or
                (not self.training and self.return_tube_queries and
                 self.tube_query_head is not None)):
            result = {"dense": outputs}
            if self.tube_denoiser is not None and targets is not None:
                result["tube_denoising"] = self.tube_denoiser(
                    p3, targets, self.clip_length
                )
            if memory_consistency is not None:
                result["memory_consistency"] = memory_consistency
            if self.tube_query_head is not None:
                query_result = (tube_query_output if tube_query_output is not None
                                else self.tube_query_head(p3))
                result["tube_queries"] = {
                    key: value for key, value in query_result.items()
                    if key not in ("proposal_indices", "frame_class_logits")
                }
            return result
        return outputs
