"""YOLO-ST Training Script.

Supports DDP multi-GPU training with mixed precision.

Usage:
    # Single GPU
    python train.py --config configs/yolost_s_ucf24.yaml

    # Multi-GPU DDP
    torchrun --nproc_per_node=2 train.py --config configs/yolost_s_ucf24.yaml
"""

import argparse
import json
import os
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy

from yolost.model import YOLOST
from yolost.loss import YOLOSTLoss
from data.ucf101_24 import UCF101_24_Dataset
from data.jhmdb import JHMDBDataset
from data.synthetic import SyntheticActionDataset
from data.ava_dataset import AVADataset
from data.multisports_dataset import MultiSportsDataset
from data.ava_kinetics_weak_dataset import AVAKineticsWeakDataset
from config_utils import load_config


def lock_frozen_batchnorm_stats(module):
    """Put BatchNorm layers with no trainable affine parameters in eval mode."""
    locked = 0
    for child in module.modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        if any(parameter.requires_grad for parameter in child.parameters(
                recurse=False)):
            continue
        child.eval()
        locked += 1
    return locked


def lock_all_batchnorm_stats(module):
    """Put every BatchNorm layer in eval mode without freezing its affine terms."""
    locked = 0
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()
            locked += 1
    return locked


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Matching YOWOv3: decay = 0.9999 * (1 - exp(-updates / 2000))
    """

    def __init__(self, model, decay=0.9999, warmup=2000):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.warmup))
        with torch.no_grad():
            # Update parameters with EMA
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.mul_(d).add_(model_p, alpha=1 - d)
            # Copy buffers (BN running_mean/var) directly from model
            for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
                ema_b.copy_(model_b)

    def state_dict(self):
        return {'ema': self.ema.state_dict(), 'updates': self.updates}

    def load_state_dict(self, state):
        self.ema.load_state_dict(state['ema'])
        self.updates = state['updates']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--init', type=str, default=None,
                        help='Load compatible model weights without optimizer state')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for debugging')
    return parser.parse_args()


_CLIP_MEAN = (0.485, 0.456, 0.406)
_CLIP_STD = (0.229, 0.224, 0.225)


def normalize_uint8_clips(clips):
    """(B, 3, T, H, W) uint8 -> ImageNet-normalised float32, as the loaders do."""
    if clips.dtype != torch.uint8:
        return clips
    mean = torch.tensor(_CLIP_MEAN, device=clips.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(_CLIP_STD, device=clips.device).view(1, 3, 1, 1, 1)
    return (clips.float() / 255.0 - mean) / std


def collate_fn(batch):
    """Custom collate that pads GT boxes to same length within batch."""
    clips = torch.stack([b[0] for b in batch])
    max_boxes = max(b[1]['boxes'].shape[0] for b in batch)
    max_boxes = max(max_boxes, 1)  # at least 1 to avoid empty tensor issues

    batch_boxes = torch.zeros(len(batch), max_boxes, 5)
    first_labels = batch[0][1]['labels']
    multilabel = first_labels.ndim == 2
    if multilabel:
        batch_labels = torch.zeros(len(batch), max_boxes, first_labels.shape[1])
    else:
        batch_labels = torch.zeros(len(batch), max_boxes, dtype=torch.long)

    has_boundary = 'boundary' in batch[0][1]
    if has_boundary:
        batch_boundary = torch.zeros(len(batch), max_boxes)

    for i, (_, targets) in enumerate(batch):
        n = targets['boxes'].shape[0]
        if n > 0:
            batch_boxes[i, :n] = targets['boxes']
            batch_labels[i, :n] = targets['labels']
            if has_boundary:
                batch_boundary[i, :n] = targets['boundary']

    targets = {'boxes': batch_boxes, 'labels': batch_labels}
    if 'supervised_frames' in batch[0][1]:
        width = max(b[1]['supervised_frames'].numel() for b in batch)
        supervised = torch.full((len(batch), width), -1, dtype=torch.long)
        for i, (_, sample_targets) in enumerate(batch):
            frames = sample_targets['supervised_frames'].reshape(-1)
            supervised[i, :frames.numel()] = frames
        targets['supervised_frames'] = supervised
    if has_boundary:
        targets['boundary'] = batch_boundary
    if 'frame_weights' in batch[0][1]:
        targets['frame_weights'] = torch.stack([b[1]['frame_weights'] for b in batch])
    if 'offline_boxes' in batch[0][1]:
        max_offline = max(
            item[1]['offline_boxes'].shape[0] for item in batch
        )
        max_offline = max(max_offline, 1)
        offline_boxes = torch.zeros(len(batch), max_offline, 5)
        offline_labels = torch.zeros(
            len(batch), max_offline, dtype=torch.long
        )
        offline_track_ids = torch.full(
            (len(batch), max_offline), -1, dtype=torch.long
        )
        offline_scores = torch.zeros(len(batch), max_offline)
        offline_quality = torch.zeros(len(batch), max_offline)
        offline_observed = torch.zeros(
            len(batch), max_offline, dtype=torch.bool
        )
        for index, (_, item_targets) in enumerate(batch):
            count = item_targets['offline_boxes'].shape[0]
            if count:
                offline_boxes[index, :count] = item_targets['offline_boxes']
                offline_labels[index, :count] = item_targets['offline_labels']
                offline_track_ids[index, :count] = item_targets[
                    'offline_track_ids'
                ]
                offline_scores[index, :count] = item_targets['offline_scores']
                offline_quality[index, :count] = item_targets.get(
                    'offline_quality',
                    torch.ones_like(item_targets['offline_scores']),
                )
                offline_observed[index, :count] = item_targets.get(
                    'offline_observed',
                    torch.ones_like(
                        item_targets['offline_scores'], dtype=torch.bool
                    ),
                )
        targets.update({
            'offline_boxes': offline_boxes,
            'offline_labels': offline_labels,
            'offline_track_ids': offline_track_ids,
            'offline_scores': offline_scores,
            'offline_quality': offline_quality,
            'offline_observed': offline_observed,
        })
    return clips, targets


def setup_ddp():
    """Initialize DDP if running with torchrun."""
    if 'RANK' in os.environ:
        # Ultralytics issues a dist.barrier() at import time via
        # torch_distributed_zero_first. yolost/keyframe_yolo11.py imports it
        # lazily during model construction, which is after the NCCL group
        # exists, and NCCL rejects that barrier. Importing it here, before
        # init_process_group, makes the barrier a no-op.
        try:
            import ultralytics  # noqa: F401
        except Exception:
            pass
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def is_main_process(rank):
    return rank == 0


def log(msg, rank=0):
    if is_main_process(rank):
        print(msg, flush=True)


def gradient_conflict_metrics(frame_loss, tube_boundary_loss, features):
    """Measure objective alignment at each shared pyramid feature tensor."""
    frame_grads = torch.autograd.grad(
        frame_loss, features, retain_graph=True, allow_unused=True
    )
    tube_grads = torch.autograd.grad(
        tube_boundary_loss, features, retain_graph=True, allow_unused=True
    )
    metrics = []
    for level, (frame_grad, tube_grad) in enumerate(
            zip(frame_grads, tube_grads), start=3):
        if frame_grad is None or tube_grad is None:
            metrics.append({'level': f'P{level}', 'connected': False})
            continue

        frame_grad = frame_grad.detach().float()
        tube_grad = tube_grad.detach().float()
        frame_flat = frame_grad.reshape(-1)
        tube_flat = tube_grad.reshape(-1)
        frame_norm = frame_flat.norm()
        tube_norm = tube_flat.norm()
        denom = (frame_norm * tube_norm).clamp(min=1e-12)
        cosine = torch.dot(frame_flat, tube_flat) / denom

        channels = frame_grad.shape[1]
        frame_channels = frame_grad.transpose(0, 1).reshape(channels, -1)
        tube_channels = tube_grad.transpose(0, 1).reshape(channels, -1)
        channel_denom = (
            frame_channels.norm(dim=1) * tube_channels.norm(dim=1)
        )
        valid = channel_denom > 1e-12
        channel_cosine = torch.zeros_like(channel_denom)
        channel_cosine[valid] = (
            (frame_channels[valid] * tube_channels[valid]).sum(dim=1) /
            channel_denom[valid]
        )
        valid_cosine = channel_cosine[valid]
        metrics.append({
            'level': f'P{level}',
            'connected': True,
            'cosine': float(cosine.item()),
            'frame_grad_norm': float(frame_norm.item()),
            'tube_boundary_grad_norm': float(tube_norm.item()),
            'tube_to_frame_norm_ratio': float(
                (tube_norm / frame_norm.clamp(min=1e-12)).item()
            ),
            'valid_channels': int(valid.sum().item()),
            'negative_channel_fraction': float(
                (valid_cosine < 0).float().mean().item()
            ) if valid_cosine.numel() else 0.0,
            'channel_cosine_mean': float(valid_cosine.mean().item())
            if valid_cosine.numel() else 0.0,
            'channel_cosine_median': float(valid_cosine.median().item())
            if valid_cosine.numel() else 0.0,
        })
    return metrics


def summarize_gradient_conflicts(records):
    summary = {}
    for level in ('P3', 'P4', 'P5'):
        level_records = [
            metric for record in records for metric in record['levels']
            if metric['level'] == level and metric.get('connected', False)
        ]
        if not level_records:
            summary[level] = {'measurements': 0}
            continue
        summary[level] = {
            'measurements': len(level_records),
            'cosine_mean': float(np.mean([m['cosine'] for m in level_records])),
            'cosine_std': float(np.std([m['cosine'] for m in level_records])),
            'cosine_negative_fraction': float(np.mean([
                m['cosine'] < 0 for m in level_records
            ])),
            'negative_channel_fraction_mean': float(np.mean([
                m['negative_channel_fraction'] for m in level_records
            ])),
            'tube_to_frame_norm_ratio_mean': float(np.mean([
                m['tube_to_frame_norm_ratio'] for m in level_records
            ])),
        }
    return summary


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train():
    args = parse_args()
    if args.resume and args.init:
        raise ValueError('--resume and --init are mutually exclusive')
    cfg = load_config(args.config)
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    base_seed = int(cfg['train'].get('seed', 42))
    process_seed = base_seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    # Create experiment directory
    exp_dir = cfg['output']['exp_dir']
    if is_main_process(rank):
        os.makedirs(exp_dir, exist_ok=True)

    # Model
    model_type = cfg['model'].get('type', 'uniform')
    if model_type == 'phase3':
        from yolost.model_phase3 import YOLOST_Phase3
        model = YOLOST_Phase3(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            target_T=cfg['model'].get('target_T', [64, 32, 16]),
            clip_length=cfg['data']['clip_length'],
        ).to(device)
    elif model_type == 'phase3_temporal_context':
        from yolost.model_phase3_temporal_context import YOLOST_Phase3TemporalContext
        model = YOLOST_Phase3TemporalContext(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            target_T=cfg['model'].get('target_T', [64, 32, 16]),
            clip_length=cfg['data']['clip_length'],
            context_dim=cfg['model'].get('context_dim', 128),
            context_heads=cfg['model'].get('context_heads', 4),
            context_depth=cfg['model'].get('context_depth', 1),
            context_dropout=cfg['model'].get('context_dropout', 0.0),
        ).to(device)
    elif model_type == 'bmvit_lite':
        from yolost.model_bmvit_lite import YOLOST_BMViTLite
        model = YOLOST_BMViTLite(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            target_T=cfg['model'].get('target_T', [64, 32, 16]),
            clip_length=cfg['data']['clip_length'],
        ).to(device)
    elif model_type == 'mvit_bmvit':
        from yolost.model_mvit_bmvit import YOLOST_MViTBMViT
        model = YOLOST_MViTBMViT(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            clip_length=cfg['data']['clip_length'],
            backbone_frames=cfg['model'].get('backbone_frames', 16),
            freeze_backbone=cfg['model'].get('freeze_backbone', True),
            unfreeze_last_n_blocks=cfg['model'].get('unfreeze_last_n_blocks', 0),
            pretrained=cfg['model'].get('pretrained', True),
            stop_before_final_pool=cfg['model'].get('stop_before_final_pool', True),
            hidden_dim=cfg['model'].get('hidden_dim', 384),
            head_depth=cfg['model'].get('head_depth', 3),
            action_temporal_pool=cfg['model'].get('action_temporal_pool', True),
        ).to(device)
    elif model_type == 'pyramid':
        from yolost.model_pyramid import YOLOST_Pyramid
        model = YOLOST_Pyramid(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            target_T=cfg['model'].get('target_T', [32, 16, 8]),
            clip_length=cfg['data']['clip_length'],
        ).to(device)
    elif model_type == 'dinov3':
        from yolost.model_dinov3 import YOLOST_DINOv3
        model = YOLOST_DINOv3(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            model_id=cfg['model'].get('model_id', 'facebook/dinov3-vitb16-pretrain-lvd1689m'),
            freeze_backbone=cfg['model'].get('freeze_backbone', True),
            unfreeze_last_n_blocks=cfg['model'].get('unfreeze_last_n_blocks', 0),
            micro_batch=cfg['model'].get('micro_batch', 8),
            dtype=cfg['model'].get('dtype', 'float16'),
            layer=cfg['model'].get('layer', -1),
            temporal_adapter=cfg['model'].get('temporal_adapter', False),
            temporal_adapter_depth=cfg['model'].get('temporal_adapter_depth', 2),
            temporal_adapter_kernel=cfg['model'].get('temporal_adapter_kernel', 5),
            temporal_adapter_expansion=cfg['model'].get('temporal_adapter_expansion', 2),
            temporal_adapter_dropout=cfg['model'].get('temporal_adapter_dropout', 0.0),
            apt_pyramid_adapter=cfg['model'].get('apt_pyramid_adapter', False),
            apt_adapter_depth=cfg['model'].get('apt_adapter_depth', 1),
            apt_adapter_kernel=cfg['model'].get('apt_adapter_kernel', 3),
            apt_adapter_temporal_kernel=cfg['model'].get('apt_adapter_temporal_kernel', 3),
            apt_adapter_dropout=cfg['model'].get('apt_adapter_dropout', 0.0),
            apt_context_gate=cfg['model'].get('apt_context_gate', False),
            apt_context_dim=cfg['model'].get('apt_context_dim', 128),
            apt_context_dropout=cfg['model'].get('apt_context_dropout', 0.0),
        ).to(device)
    elif model_type == 'videomae':
        from yolost.model_videomae import YOLOST_VideoMAE
        model = YOLOST_VideoMAE(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            clip_length=cfg['data']['clip_length'],
            model_id=cfg['model'].get('model_id', 'MCG-NJU/videomae-base-finetuned-kinetics'),
            freeze_backbone=cfg['model'].get('freeze_backbone', True),
            unfreeze_last_n_blocks=cfg['model'].get('unfreeze_last_n_blocks', 0),
            backbone_frames=cfg['model'].get('backbone_frames', 16),
            backbone_frame_sampling=cfg['model'].get('backbone_frame_sampling', 'uniform'),
            backbone_sampling_temperature=cfg['model'].get('backbone_sampling_temperature', 0.5),
            backbone_sampling_blend=cfg['model'].get('backbone_sampling_blend', 0.5),
            backbone_sampling_hidden=cfg['model'].get('backbone_sampling_hidden', 8),
            dtype=cfg['model'].get('dtype', 'float16'),
            temporal_adapter=cfg['model'].get('temporal_adapter', True),
            temporal_adapter_depth=cfg['model'].get('temporal_adapter_depth', 2),
            temporal_adapter_kernel=cfg['model'].get('temporal_adapter_kernel', 5),
            temporal_adapter_expansion=cfg['model'].get('temporal_adapter_expansion', 2),
            temporal_adapter_dropout=cfg['model'].get('temporal_adapter_dropout', 0.0),
            spatial_query_adapter=cfg['model'].get('spatial_query_adapter', False),
            spatial_query_grid=cfg['model'].get('spatial_query_grid', 7),
            spatial_query_heads=cfg['model'].get('spatial_query_heads', 8),
            spatial_query_dropout=cfg['model'].get('spatial_query_dropout', 0.1),
            backbone_gradient_checkpointing=cfg['model'].get(
                'backbone_gradient_checkpointing', False
            ),
            keyframe_pyramid=cfg['model'].get('keyframe_pyramid', False),
            keyframe_version=cfg['model'].get('keyframe_version', 'n'),
            keyframe_pretrained_path=cfg['model'].get('keyframe_pretrained_path'),
            keyframe_freeze=cfg['model'].get('keyframe_freeze', True),
            keyframe_bn_eval=cfg['model'].get('keyframe_bn_eval', True),
            keyframe_grad_checkpoint=cfg['model'].get('keyframe_grad_checkpoint', False),
            class_ctx_grad_checkpoint=cfg['model'].get('class_ctx_grad_checkpoint', False),
            keyframe_frames=cfg['model'].get('keyframe_frames', 16),
            keyframe_micro_batch=cfg['model'].get('keyframe_micro_batch', 16),
            keyframe_scales=cfg['model'].get('keyframe_scales', [True, True, True]),
            apt_pyramid_adapter=cfg['model'].get('apt_pyramid_adapter', False),
            apt_adapter_depth=cfg['model'].get('apt_adapter_depth', 1),
            apt_adapter_kernel=cfg['model'].get('apt_adapter_kernel', 3),
            apt_adapter_temporal_kernel=cfg['model'].get('apt_adapter_temporal_kernel', 3),
            apt_adapter_dropout=cfg['model'].get('apt_adapter_dropout', 0.0),
            apt_dilated_state=cfg['model'].get('apt_dilated_state', False),
            apt_dilated_state_scales=cfg['model'].get('apt_dilated_state_scales', [False, True, True]),
            apt_dilated_state_dilations=cfg['model'].get('apt_dilated_state_dilations', [1, 2, 4, 8]),
            apt_dilated_state_dropout=cfg['model'].get('apt_dilated_state_dropout', 0.0),
            apt_context_gate=cfg['model'].get('apt_context_gate', False),
            apt_context_dim=cfg['model'].get('apt_context_dim', 128),
            apt_context_dropout=cfg['model'].get('apt_context_dropout', 0.0),
            apt_actor_context=cfg['model'].get('apt_actor_context', False),
            apt_actor_context_dim=cfg['model'].get('apt_actor_context_dim', 128),
            apt_actor_context_heads=cfg['model'].get('apt_actor_context_heads', 4),
            apt_actor_context_depth=cfg['model'].get('apt_actor_context_depth', 1),
            apt_actor_context_slots=cfg['model'].get('apt_actor_context_slots', 1),
            apt_actor_context_dropout=cfg['model'].get('apt_actor_context_dropout', 0.0),
            apt_class_context=cfg['model'].get('apt_class_context', False),
            apt_class_context_dim=cfg['model'].get('apt_class_context_dim', 128),
            apt_class_context_dropout=cfg['model'].get('apt_class_context_dropout', 0.0),
            apt_trajectory_align=cfg['model'].get('apt_trajectory_align', False),
            apt_trajectory_hidden=cfg['model'].get('apt_trajectory_hidden', 32),
            apt_trajectory_radius=cfg['model'].get('apt_trajectory_radius', 2),
            apt_trajectory_temperature=cfg['model'].get('apt_trajectory_temperature', 0.07),
            apt_trajectory_scales=cfg['model'].get('apt_trajectory_scales', [True, False, False]),
            apt_tube_denoising=cfg['model'].get('apt_tube_denoising', False),
            apt_tube_denoising_dim=cfg['model'].get('apt_tube_denoising_dim', 128),
            apt_tube_denoising_heads=cfg['model'].get('apt_tube_denoising_heads', 4),
            apt_tube_denoising_depth=cfg['model'].get('apt_tube_denoising_depth', 2),
            apt_tube_denoising_max_tubes=cfg['model'].get('apt_tube_denoising_max_tubes', 8),
            apt_tube_denoising_box_noise=cfg['model'].get('apt_tube_denoising_box_noise', 0.1),
            apt_tube_denoising_label_noise=cfg['model'].get('apt_tube_denoising_label_noise', 0.2),
            apt_cross_clip_memory=cfg['model'].get('apt_cross_clip_memory', False),
            apt_cross_clip_memory_dim=cfg['model'].get('apt_cross_clip_memory_dim', 128),
            apt_cross_clip_memory_chunk=cfg['model'].get('apt_cross_clip_memory_chunk', 16),
            apt_cross_clip_memory_bidirectional=cfg['model'].get('apt_cross_clip_memory_bidirectional', True),
            apt_cross_clip_memory_dropout=cfg['model'].get('apt_cross_clip_memory_dropout', 0.0),
            apt_cross_clip_memory_target=cfg['model'].get(
                'apt_cross_clip_memory_target', 'features'
            ),
            apt_cross_clip_decision_target=cfg['model'].get(
                'apt_cross_clip_decision_target', 'class_boundary'
            ),
            apt_pyramid_actor_memory=cfg['model'].get(
                'apt_pyramid_actor_memory', False
            ),
            apt_pyramid_actor_memory_dim=cfg['model'].get(
                'apt_pyramid_actor_memory_dim', 128
            ),
            apt_pyramid_actor_memory_chunk=cfg['model'].get(
                'apt_pyramid_actor_memory_chunk', 8
            ),
            apt_pyramid_actor_memory_bidirectional=cfg['model'].get(
                'apt_pyramid_actor_memory_bidirectional', True
            ),
            apt_pyramid_actor_memory_dropout=cfg['model'].get(
                'apt_pyramid_actor_memory_dropout', 0.0
            ),
            apt_pyramid_actor_memory_learned_routing=cfg['model'].get(
                'apt_pyramid_actor_memory_learned_routing', True
            ),
            apt_tube_queries=cfg['model'].get('apt_tube_queries', False),
            apt_tube_query_dim=cfg['model'].get('apt_tube_query_dim', 256),
            apt_tube_query_count=cfg['model'].get('apt_tube_query_count', 8),
            apt_tube_query_heads=cfg['model'].get('apt_tube_query_heads', 8),
            apt_tube_query_depth=cfg['model'].get('apt_tube_query_depth', 2),
            apt_tube_query_dropout=cfg['model'].get('apt_tube_query_dropout', 0.1),
            apt_tube_query_actor_aligned=cfg['model'].get('apt_tube_query_actor_aligned', False),
            apt_tube_query_factorized=cfg['model'].get('apt_tube_query_factorized', False),
            apt_tube_query_frames=cfg['model'].get('apt_tube_query_frames', 32),
            apt_tube_query_memory_grid=cfg['model'].get('apt_tube_query_memory_grid', 14),
            apt_tube_query_boundary_gate=cfg['model'].get('apt_tube_query_boundary_gate', False),
            apt_tube_query_iterative_refinement=cfg['model'].get('apt_tube_query_iterative_refinement', False),
            apt_tube_query_trajectory_sampling=cfg['model'].get('apt_tube_query_trajectory_sampling', False),
            apt_tube_query_trajectory_points=cfg['model'].get('apt_tube_query_trajectory_points', 5),
            apt_tube_query_intervals=cfg['model'].get('apt_tube_query_intervals', False),
            apt_tube_query_instances_per_actor=cfg['model'].get('apt_tube_query_instances_per_actor', 1),
            apt_tube_query_interval_pyramid=cfg['model'].get('apt_tube_query_interval_pyramid', False),
            apt_tube_query_drop_path_rate=cfg['model'].get('apt_tube_query_drop_path_rate', 0.0),
            apt_tube_query_class_prior_probability=cfg['model'].get('apt_tube_query_class_prior_probability'),
            apt_tube_query_quality=cfg['model'].get('apt_tube_query_quality', False),
            apt_tube_query_boundary_distance=cfg['model'].get(
                'apt_tube_query_boundary_distance', False
            ),
            apt_tube_query_boundary_distance_temperature=cfg['model'].get(
                'apt_tube_query_boundary_distance_temperature', 0.08
            ),
            apt_tube_query_duration_router=cfg['model'].get('apt_tube_query_duration_router', False),
            apt_tube_query_duration_kernels=cfg['model'].get('apt_tube_query_duration_kernels', [3, 7, 15]),
            apt_tube_query_change_point_pyramid=cfg['model'].get('apt_tube_query_change_point_pyramid', False),
            apt_tube_query_change_point_dilations=cfg['model'].get('apt_tube_query_change_point_dilations', [1, 2, 4, 8]),
            apt_tube_query_change_point_router_mode=cfg['model'].get('apt_tube_query_change_point_router_mode', 'actor_duration'),
            apt_tube_query_change_point_shared_projection=cfg['model'].get('apt_tube_query_change_point_shared_projection', False),
            apt_tube_query_identity_transport=cfg['model'].get('apt_tube_query_identity_transport', False),
            apt_tube_query_transport_proposals=cfg['model'].get('apt_tube_query_transport_proposals', 16),
            apt_tube_query_transport_sinkhorn_iterations=cfg['model'].get('apt_tube_query_transport_sinkhorn_iterations', 4),
            apt_tube_query_transport_temperature=cfg['model'].get('apt_tube_query_transport_temperature', 0.2),
            apt_tube_query_action_reset_state=cfg['model'].get('apt_tube_query_action_reset_state', False),
            apt_tube_query_action_fork_state=cfg['model'].get('apt_tube_query_action_fork_state', False),
            apt_tube_query_action_fork_mode=cfg['model'].get('apt_tube_query_action_fork_mode', 'state_boundary'),
            apt_tube_query_action_fork_temperature=cfg['model'].get('apt_tube_query_action_fork_temperature', 0.5),
            apt_tube_query_shared_grad_scale=cfg['model'].get('apt_tube_query_shared_grad_scale', 1.0),
            apt_tube_query_shared_grad_scales=cfg['model'].get('apt_tube_query_shared_grad_scales'),
            apt_tube_query_task_adapters=cfg['model'].get('apt_tube_query_task_adapters', False),
            apt_tube_query_task_adapter_scales=cfg['model'].get('apt_tube_query_task_adapter_scales', [True, False, True]),
            apt_tube_query_task_adapter_ratio=cfg['model'].get('apt_tube_query_task_adapter_ratio', 0.25),
            apt_tube_query_task_adapter_temporal_kernel=cfg['model'].get('apt_tube_query_task_adapter_temporal_kernel', 3),
            apt_tube_query_feedback=cfg['model'].get('apt_tube_query_feedback', True),
            apt_tube_query_deformable_points=cfg['model'].get('apt_tube_query_deformable_points', 1),
            apt_tube_query_boundary_recurrent=cfg['model'].get('apt_tube_query_boundary_recurrent', False),
            apt_tube_query_sparse_refine=cfg['model'].get('apt_tube_query_sparse_refine', False),
            apt_sparse_refine_cls=cfg['model'].get('apt_sparse_refine_cls', True),
            apt_sparse_refine_box=cfg['model'].get('apt_sparse_refine_box', True),
            apt_sparse_refine_obj=cfg['model'].get('apt_sparse_refine_obj', True),
            apt_dense_residual_adapter=cfg['model'].get('apt_dense_residual_adapter', False),
            apt_dense_residual_hidden_ratio=cfg['model'].get('apt_dense_residual_hidden_ratio', 0.125),
            apt_dense_residual_temporal_kernels=cfg['model'].get('apt_dense_residual_temporal_kernels', [3, 3, 3]),
            apt_dense_residual_scales=cfg['model'].get('apt_dense_residual_scales', [True, True, True]),
            apt_dense_residual_class=cfg['model'].get('apt_dense_residual_class', True),
            apt_dense_residual_box=cfg['model'].get('apt_dense_residual_box', True),
            apt_dense_residual_object=cfg['model'].get('apt_dense_residual_object', True),
            apt_query_trajectory_residual_adapter=cfg['model'].get(
                'apt_query_trajectory_residual_adapter', False
            ),
            apt_query_trajectory_residual_hidden=cfg['model'].get(
                'apt_query_trajectory_residual_hidden', 64
            ),
            apt_query_trajectory_residual_kernels=cfg['model'].get(
                'apt_query_trajectory_residual_kernels', [3, 7, 15, 31]
            ),
            apt_query_trajectory_residual_max_box_delta=cfg['model'].get(
                'apt_query_trajectory_residual_max_box_delta', 0.05
            ),
            apt_query_trajectory_residual_class=cfg['model'].get(
                'apt_query_trajectory_residual_class', True
            ),
            apt_query_trajectory_residual_box=cfg['model'].get(
                'apt_query_trajectory_residual_box', True
            ),
            apt_query_trajectory_residual_visibility=cfg['model'].get(
                'apt_query_trajectory_residual_visibility', True
            ),
            apt_query_trajectory_residual_boundary=cfg['model'].get(
                'apt_query_trajectory_residual_boundary', True
            ),
            apt_query_trajectory_residual_endpoints=cfg['model'].get(
                'apt_query_trajectory_residual_endpoints', True
            ),
            apt_dense_tube_geometry_contract=cfg['model'].get(
                'apt_dense_tube_geometry_contract', False
            ),
            apt_dense_tube_geometry_hidden=cfg['model'].get(
                'apt_dense_tube_geometry_hidden', 64
            ),
            apt_dense_tube_geometry_kernels=cfg['model'].get(
                'apt_dense_tube_geometry_kernels', [3, 7, 15, 31]
            ),
            apt_dense_tube_geometry_proposals=cfg['model'].get(
                'apt_dense_tube_geometry_proposals', 16
            ),
            apt_dense_tube_geometry_match_temperature=cfg['model'].get(
                'apt_dense_tube_geometry_match_temperature', 0.2
            ),
            apt_dense_tube_geometry_max_blend=cfg['model'].get(
                'apt_dense_tube_geometry_max_blend', 0.5
            ),
            apt_dense_tube_geometry_smooth_corrections=cfg['model'].get(
                'apt_dense_tube_geometry_smooth_corrections', False
            ),
            native_motion_pyramid=cfg['model'].get('native_motion_pyramid', False),
            native_motion_init=cfg['model'].get('native_motion_init'),
            native_motion_freeze=cfg['model'].get('native_motion_freeze', False),
            native_motion_actor_local=cfg['model'].get('native_motion_actor_local', False),
            native_motion_actor_floor=cfg['model'].get('native_motion_actor_floor', 0.1),
            native_motion_channel_ratio=cfg['model'].get('native_motion_channel_ratio', 1.0),
            native_motion_ablation_disabled=cfg['model'].get('native_motion_ablation_disabled', False),
            position_embedding_mode=cfg['model'].get('position_embedding_mode', 'interpolate'),
            reg_max=cfg['model'].get('reg_max', 0),
        ).to(device)
    else:
        model = YOLOST(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
        ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f'Model parameters: {num_params:.1f}M', rank)

    if world_size > 1:
        ddp_find_unused_parameters = bool(
            cfg['train'].get('ddp_find_unused_parameters', False)
        )
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=ddp_find_unused_parameters,
        )
        log(
            f'DDP find_unused_parameters={ddp_find_unused_parameters}',
            rank,
        )

    raw_model = model.module if world_size > 1 else model

    # Dataset
    use_boundary = model_type in ('phase3', 'phase3_temporal_context', 'dinov3', 'videomae')
    if args.synthetic:
        train_dataset = SyntheticActionDataset(
            num_samples=200,
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
            num_classes=cfg['model']['num_classes'],
        )
        val_dataset = SyntheticActionDataset(
            num_samples=50,
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
            num_classes=cfg['model']['num_classes'],
        )
    dataset_name = cfg['data'].get('dataset', 'ucf101-24')
    augment_scale_range = cfg['data'].get('augment_scale_range', [0.5, 1.5])
    offline_trajectory_cfg = cfg.get('offline_trajectory_distillation', {})
    offline_trajectory_enabled = bool(
        offline_trajectory_cfg.get('enabled', False)
    )
    offline_dense_cfg = cfg.get(
        'offline_dense_pyramid_distillation', {}
    )
    offline_dense_enabled = bool(offline_dense_cfg.get('enabled', False))
    if (
        (offline_trajectory_enabled or offline_dense_enabled)
        and dataset_name != 'ucf101-24'
    ):
        raise ValueError(
            'Offline trajectory supervision currently requires UCF101-24'
        )
    if offline_trajectory_enabled and offline_dense_enabled:
        trajectory_track_dir = offline_trajectory_cfg.get('track_dir')
        dense_track_dir = offline_dense_cfg.get('track_dir')
        if trajectory_track_dir != dense_track_dir:
            raise ValueError(
                'Concurrent offline losses must use the same track_dir'
            )
    offline_data_cfg = (
        offline_dense_cfg if offline_dense_enabled
        else offline_trajectory_cfg
    )
    offline_data_enabled = (
        offline_trajectory_enabled or offline_dense_enabled
    )
    if False:
        pass
    elif dataset_name == 'ava':
        ava_clip_kwargs = dict(
            keyframe_position=cfg['data'].get('keyframe_position', 'end'),
            frame_stride=cfg['data'].get('frame_stride'),
            supervised_frames=cfg['data'].get('supervised_frames', False),
            uint8_clips=cfg['data'].get('uint8_clips', False),
        )
        train_dataset = AVADataset(
            frames_root=cfg['data']['frames_root'],
            annot_file=cfg['data']['train_annot'],
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
            split='train',
            augment=True,
            multi_label=cfg['data'].get('multi_label', True),
            max_samples=cfg['data'].get('max_train_samples'),
            augment_scale_range=augment_scale_range,
            **ava_clip_kwargs,
        )
        if cfg['data'].get('kinetics_frames_root'):
            from torch.utils.data import ConcatDataset
            from data.ava_kinetics_dataset import AVAKineticsDataset
            kinetics_dataset = AVAKineticsDataset(
                frames_root=cfg['data']['kinetics_frames_root'],
                manifest_dir=cfg['data']['kinetics_manifest_dir'],
                annot_file=cfg['data']['kinetics_annot'],
                clip_length=cfg['data']['clip_length'],
                img_size=cfg['data']['img_size'],
                split='train',
                augment=True,
                max_samples=cfg['data'].get('max_kinetics_samples'),
                augment_scale_range=augment_scale_range,
                uint8_clips=cfg['data'].get('uint8_clips', False),
                keyframe_position=ava_clip_kwargs['keyframe_position'],
                frame_stride=ava_clip_kwargs['frame_stride'],
                supervised_frames=ava_clip_kwargs['supervised_frames'],
            )
            log(f'AVA + AVA-Kinetics training: {len(train_dataset)} + '
                f'{len(kinetics_dataset)} keyframes', rank)
            train_dataset = ConcatDataset([train_dataset, kinetics_dataset])
        val_dataset = AVADataset(
            frames_root=cfg['data']['frames_root'],
            annot_file=cfg['data']['val_annot'],
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
            split='val',
            augment=False,
            multi_label=cfg['data'].get('multi_label', True),
            max_samples=cfg['data'].get('max_val_samples'),
            **ava_clip_kwargs,
        )
    elif dataset_name == 'multisports':
        train_dataset = MultiSportsDataset(
            root=cfg['data']['root'],
            annot_file=cfg['data']['annot_file'],
            clip_length=cfg['data']['clip_length'],
            stride=cfg['data'].get('stride', 1),
            split='train',
            img_size=cfg['data']['img_size'],
            augment=True,
            boundary_labels=use_boundary,
            filter_empty_clips=cfg['data'].get('filter_empty_clips', True),
            augment_scale_range=augment_scale_range,
        )
        val_dataset = MultiSportsDataset(
            root=cfg['data']['root'],
            annot_file=cfg['data']['annot_file'],
            clip_length=cfg['data']['clip_length'],
            stride=cfg['data'].get('stride', 1),
            split='test',
            img_size=cfg['data']['img_size'],
            augment=False,
            boundary_labels=use_boundary,
        )
    elif dataset_name == 'ava_kinetics_weak':
        train_dataset = AVAKineticsWeakDataset(
            frames_root=cfg['data']['frames_root'],
            annot_file=cfg['data']['train_annot'],
            weak_label_file=cfg['data']['weak_label_file'],
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
            split='train',
            augment=True,
            multi_label=True,
            num_classes=cfg['model']['num_classes'],
            augment_scale_range=augment_scale_range,
        )
        val_dataset = AVAKineticsWeakDataset(
            frames_root=cfg['data']['frames_root'],
            annot_file=cfg['data']['val_annot'],
            weak_label_file=cfg['data']['weak_label_file'],
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
            split='val',
            augment=False,
            multi_label=True,
            num_classes=cfg['model']['num_classes'],
        )
    elif dataset_name == 'jhmdb':
        jhmdb_common = dict(
            root=cfg['data']['root'],
            annot_file=cfg['data']['annot_file'],
            clip_length=cfg['data']['clip_length'],
            stride=cfg['data'].get('stride', 1),
            img_size=cfg['data']['img_size'],
            boundary_labels=use_boundary,
            split_index=int(cfg['data'].get('split_index', 0)),
        )
        train_dataset = JHMDBDataset(
            split='train',
            augment=True,
            filter_empty_clips=cfg['data'].get('filter_empty_clips', False),
            empty_clip_repeats=cfg['data'].get('empty_clip_repeats', 1),
            augment_scale_range=augment_scale_range,
            **jhmdb_common,
        )
        val_dataset = JHMDBDataset(split='test', augment=False, **jhmdb_common)
        log(
            f"JHMDB-21 split_index={jhmdb_common['split_index']}: "
            f'{len(train_dataset.clips)} train clips, '
            f'whole-video clips={train_dataset.whole_video_clips}',
            rank,
        )
    else:
        if dataset_name not in ('ucf101-24', 'synthetic'):
            raise ValueError(f'unknown data.dataset {dataset_name!r}')
        train_dataset = UCF101_24_Dataset(
            temporal_jitter=cfg['data'].get('temporal_jitter', 0),
            root=cfg['data']['root'],
            annot_file=cfg['data']['annot_file'],
            clip_length=cfg['data']['clip_length'],
            stride=cfg['data']['stride'],
            split='train',
            img_size=cfg['data']['img_size'],
            augment=True,
            boundary_labels=use_boundary,
            filter_empty_clips=cfg['data'].get('filter_empty_clips', False),
            empty_clip_repeats=cfg['data'].get('empty_clip_repeats', 1),
            boundary_negative_window=cfg['data'].get('boundary_negative_window', 0),
            boundary_negative_weight=cfg['data'].get('boundary_negative_weight', 1.0),
            augment_scale_range=augment_scale_range,
            offline_track_dir=(
                offline_data_cfg.get('track_dir')
                if offline_data_enabled else None
            ),
            offline_min_quality=offline_data_cfg.get(
                'min_track_quality', 0.15
            ),
            offline_min_score=offline_data_cfg.get(
                'min_observation_score', 0.03
            ),
            offline_min_clip_observations=offline_data_cfg.get(
                'min_clip_observations', 4
            ),
            offline_max_tracks=offline_data_cfg.get(
                'max_tracks', 16
            ),
        )
        val_dataset = UCF101_24_Dataset(
            root=cfg['data']['root'],
            annot_file=cfg['data']['annot_file'],
            clip_length=cfg['data']['clip_length'],
            stride=cfg['data']['stride'],
            split='test',
            img_size=cfg['data']['img_size'],
            augment=False,
            boundary_labels=use_boundary,
        )

    train_sampler = DistributedSampler(
        train_dataset, seed=base_seed
    ) if world_size > 1 else None
    loader_generator = torch.Generator()
    loader_generator.manual_seed(process_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    # Loss
    if model_type in ('phase3', 'phase3_temporal_context', 'dinov3', 'videomae'):
        from yolost.loss_boundary import YOLOSTLossBoundary
        criterion = YOLOSTLossBoundary(
            num_classes=cfg['model']['num_classes'],
            lambda_cls=cfg['loss']['lambda_cls'],
            lambda_box=cfg['loss']['lambda_box'],
            lambda_obj=cfg['loss']['lambda_obj'],
            lambda_bnd=cfg['loss'].get('lambda_bnd', 0.5),
            lambda_tube_cls=cfg['loss'].get('lambda_tube_cls', 0.0),
            lambda_tube_box=cfg['loss'].get('lambda_tube_box', 0.0),
            tube_obj_power=cfg['loss'].get('tube_obj_power', 0.5),
            lambda_dn_cls=cfg['loss'].get('lambda_dn_cls', 0.0),
            lambda_dn_box=cfg['loss'].get('lambda_dn_box', 0.0),
            lambda_memory=cfg['loss'].get('lambda_memory', 0.0),
            lambda_query_cls=cfg['loss'].get('lambda_query_cls', 0.0),
            lambda_query_box=cfg['loss'].get('lambda_query_box', 0.0),
            lambda_query_giou=cfg['loss'].get('lambda_query_giou', 0.0),
            lambda_query_visibility=cfg['loss'].get('lambda_query_visibility', 0.0),
            lambda_query_boundary=cfg['loss'].get('lambda_query_boundary', 0.0),
            lambda_query_velocity=cfg['loss'].get('lambda_query_velocity', 0.0),
            lambda_query_acceleration=cfg['loss'].get('lambda_query_acceleration', 0.0),
            lambda_query_start=cfg['loss'].get('lambda_query_start', 0.0),
            lambda_query_end=cfg['loss'].get('lambda_query_end', 0.0),
            lambda_query_interval_iou=cfg['loss'].get('lambda_query_interval_iou', 0.0),
            lambda_query_coverage=cfg['loss'].get('lambda_query_coverage', 0.0),
            lambda_query_fragmentation=cfg['loss'].get('lambda_query_fragmentation', 0.0),
            lambda_query_boundary_distance=cfg['loss'].get(
                'lambda_query_boundary_distance', 0.0
            ),
            lambda_query_boundary_distance_slope=cfg['loss'].get(
                'lambda_query_boundary_distance_slope', 0.0
            ),
            lambda_query_quality=cfg['loss'].get('lambda_query_quality', 0.0),
            lambda_query_transport=cfg['loss'].get('lambda_query_transport', 0.0),
            lambda_query_geometry_preservation=cfg['loss'].get(
                'lambda_query_geometry_preservation', 0.0
            ),
            query_cost_cls=cfg['loss'].get('query_cost_cls', 2.0),
            query_cost_box=cfg['loss'].get('query_cost_box', 5.0),
            query_cost_giou=cfg['loss'].get('query_cost_giou', 2.0),
            query_cost_visibility=cfg['loss'].get('query_cost_visibility', 1.0),
            query_cost_interval=cfg['loss'].get('query_cost_interval', 0.0),
            query_cost_coverage=cfg['loss'].get('query_cost_coverage', 0.0),
            query_cost_fragmentation=cfg['loss'].get('query_cost_fragmentation', 0.0),
            query_cost_transport=cfg['loss'].get('query_cost_transport', 0.0),
            query_boundary_pos_weight=cfg['loss'].get(
                'query_boundary_pos_weight', 1.0
            ),
            query_boundary_focal_gamma=cfg['loss'].get(
                'query_boundary_focal_gamma', 0.0
            ),
            query_class_focal_alpha=cfg['loss'].get('query_class_focal_alpha'),
            query_class_focal_gamma=cfg['loss'].get('query_class_focal_gamma', 2.0),
            query_quality_target_mode=cfg['loss'].get(
                'query_quality_target_mode', 'sqrt_product'
            ),
            query_quality_strict_blend=cfg['loss'].get(
                'query_quality_strict_blend', 0.5
            ),
            clip_length=cfg['data']['clip_length'],
            img_size=cfg['data']['img_size'],
        )
    elif model_type in ('bmvit_lite', 'mvit_bmvit'):
        from yolost.loss_bipartite import YOLOSTBipartiteLoss
        criterion = YOLOSTBipartiteLoss(
            num_classes=cfg['model']['num_classes'],
            lambda_cls=cfg['loss']['lambda_cls'],
            lambda_box=cfg['loss']['lambda_box'],
            lambda_giou=cfg['loss'].get('lambda_giou', 2.0),
            lambda_obj=cfg['loss']['lambda_obj'],
            cost_cls=cfg['loss'].get('cost_cls', 2.0),
            cost_box=cfg['loss'].get('cost_box', 5.0),
            cost_giou=cfg['loss'].get('cost_giou', 2.0),
            cost_obj=cfg['loss'].get('cost_obj', 1.0),
            no_object_weight=cfg['loss'].get('no_object_weight', 0.05),
            img_size=cfg['data']['img_size'],
        )
    else:
        criterion = YOLOSTLoss(
            num_classes=cfg['model']['num_classes'],
            lambda_cls=cfg['loss']['lambda_cls'],
            lambda_box=cfg['loss']['lambda_box'],
            lambda_obj=cfg['loss']['lambda_obj'],
            img_size=cfg['data']['img_size'],
        )

    conflict_cfg = cfg['train'].get('gradient_conflict_diagnostic', {})
    if isinstance(conflict_cfg, bool):
        conflict_cfg = {'enabled': conflict_cfg}
    conflict_enabled = bool(conflict_cfg.get('enabled', False))
    conflict_capture = {}
    conflict_records = []
    conflict_hook = None
    conflict_interval = max(1, int(conflict_cfg.get('interval', 1)))
    conflict_max_measurements = max(
        1, int(conflict_cfg.get('max_measurements', 8))
    )
    conflict_report_path = conflict_cfg.get(
        'report_path', os.path.join(exp_dir, 'gradient_conflict_report.json')
    )

    def write_conflict_report():
        if not is_main_process(rank):
            return
        report_dir = os.path.dirname(conflict_report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(conflict_report_path, 'w', encoding='utf-8') as report_file:
            json.dump({
                'config': args.config,
                'checkpoint': args.init or args.resume,
                'objective_partition': {
                    'frame': ['dense_cls', 'dense_box', 'dense_objectness'],
                    'tube_boundary': [
                        'dense_boundary', 'dense_tube_consistency',
                        'factorized_query_objectives',
                    ],
                },
                'records': conflict_records,
                'summary': summarize_gradient_conflicts(conflict_records),
            }, report_file, indent=2)

    # Weight of the distribution-regression term. Inert unless the head emits
    # more than four regression channels, i.e. unless model.reg_max > 0.
    criterion.lambda_dfl = float(cfg['loss'].get('lambda_dfl', 1.5))
    if getattr(raw_model, 'reg_max', 0):
        log(
            f'Distribution box regression: reg_max={raw_model.reg_max}, '
            f'lambda_dfl={criterion.lambda_dfl:g}',
            rank,
        )

    if conflict_enabled:
        if world_size != 1:
            raise ValueError('gradient conflict diagnostics require one process')
        if not isinstance(criterion, YOLOSTLossBoundary):
            raise ValueError('gradient conflict diagnostics require boundary loss')

        def capture_shared_features(_module, module_inputs):
            conflict_capture['features'] = tuple(module_inputs[0])

        conflict_hook = raw_model.head.register_forward_pre_hook(
            capture_shared_features
        )
        log(
            'Gradient-conflict diagnostic enabled: '
            f'{conflict_max_measurements} measurements every '
            f'{conflict_interval} step(s), report={conflict_report_path}',
            rank,
        )

    trainable_prefixes = cfg['train'].get('trainable_parameter_prefixes')
    if trainable_prefixes:
        if isinstance(trainable_prefixes, str):
            trainable_prefixes = [trainable_prefixes]
        trainable_prefixes = tuple(str(value) for value in trainable_prefixes)
        matched_names = []
        for name, parameter in raw_model.named_parameters():
            trainable = name.startswith(trainable_prefixes)
            parameter.requires_grad_(trainable)
            if trainable:
                matched_names.append(name)
        if not matched_names:
            raise ValueError(
                'trainable_parameter_prefixes matched no parameters: '
                f'{trainable_prefixes}'
            )
        trainable_count = sum(
            parameter.numel() for parameter in raw_model.parameters()
            if parameter.requires_grad
        )
        total_count = sum(parameter.numel() for parameter in raw_model.parameters())
        log(
            'Selective optimization: '
            f'{len(matched_names)} tensors, {trainable_count:,}/{total_count:,} '
            f'parameters, prefixes={trainable_prefixes}',
            rank,
        )

    freeze_all_batchnorm_stats = bool(
        cfg['train'].get('freeze_all_batchnorm_stats', False)
    )
    freeze_frozen_batchnorm_stats = bool(
        cfg['train'].get('freeze_frozen_batchnorm_stats', False)
    )
    if freeze_all_batchnorm_stats:
        locked_batchnorm_count = lock_all_batchnorm_stats(raw_model)
        log(
            f'All-layer BatchNorm-stat lock: {locked_batchnorm_count} layers',
            rank,
        )
    elif freeze_frozen_batchnorm_stats:
        locked_batchnorm_count = lock_frozen_batchnorm_stats(raw_model)
        log(
            f'Frozen-module BatchNorm lock: {locked_batchnorm_count} layers',
            rank,
        )

    # Optimizer
    head_lr_multiplier = float(cfg['train'].get('tube_query_lr_multiplier', 1.0))
    spatial_query_lr_multiplier = float(
        cfg['train'].get('spatial_query_lr_multiplier', 1.0)
    )
    keyframe_lr_multiplier = float(
        cfg['train'].get('keyframe_lr_multiplier', 1.0)
    )
    long_state_lr_multiplier = float(
        cfg['train'].get('long_state_lr_multiplier', 1.0)
    )
    motion_fusion_lr_multiplier = float(
        cfg['train'].get('motion_fusion_lr_multiplier', 1.0)
    )
    tube_task_adapter_lr_multiplier = float(
        cfg['train'].get('tube_task_adapter_lr_multiplier', 1.0)
    )
    backbone_lr_multiplier = float(
        cfg['train'].get('backbone_lr_multiplier', 1.0)
    )
    tube_query_head = getattr(raw_model, 'tube_query_head', None)
    spatial_query_adapter = getattr(raw_model, 'spatial_query_adapter', None)
    keyframe_fusions = getattr(raw_model, 'keyframe_fusions', None)
    tube_task_adapters = getattr(raw_model, 'tube_task_adapters', None)
    specialized_groups = []
    specialized_parameter_ids = set()
    if tube_query_head is not None and head_lr_multiplier != 1.0:
        head_parameters = list(tube_query_head.parameters())
        specialized_parameter_ids.update(id(parameter) for parameter in head_parameters)
        specialized_groups.append({
            'params': head_parameters,
            'lr': cfg['train']['lr'] * head_lr_multiplier,
        })
        log(f'Tube-query LR multiplier: {head_lr_multiplier:g}', rank)
    if spatial_query_adapter is not None and spatial_query_lr_multiplier != 1.0:
        spatial_query_parameters = list(spatial_query_adapter.parameters())
        specialized_parameter_ids.update(
            id(parameter) for parameter in spatial_query_parameters
        )
        specialized_groups.append({
            'params': spatial_query_parameters,
            'lr': cfg['train']['lr'] * spatial_query_lr_multiplier,
        })
        log(f'Spatial-query LR multiplier: {spatial_query_lr_multiplier:g}', rank)
    if keyframe_fusions is not None and len(keyframe_fusions) > 0 \
            and keyframe_lr_multiplier != 1.0:
        keyframe_parameters = [
            parameter for parameter in keyframe_fusions.parameters()
            if parameter.requires_grad
        ]
        specialized_parameter_ids.update(
            id(parameter) for parameter in keyframe_parameters
        )
        specialized_groups.append({
            'params': keyframe_parameters,
            'lr': cfg['train']['lr'] * keyframe_lr_multiplier,
        })
        log(f'Keyframe-fusion LR multiplier: {keyframe_lr_multiplier:g}', rank)
    if tube_task_adapters is not None and len(tube_task_adapters) > 0 \
            and tube_task_adapter_lr_multiplier != 1.0:
        task_adapter_parameters = [
            parameter for parameter in tube_task_adapters.parameters()
            if parameter.requires_grad
        ]
        specialized_parameter_ids.update(
            id(parameter) for parameter in task_adapter_parameters
        )
        specialized_groups.append({
            'params': task_adapter_parameters,
            'lr': cfg['train']['lr'] * tube_task_adapter_lr_multiplier,
        })
        log(
            'Tube-task-adapter LR multiplier: '
            f'{tube_task_adapter_lr_multiplier:g}', rank
        )
    state_modules = [
        getattr(raw_model, name, None)
        for name in ('state_p3', 'state_p4', 'state_p5')
    ]
    state_parameters = [
        parameter for module in state_modules if module is not None
        for parameter in module.parameters() if parameter.requires_grad
    ]
    if state_parameters and long_state_lr_multiplier != 1.0:
        specialized_parameter_ids.update(id(parameter) for parameter in state_parameters)
        specialized_groups.append({
            'params': state_parameters,
            'lr': cfg['train']['lr'] * long_state_lr_multiplier,
        })
        log(f'Long-state LR multiplier: {long_state_lr_multiplier:g}', rank)
    motion_fusions = [
        getattr(raw_model, name, None)
        for name in ('motion_fuse_p3', 'motion_fuse_p4', 'motion_fuse_p5')
    ]
    motion_fusion_parameters = [
        parameter for module in motion_fusions if module is not None
        for parameter in module.parameters() if parameter.requires_grad
    ]
    if motion_fusion_parameters and motion_fusion_lr_multiplier != 1.0:
        specialized_parameter_ids.update(
            id(parameter) for parameter in motion_fusion_parameters
        )
        specialized_groups.append({
            'params': motion_fusion_parameters,
            'lr': cfg['train']['lr'] * motion_fusion_lr_multiplier,
        })
        log(f'Motion-fusion LR multiplier: {motion_fusion_lr_multiplier:g}', rank)
    keyframe_backbone_lr_multiplier = float(
        cfg['train'].get('keyframe_backbone_lr_multiplier', 1.0)
    )
    keyframe_backbone = getattr(raw_model, 'keyframe_backbone', None)
    if keyframe_backbone is not None:
        keyframe_backbone_parameters = [
            parameter for parameter in keyframe_backbone.parameters()
            if parameter.requires_grad
            and id(parameter) not in specialized_parameter_ids
        ]
        log(
            f'Keyframe backbone: {len(keyframe_backbone_parameters)} trainable tensors, '
            f'LR multiplier {keyframe_backbone_lr_multiplier:g}',
            rank,
        )
        if keyframe_backbone_parameters and keyframe_backbone_lr_multiplier != 1.0:
            specialized_parameter_ids.update(
                id(parameter) for parameter in keyframe_backbone_parameters
            )
            specialized_groups.append({
                'params': keyframe_backbone_parameters,
                'lr': cfg['train']['lr'] * keyframe_backbone_lr_multiplier,
            })
    backbone_module = getattr(raw_model, 'backbone', None)
    if backbone_module is not None and backbone_lr_multiplier != 1.0:
        backbone_parameters = [
            parameter for parameter in backbone_module.parameters()
            if parameter.requires_grad
            and id(parameter) not in specialized_parameter_ids
        ]
        if backbone_parameters:
            specialized_parameter_ids.update(
                id(parameter) for parameter in backbone_parameters
            )
            specialized_groups.append({
                'params': backbone_parameters,
                'lr': cfg['train']['lr'] * backbone_lr_multiplier,
            })
            log(
                f'Backbone LR multiplier: {backbone_lr_multiplier:g} '
                f'({len(backbone_parameters)} tensors)',
                rank,
            )
    if specialized_groups:
        base_parameters = [
            parameter for parameter in raw_model.parameters()
            if parameter.requires_grad
            and id(parameter) not in specialized_parameter_ids
        ]
        parameter_groups = [
            {'params': base_parameters, 'lr': cfg['train']['lr']},
            *specialized_groups,
        ]
    else:
        parameter_groups = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )

    # Scheduler: cosine with warmup
    epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train']['warmup_epochs']
    steps_per_epoch = len(train_loader) // cfg['train'].get('grad_accum', 1)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    amp_dtype_name = str(cfg['train'].get('amp_dtype', 'float16')).lower()
    amp_dtype = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }.get(amp_dtype_name)
    if amp_dtype is None:
        raise ValueError(
            f"train.amp_dtype must be float16 or bfloat16, got {amp_dtype_name}"
        )
    # bfloat16 carries the float32 exponent range, so loss scaling is
    # unnecessary and the scaler is disabled for it.
    use_grad_scaler = (
        cfg['train']['mixed_precision'] and amp_dtype is torch.float16
    )
    scaler = GradScaler(
        enabled=use_grad_scaler,
        init_scale=float(cfg['train'].get('amp_init_scale', 65536.0)),
    )
    if cfg['train']['mixed_precision']:
        log(
            f'AMP dtype: {amp_dtype_name} '
            f'(GradScaler {"on" if use_grad_scaler else "off"})',
            rank,
        )
    # A NaN forward pass permanently poisons BatchNorm running statistics, so a
    # diverged run never recovers. Abort instead of burning GPU hours on it.
    nonfinite_abort_steps = int(
        cfg['train'].get('nonfinite_abort_steps', 25)
    )
    nonfinite_streak = 0
    nonfinite_reported = False

    # EMA
    use_ema = cfg['train'].get('ema', False)
    ema = None
    if use_ema:
        ema_decay = float(cfg['train'].get('ema_decay', 0.9999))
        ema_warmup = int(cfg['train'].get('ema_warmup', 2000))
        ema = ModelEMA(raw_model, decay=ema_decay, warmup=ema_warmup)
        log(f'EMA enabled (decay={ema_decay:g}, warmup={ema_warmup})', rank)

    # Model-only initialization supports adding a new head to an existing detector.
    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        model_state = ckpt.get('model', ckpt)
        current_state = raw_model.state_dict()
        compatible_state = {
            key: value for key, value in model_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        incompatible = raw_model.load_state_dict(compatible_state, strict=False)
        log(f'Initialized {len(compatible_state)} tensors from {args.init}', rank)
        log(f'Randomly initialized tensors: {len(incompatible.missing_keys)}', rank)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if all(k in ckpt for k in ('optimizer', 'scheduler', 'scaler')):
            raw_model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            start_epoch = ckpt['epoch'] + 1
            if use_ema and 'ema' in ckpt:
                ema.load_state_dict(ckpt['ema'])
                log(f'Resumed EMA (updates={ema.updates})', rank)
            log(f'Resumed from epoch {start_epoch}', rank)
        else:
            model_state = ckpt['model']
            current_state = raw_model.state_dict()
            skipped_shape_keys = []
            compatible_state = {}
            for key, value in model_state.items():
                if key in current_state and current_state[key].shape != value.shape:
                    skipped_shape_keys.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                    continue
                compatible_state[key] = value

            incompatible = raw_model.load_state_dict(compatible_state, strict=False)
            if skipped_shape_keys:
                log(f'Skipped checkpoint keys with incompatible shapes: {len(skipped_shape_keys)}', rank)
                for key, old_shape, new_shape in skipped_shape_keys[:10]:
                    log(f'  skipped: {key} {old_shape} -> {new_shape}', rank)
            if incompatible.missing_keys:
                log(f'Missing keys initialized randomly: {len(incompatible.missing_keys)}', rank)
                for key in incompatible.missing_keys[:10]:
                    log(f'  missing: {key}', rank)
            if incompatible.unexpected_keys:
                log(f'Unexpected checkpoint keys ignored: {len(incompatible.unexpected_keys)}', rank)
                for key in incompatible.unexpected_keys[:10]:
                    log(f'  unexpected: {key}', rank)
            log(f'Loaded compatible model weights from {args.resume} (model-only checkpoint)', rank)

    distillation_cfg = cfg.get('distillation', {})
    distillation_enabled = bool(distillation_cfg.get('enabled', False))
    distillation_criterion = None
    distillation_teachers = []
    distillation_lambda = float(distillation_cfg.get('lambda_total', 1.0))
    if distillation_enabled:
        if world_size != 1:
            raise ValueError('Online distillation currently requires one process')
        from eval_video_map import load_model
        from yolost.distillation import COMPONENTS, YOLOSTDistillationLoss

        teacher_specs = distillation_cfg.get('teachers', [])
        if not teacher_specs:
            raise ValueError('distillation.teachers must contain at least one teacher')
        for index, teacher_spec in enumerate(teacher_specs):
            teacher_config_path = teacher_spec['config']
            teacher_checkpoint = teacher_spec['checkpoint']
            teacher_cfg = load_config(teacher_config_path)
            teacher_model = load_model(
                teacher_cfg, teacher_checkpoint, device
            )
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad_(False)
            if hasattr(teacher_model, 'enable_tube_query_output'):
                teacher_model.enable_tube_query_output(True)
            if hasattr(teacher_model, 'enable_persistent_memory'):
                teacher_model.enable_persistent_memory(False)
            teacher_name = teacher_spec.get('name', f'teacher_{index}')
            distillation_teachers.append({
                'name': teacher_name,
                'model': teacher_model,
                'weight': float(teacher_spec.get('weight', 1.0)),
                'class_ids': teacher_spec.get('class_ids'),
                'component_weights': teacher_spec.get('component_weights'),
            })
            log(
                f'Distillation teacher {teacher_name}: '
                f'{teacher_checkpoint} weight={teacher_spec.get("weight", 1.0)} '
                f'class_ids={teacher_spec.get("class_ids")} '
                f'components={teacher_spec.get("component_weights", "all")}',
                rank,
            )
        component_weights = {
            name: distillation_cfg.get(name, 0.0) for name in COMPONENTS
        }
        distillation_criterion = YOLOSTDistillationLoss(
            component_weights=component_weights,
            temperature=distillation_cfg.get('temperature', 1.0),
            min_teacher_confidence=distillation_cfg.get(
                'min_teacher_confidence', 0.05
            ),
            max_teacher_queries=distillation_cfg.get(
                'max_teacher_queries', 16
            ),
            match_class_cost=distillation_cfg.get('match_class_cost', 2.0),
            match_box_cost=distillation_cfg.get('match_box_cost', 5.0),
            match_visibility_cost=distillation_cfg.get(
                'match_visibility_cost', 1.0
            ),
            dense_positive_floor=distillation_cfg.get(
                'dense_positive_floor', 0.05
            ),
        ).to(device)
        log(
            f'Online distillation enabled: teachers={len(distillation_teachers)} '
            f'lambda={distillation_lambda:g}',
            rank,
        )

    offline_trajectory_criterion = None
    offline_trajectory_lambda = float(
        offline_trajectory_cfg.get('lambda_total', 1.0)
    )
    if offline_trajectory_enabled:
        track_dir = offline_trajectory_cfg.get('track_dir')
        if not track_dir or not os.path.isdir(track_dir):
            raise FileNotFoundError(
                f'Offline trajectory track directory not found: {track_dir}'
            )
        from yolost.offline_trajectory_distillation import (
            OfflineTrajectoryDistillationLoss,
        )
        component_names = OfflineTrajectoryDistillationLoss.COMPONENTS
        offline_trajectory_criterion = OfflineTrajectoryDistillationLoss(
            component_weights={
                name: offline_trajectory_cfg.get(f'{name}_weight', 0.0)
                for name in component_names
            },
            cost_class=offline_trajectory_cfg.get('match_class_cost', 2.0),
            cost_box=offline_trajectory_cfg.get('match_box_cost', 5.0),
            cost_giou=offline_trajectory_cfg.get('match_giou_cost', 2.0),
            cost_visibility=offline_trajectory_cfg.get(
                'match_visibility_cost', 1.0
            ),
            cost_interval=offline_trajectory_cfg.get(
                'match_interval_cost', 0.0
            ),
            cost_coverage=offline_trajectory_cfg.get(
                'match_coverage_cost', 0.0
            ),
            cost_fragmentation=offline_trajectory_cfg.get(
                'match_fragmentation_cost', 0.0
            ),
            confidence_weighted=offline_trajectory_cfg.get(
                'confidence_weighted', False
            ),
            confidence_floor=offline_trajectory_cfg.get(
                'confidence_floor', 0.05
            ),
        ).to(device)
        log(
            'Offline full-video trajectory distillation enabled: '
            f'track_dir={track_dir} lambda={offline_trajectory_lambda:g} '
            f'confidence_weighted='
            f'{offline_trajectory_criterion.confidence_weighted}',
            rank,
        )

    offline_dense_criterion = None
    offline_dense_lambda = float(
        offline_dense_cfg.get('lambda_total', 1.0)
    )
    if offline_dense_enabled:
        track_dir = offline_dense_cfg.get('track_dir')
        if not track_dir or not os.path.isdir(track_dir):
            raise FileNotFoundError(
                f'Offline dense-pyramid track directory not found: {track_dir}'
            )
        from yolost.offline_dense_pyramid_distillation import (
            OfflineDensePyramidDistillationLoss,
        )
        offline_dense_criterion = OfflineDensePyramidDistillationLoss(
            num_classes=cfg['model']['num_classes'],
            class_weight=offline_dense_cfg.get('class_weight', 1.0),
            object_weight=offline_dense_cfg.get('object_weight', 0.25),
            box_weight=offline_dense_cfg.get('box_weight', 1.0),
            giou_weight=offline_dense_cfg.get('giou_weight', 1.0),
            velocity_weight=offline_dense_cfg.get('velocity_weight', 0.0),
            velocity_scale_weights=offline_dense_cfg.get(
                'velocity_scale_weights'
            ),
            velocity_rate_normalize=offline_dense_cfg.get(
                'velocity_rate_normalize', False
            ),
            confidence_floor=offline_dense_cfg.get(
                'confidence_floor', 0.05
            ),
            interpolated_weight=offline_dense_cfg.get(
                'interpolated_weight', 0.25
            ),
            top_k=offline_dense_cfg.get('top_k', 10),
        ).to(device)
        log(
            'Offline dense-pyramid trajectory distillation enabled: '
            f'track_dir={track_dir} lambda={offline_dense_lambda:g} '
            f'interpolated_weight='
            f'{offline_dense_criterion.interpolated_weight:g} '
            f'velocity_scale_weights='
            f'{offline_dense_cfg.get("velocity_scale_weights", "all")} '
            f'velocity_rate_normalize='
            f'{offline_dense_criterion.velocity_rate_normalize}',
            rank,
        )

    yowo_distillation_cfg = cfg.get('yowoformer_distillation', {})
    yowo_distillation_enabled = bool(
        yowo_distillation_cfg.get('enabled', False)
    )
    yowo_distillation_criterion = None
    yowo_distillation_teacher = None
    yowo_distillation_lambda = float(
        yowo_distillation_cfg.get('lambda_total', 1.0)
    )
    yowo_distillation_endpoints = []
    yowo_teacher_clip_length = 0
    yowo_views_per_step = 0
    yowo_distillation_interval = 1
    if yowo_distillation_enabled:
        if world_size != 1:
            raise ValueError(
                'YOWOFormer distillation currently requires one process'
            )
        import hashlib
        import importlib.util

        yowo_root = os.path.abspath(yowo_distillation_cfg['yowo_root'])
        yowo_checkpoint = os.path.abspath(
            yowo_distillation_cfg['checkpoint']
        )
        evaluate_path = os.path.join(yowo_root, 'evaluate.py')
        if not os.path.isfile(evaluate_path):
            raise FileNotFoundError(
                f'YOWOFormer evaluate.py not found: {evaluate_path}'
            )
        if not os.path.isfile(yowo_checkpoint):
            raise FileNotFoundError(
                f'YOWOFormer checkpoint not found: {yowo_checkpoint}'
            )
        expected_sha = yowo_distillation_cfg.get('expected_sha256')
        if expected_sha:
            digest = hashlib.sha256()
            with open(yowo_checkpoint, 'rb') as checkpoint_file:
                for chunk in iter(
                    lambda: checkpoint_file.read(16 * 1024 * 1024), b''
                ):
                    digest.update(chunk)
            actual_sha = digest.hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(
                    'YOWOFormer checkpoint SHA mismatch: '
                    f'expected={expected_sha} actual={actual_sha}'
                )
        else:
            actual_sha = 'unchecked'

        module_spec = importlib.util.spec_from_file_location(
            '_yolost_yowoformer_evaluate', evaluate_path
        )
        yowo_evaluate = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(yowo_evaluate)
        (
            yowo_distillation_teacher,
            yowo_teacher_config,
            yowo_teacher_info,
        ) = yowo_evaluate.load_model(yowo_checkpoint, device)
        yowo_distillation_teacher.eval()
        for parameter in yowo_distillation_teacher.parameters():
            parameter.requires_grad_(False)

        yowo_teacher_clip_length = int(yowo_teacher_config['clip_length'])
        if int(yowo_teacher_config['num_classes']) != int(
                cfg['model']['num_classes']):
            raise ValueError(
                'YOWOFormer and YOLO-ST class counts must match for '
                'pyramid-aligned distillation'
            )
        if int(yowo_teacher_config['img_size']) != int(cfg['data']['img_size']):
            raise ValueError(
                'YOWOFormer and YOLO-ST image sizes must match for '
                'pyramid-aligned distillation'
            )
        yowo_distillation_endpoints = [
            int(value) for value in yowo_distillation_cfg.get(
                'endpoints', [cfg['data']['clip_length'] - 1]
            )
        ]
        if not yowo_distillation_endpoints:
            raise ValueError('YOWOFormer distillation endpoints cannot be empty')
        if any(
            endpoint < 0 or endpoint >= int(cfg['data']['clip_length'])
            for endpoint in yowo_distillation_endpoints
        ):
            raise ValueError(
                'YOWOFormer distillation endpoints must lie inside the clip'
            )
        yowo_views_per_step = int(yowo_distillation_cfg.get(
            'views_per_step', len(yowo_distillation_endpoints)
        ))
        if not 1 <= yowo_views_per_step <= len(yowo_distillation_endpoints):
            raise ValueError(
                'YOWOFormer views_per_step must be between 1 and the number '
                'of endpoints'
            )
        yowo_distillation_interval = int(
            yowo_distillation_cfg.get('interval', 1)
        )
        if yowo_distillation_interval < 1:
            raise ValueError('YOWOFormer distillation interval must be positive')

        from yolost.yowoformer_distillation import (
            PyramidAlignedYOWOFormerDistillationLoss,
        )
        yowo_distillation_criterion = (
            PyramidAlignedYOWOFormerDistillationLoss(
                num_classes=cfg['model']['num_classes'],
                class_weight=yowo_distillation_cfg.get(
                    'class_weight', 1.0
                ),
                object_weight=yowo_distillation_cfg.get(
                    'object_weight', 1.0
                ),
                box_weight=yowo_distillation_cfg.get('box_weight', 1.0),
                temperature=yowo_distillation_cfg.get('temperature', 2.0),
                min_teacher_confidence=yowo_distillation_cfg.get(
                    'min_teacher_confidence', 0.03
                ),
                background_floor=yowo_distillation_cfg.get(
                    'background_floor', 0.01
                ),
                box_ciou_weight=yowo_distillation_cfg.get(
                    'box_ciou_weight', 1.0
                ),
                query_class_weight=yowo_distillation_cfg.get(
                    'query_class_weight', 0.0
                ),
                query_box_weight=yowo_distillation_cfg.get(
                    'query_box_weight', 0.0
                ),
                query_visibility_weight=yowo_distillation_cfg.get(
                    'query_visibility_weight', 0.0
                ),
                query_max_detections=yowo_distillation_cfg.get(
                    'query_max_detections', 16
                ),
                query_nms_iou=yowo_distillation_cfg.get(
                    'query_nms_iou', 0.5
                ),
                query_track_iou=yowo_distillation_cfg.get(
                    'query_track_iou', 0.3
                ),
                query_match_class_cost=yowo_distillation_cfg.get(
                    'query_match_class_cost', 2.0
                ),
                query_match_box_cost=yowo_distillation_cfg.get(
                    'query_match_box_cost', 5.0
                ),
                query_match_visibility_cost=yowo_distillation_cfg.get(
                    'query_match_visibility_cost', 1.0
                ),
            ).to(device)
        )
        log(
            'YOWOFormer pyramid distillation enabled: '
            f'checkpoint={yowo_checkpoint} sha256={actual_sha} '
            f'variant={yowo_teacher_info.get("videomae_variant")} '
            f'endpoints={yowo_distillation_endpoints} '
            f'views_per_step={yowo_views_per_step} '
            f'interval={yowo_distillation_interval} '
            f'lambda={yowo_distillation_lambda:g}',
            rank,
        )

    # Training loop
    log(f'Starting training: {epochs} epochs, {len(train_loader)} steps/epoch', rank)
    log(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}', rank)

    grad_accum = cfg['train'].get('grad_accum', 1)
    # backward_loss_divisor defaults to grad_accum (average the accumulated
    # micro-batch losses). The reported configs set it to 1, which sums them:
    # the gradient is grad_accum times the average. AdamW's update is invariant
    # to that scale up to eps, but clip_grad_norm then acts on the summed norm.
    backward_loss_divisor = float(
        cfg['train'].get('backward_loss_divisor', grad_accum)
    )
    if backward_loss_divisor <= 0:
        raise ValueError('train.backward_loss_divisor must be positive')
    clip_grad = cfg['train'].get('clip_grad_norm', 0)
    max_steps_per_epoch = cfg['train'].get('max_steps_per_epoch')

    for epoch in range(start_epoch, epochs):
        model.train()
        if freeze_all_batchnorm_stats:
            lock_all_batchnorm_stats(raw_model)
        elif freeze_frozen_batchnorm_stats:
            lock_frozen_batchnorm_stats(raw_model)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_losses = {'cls_loss': 0, 'box_loss': 0, 'obj_loss': 0, 'num_fg': 0}
        if model_type in ('phase3', 'phase3_temporal_context', 'dinov3', 'videomae'):
            epoch_losses['bnd_loss'] = 0
            if cfg['loss'].get('lambda_tube_cls', 0.0) > 0:
                epoch_losses['tube_cls_loss'] = 0
            if cfg['loss'].get('lambda_tube_box', 0.0) > 0:
                epoch_losses['tube_box_loss'] = 0
            if cfg['loss'].get('lambda_dn_cls', 0.0) > 0:
                epoch_losses['dn_cls_loss'] = 0
            if cfg['loss'].get('lambda_dn_box', 0.0) > 0:
                epoch_losses['dn_box_loss'] = 0
            if cfg['loss'].get('lambda_memory', 0.0) > 0:
                epoch_losses['memory_loss'] = 0
            if cfg['model'].get('apt_tube_queries', False):
                for key in ('query_cls_loss', 'query_box_loss',
                            'query_giou_loss',
                            'query_visibility_loss', 'query_boundary_loss',
                            'query_velocity_loss', 'query_acceleration_loss',
                            'query_start_loss', 'query_end_loss',
                            'query_interval_iou_loss', 'query_coverage_loss',
                            'query_fragmentation_loss',
                            'query_boundary_distance_loss',
                            'query_boundary_distance_slope_loss',
                            'query_quality_loss',
                            'query_transport_loss',
                            'query_geometry_preservation_loss'):
                    epoch_losses[key] = 0
        if model_type in ('bmvit_lite', 'mvit_bmvit'):
            epoch_losses['giou_loss'] = 0
        if distillation_enabled:
            epoch_losses['distill_loss'] = 0
        if offline_trajectory_enabled:
            epoch_losses['offline_trajectory_loss'] = 0
        if offline_dense_enabled:
            epoch_losses['offline_dense_loss'] = 0
        if yowo_distillation_enabled:
            epoch_losses['yowo_distill_loss'] = 0
            epoch_losses['yowo_teacher_invalid_cells'] = 0
        t_start = time.time()
        processed_steps = 0

        for step, (clips, targets) in enumerate(train_loader):
            clips = normalize_uint8_clips(clips.to(device, non_blocking=True))
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            with autocast(
                    enabled=cfg['train']['mixed_precision'], dtype=amp_dtype):
                outputs = model(clips, targets=targets) if model_type == 'videomae' else model(clips)
                loss, loss_dict = criterion(
                    outputs, targets,
                    temporal_strides=raw_model.temporal_strides,
                    spatial_strides=raw_model.spatial_strides,
                    img_size=cfg['data']['img_size'],
                )
                if offline_trajectory_enabled:
                    (
                        offline_trajectory_loss,
                        offline_trajectory_metrics,
                    ) = offline_trajectory_criterion(
                        outputs, targets,
                        clip_length=cfg['data']['clip_length'],
                    )
                    if not torch.isfinite(offline_trajectory_loss):
                        raise FloatingPointError(
                            'Non-finite offline trajectory loss at '
                            f'epoch={epoch} step={step}: '
                            f'{offline_trajectory_metrics}'
                        )
                    loss = (
                        loss
                        + offline_trajectory_lambda
                        * offline_trajectory_loss
                    )
                    loss_dict.update(offline_trajectory_metrics)
                    loss_dict['loss'] = float(loss.detach().item())
                if offline_dense_enabled:
                    offline_dense_loss, offline_dense_metrics = (
                        offline_dense_criterion(
                            outputs,
                            targets,
                            temporal_strides=raw_model.temporal_strides,
                            spatial_strides=raw_model.spatial_strides,
                            img_size=cfg['data']['img_size'],
                        )
                    )
                    if not torch.isfinite(offline_dense_loss):
                        raise FloatingPointError(
                            'Non-finite offline dense-pyramid loss at '
                            f'epoch={epoch} step={step}: '
                            f'{offline_dense_metrics}'
                        )
                    loss = loss + offline_dense_lambda * offline_dense_loss
                    loss_dict.update(offline_dense_metrics)
                    loss_dict['loss'] = float(loss.detach().item())
                if distillation_enabled:
                    teacher_outputs = []
                    with torch.no_grad():
                        for teacher in distillation_teachers:
                            teacher_outputs.append({
                                'outputs': teacher['model'](clips),
                                'weight': teacher['weight'],
                                'class_ids': teacher['class_ids'],
                                'component_weights': teacher[
                                    'component_weights'
                                ],
                            })
                    distillation_loss, distillation_metrics = (
                        distillation_criterion(outputs, teacher_outputs)
                    )
                    loss = loss + distillation_lambda * distillation_loss
                    loss_dict.update(distillation_metrics)
                    loss_dict['loss'] = float(loss.detach().item())
                if (
                    yowo_distillation_enabled
                    and step % yowo_distillation_interval == 0
                ):
                    yowo_teacher_views = []
                    distillation_step = (
                        epoch * len(train_loader) + step
                    ) // yowo_distillation_interval
                    endpoint_offset = (
                        distillation_step * yowo_views_per_step
                    ) % len(yowo_distillation_endpoints)
                    selected_endpoints = [
                        yowo_distillation_endpoints[
                            (endpoint_offset + index)
                            % len(yowo_distillation_endpoints)
                        ]
                        for index in range(yowo_views_per_step)
                    ]
                    with torch.no_grad():
                        for endpoint in selected_endpoints:
                            start = endpoint + 1 - yowo_teacher_clip_length
                            if start >= 0:
                                teacher_clip = clips[:, :, start:endpoint + 1]
                            else:
                                padding = clips[:, :, :1].expand(
                                    -1, -1, -start, -1, -1
                                )
                                teacher_clip = torch.cat(
                                    (padding, clips[:, :, :endpoint + 1]),
                                    dim=2,
                                )
                            if teacher_clip.shape[2] != yowo_teacher_clip_length:
                                raise RuntimeError(
                                    'Incorrect YOWOFormer temporal view length: '
                                    f'{teacher_clip.shape[2]} != '
                                    f'{yowo_teacher_clip_length}'
                                )
                            yowo_teacher_views.append({
                                'endpoint': endpoint,
                                'outputs': yowo_distillation_teacher(
                                    teacher_clip
                                ),
                            })
                    yowo_distillation_loss, yowo_distillation_metrics = (
                        yowo_distillation_criterion(
                            outputs,
                            yowo_teacher_views,
                            temporal_strides=raw_model.temporal_strides,
                            spatial_strides=raw_model.spatial_strides,
                            img_size=cfg['data']['img_size'],
                            clip_length=cfg['data']['clip_length'],
                        )
                    )
                    if not torch.isfinite(yowo_distillation_loss):
                        raise FloatingPointError(
                            'Non-finite YOWOFormer distillation loss at '
                            f'epoch={epoch} step={step}: '
                            f'{yowo_distillation_metrics}'
                        )
                    loss = (
                        loss
                        + yowo_distillation_lambda * yowo_distillation_loss
                    )
                    loss_dict.update(yowo_distillation_metrics)
                    loss_dict['loss'] = float(loss.detach().item())
                loss = loss / backward_loss_divisor

            if (conflict_enabled and
                    len(conflict_records) < conflict_max_measurements and
                    step % conflict_interval == 0):
                features = conflict_capture.get('features')
                components = getattr(criterion, 'last_loss_components', None)
                if features is None or components is None:
                    raise RuntimeError(
                        'gradient conflict diagnostic did not capture features '
                        'or differentiable loss components'
                    )
                level_metrics = gradient_conflict_metrics(
                    components['frame'], components['tube_boundary'], features
                )
                conflict_records.append({
                    'epoch': epoch,
                    'step': step,
                    'frame_loss': float(components['frame'].detach().item()),
                    'tube_boundary_loss': float(
                        components['tube_boundary'].detach().item()
                    ),
                    'levels': level_metrics,
                })
                metric_text = ' '.join(
                    f"{metric['level']}:cos={metric['cosine']:.4f},"
                    f"negch={metric['negative_channel_fraction']:.3f},"
                    f"ratio={metric['tube_to_frame_norm_ratio']:.3f}"
                    for metric in level_metrics if metric.get('connected', False)
                )
                log(
                    f'Gradient conflict [{len(conflict_records)}/'
                    f'{conflict_max_measurements}] {metric_text}', rank
                )
                write_conflict_report()

            if not torch.isfinite(loss):
                nonfinite_streak += 1
                if not nonfinite_reported:
                    log(
                        f'NON-FINITE LOSS at epoch {epoch} step {step}: '
                        f'{loss_dict}. GradScaler protects the weights but '
                        'not BatchNorm running statistics, which are updated '
                        'in the forward pass, so the model may already be '
                        'unrecoverable.',
                        rank,
                    )
                    nonfinite_reported = True
                if (nonfinite_abort_steps > 0
                        and nonfinite_streak >= nonfinite_abort_steps):
                    raise RuntimeError(
                        f'aborting: loss was non-finite for '
                        f'{nonfinite_streak} consecutive steps, ending at '
                        f'epoch {epoch} step {step}. Consider '
                        'train.amp_dtype: bfloat16 or a lower learning rate.'
                    )
            else:
                nonfinite_streak = 0

            scaler.scale(loss).backward()
            if hasattr(criterion, 'last_loss_components'):
                criterion.last_loss_components = None

            if (step + 1) % grad_accum == 0:
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                if ema is not None:
                    ema.update(raw_model)

            epoch_loss += loss_dict['loss']
            processed_steps += 1
            for k in epoch_losses:
                epoch_losses[k] += loss_dict.get(k, 0)

            if step % 20 == 0 and is_main_process(rank):
                lr = optimizer.param_groups[0]['lr']
                bnd_str = f' bnd={loss_dict["bnd_loss"]:.4f}' if 'bnd_loss' in loss_dict else ''
                giou_str = f' giou={loss_dict["giou_loss"]:.4f}' if 'giou_loss' in loss_dict else ''
                tube_cls_str = f' tube_cls={loss_dict["tube_cls_loss"]:.4f}' if 'tube_cls_loss' in loss_dict else ''
                tube_box_str = f' tube_box={loss_dict["tube_box_loss"]:.4f}' if 'tube_box_loss' in loss_dict else ''
                dn_cls_str = f' dn_cls={loss_dict["dn_cls_loss"]:.4f}' if 'dn_cls_loss' in loss_dict else ''
                dn_box_str = f' dn_box={loss_dict["dn_box_loss"]:.4f}' if 'dn_box_loss' in loss_dict else ''
                memory_str = f' memory={loss_dict["memory_loss"]:.4f}' if 'memory_loss' in loss_dict else ''
                distill_str = (f' distill={loss_dict["distill_loss"]:.4f}'
                               if 'distill_loss' in loss_dict else '')
                offline_trajectory_str = (
                    f' offtraj={loss_dict["offline_trajectory_loss"]:.4f}'
                    f' offcls={loss_dict["offline_trajectory_class"]:.4f}'
                    f' offbox={loss_dict["offline_trajectory_box"]:.4f}'
                    f' offvis={loss_dict["offline_trajectory_visibility"]:.4f}'
                    f' offtracks={loss_dict["offline_trajectory_tracks"]}'
                    f' offweight='
                    f'{loss_dict["offline_trajectory_target_weight"]:.3f}'
                    if 'offline_trajectory_loss' in loss_dict else ''
                )
                offline_dense_str = (
                    f' offdense={loss_dict["offline_dense_loss"]:.4f}'
                    f' odcls={loss_dict["offline_dense_class"]:.4f}'
                    f' odobj={loss_dict["offline_dense_object"]:.4f}'
                    f' odbox={loss_dict["offline_dense_box"]:.4f}'
                    f' odgiou={loss_dict["offline_dense_giou"]:.4f}'
                    f' odvel={loss_dict["offline_dense_velocity"]:.4f}'
                    f' odvpairs={loss_dict["offline_dense_velocity_pairs"]}'
                    f' odtargets={loss_dict["offline_dense_targets"]}'
                    f' odfg={loss_dict["offline_dense_foreground"]}'
                    if 'offline_dense_loss' in loss_dict else ''
                )
                yowo_distill_str = (
                    f' yowo={loss_dict["yowo_distill_loss"]:.4f}'
                    f' ycls={loss_dict["yowo_distill_class"]:.4f}'
                    f' yobj={loss_dict["yowo_distill_object"]:.4f}'
                    f' ybox={loss_dict["yowo_distill_box"]:.4f}'
                    f' yqcls={loss_dict["yowo_distill_query_class"]:.4f}'
                    f' yqbox={loss_dict["yowo_distill_query_box"]:.4f}'
                    f' yqvis={loss_dict["yowo_distill_query_visibility"]:.4f}'
                    f' ytracks={loss_dict["yowo_teacher_tracks"]}'
                    f' yinv={loss_dict["yowo_teacher_invalid_cells"]}'
                    if 'yowo_distill_loss' in loss_dict else ''
                )
                query_str = (f' qcls={loss_dict["query_cls_loss"]:.4f}'
                             f' qbox={loss_dict["query_box_loss"]:.4f}'
                             f' qgiou={loss_dict["query_giou_loss"]:.4f}'
                             f' qvis={loss_dict["query_visibility_loss"]:.4f}'
                             f' qbnd={loss_dict["query_boundary_loss"]:.4f}'
                             f' qvel={loss_dict["query_velocity_loss"]:.4f}'
                             f' qacc={loss_dict["query_acceleration_loss"]:.4f}'
                             f' qstart={loss_dict["query_start_loss"]:.4f}'
                             f' qend={loss_dict["query_end_loss"]:.4f}'
                             f' qiou={loss_dict["query_interval_iou_loss"]:.4f}'
                             f' qcov={loss_dict["query_coverage_loss"]:.4f}'
                             f' qfrag={loss_dict["query_fragmentation_loss"]:.4f}'
                             f' qdist={loss_dict["query_boundary_distance_loss"]:.4f}'
                             f' qdslope={loss_dict["query_boundary_distance_slope_loss"]:.4f}'
                             f' qquality={loss_dict["query_quality_loss"]:.4f}'
                             f' qtransport={loss_dict["query_transport_loss"]:.4f}'
                             f' qprotect={loss_dict["query_geometry_preservation_loss"]:.4f}'
                             if 'query_cls_loss' in loss_dict else '')
                print(f'  [{epoch}/{epochs}] step {step}/{len(train_loader)} '
                      f'loss={loss_dict["loss"]:.4f} '
                      f'cls={loss_dict["cls_loss"]:.4f} '
                      f'box={loss_dict["box_loss"]:.4f} '
                      f'obj={loss_dict["obj_loss"]:.4f}{bnd_str}{giou_str}{tube_cls_str}{tube_box_str}{dn_cls_str}{dn_box_str}{memory_str}{distill_str}{offline_trajectory_str}{offline_dense_str}{yowo_distill_str}{query_str} '
                      f'fg={loss_dict["num_fg"]} '
                      f'lr={lr:.6f}', flush=True)

            if max_steps_per_epoch is not None and processed_steps >= int(max_steps_per_epoch):
                log(f'Stopping epoch at configured max_steps_per_epoch={max_steps_per_epoch}', rank)
                break

        # Epoch summary
        n_steps = max(processed_steps, 1)
        elapsed = time.time() - t_start
        bnd_str = f' bnd={epoch_losses["bnd_loss"]/n_steps:.4f}' if 'bnd_loss' in epoch_losses else ''
        giou_str = f' giou={epoch_losses["giou_loss"]/n_steps:.4f}' if 'giou_loss' in epoch_losses else ''
        tube_cls_str = f' tube_cls={epoch_losses["tube_cls_loss"]/n_steps:.4f}' if 'tube_cls_loss' in epoch_losses else ''
        tube_box_str = f' tube_box={epoch_losses["tube_box_loss"]/n_steps:.4f}' if 'tube_box_loss' in epoch_losses else ''
        dn_cls_str = f' dn_cls={epoch_losses["dn_cls_loss"]/n_steps:.4f}' if 'dn_cls_loss' in epoch_losses else ''
        dn_box_str = f' dn_box={epoch_losses["dn_box_loss"]/n_steps:.4f}' if 'dn_box_loss' in epoch_losses else ''
        memory_str = f' memory={epoch_losses["memory_loss"]/n_steps:.4f}' if 'memory_loss' in epoch_losses else ''
        distill_str = (f' distill={epoch_losses["distill_loss"]/n_steps:.4f}'
                       if 'distill_loss' in epoch_losses else '')
        offline_dense_str = (
            f' offdense={epoch_losses["offline_dense_loss"]/n_steps:.4f}'
            if 'offline_dense_loss' in epoch_losses else ''
        )
        yowo_distill_str = (
            f' yowo={epoch_losses["yowo_distill_loss"]/n_steps:.4f}'
            f' yinv={epoch_losses["yowo_teacher_invalid_cells"]:.0f}'
            if 'yowo_distill_loss' in epoch_losses else ''
        )
        query_str = (f' qcls={epoch_losses["query_cls_loss"]/n_steps:.4f}'
                     f' qbox={epoch_losses["query_box_loss"]/n_steps:.4f}'
                     f' qgiou={epoch_losses["query_giou_loss"]/n_steps:.4f}'
                     f' qvis={epoch_losses["query_visibility_loss"]/n_steps:.4f}'
                     f' qbnd={epoch_losses["query_boundary_loss"]/n_steps:.4f}'
                     f' qvel={epoch_losses["query_velocity_loss"]/n_steps:.4f}'
                     f' qacc={epoch_losses["query_acceleration_loss"]/n_steps:.4f}'
                     f' qstart={epoch_losses["query_start_loss"]/n_steps:.4f}'
                     f' qend={epoch_losses["query_end_loss"]/n_steps:.4f}'
                     f' qiou={epoch_losses["query_interval_iou_loss"]/n_steps:.4f}'
                     f' qcov={epoch_losses["query_coverage_loss"]/n_steps:.4f}'
                     f' qfrag={epoch_losses["query_fragmentation_loss"]/n_steps:.4f}'
                     f' qdist={epoch_losses["query_boundary_distance_loss"]/n_steps:.4f}'
                     f' qdslope={epoch_losses["query_boundary_distance_slope_loss"]/n_steps:.4f}'
                     f' qquality={epoch_losses["query_quality_loss"]/n_steps:.4f}'
                     f' qtransport={epoch_losses["query_transport_loss"]/n_steps:.4f}'
                     f' qprotect={epoch_losses["query_geometry_preservation_loss"]/n_steps:.4f}'
                     if 'query_cls_loss' in epoch_losses else '')
        log(f'Epoch {epoch}/{epochs} done in {elapsed:.0f}s — '
            f'avg_loss={epoch_loss/n_steps:.4f} '
            f'cls={epoch_losses["cls_loss"]/n_steps:.4f} '
            f'box={epoch_losses["box_loss"]/n_steps:.4f} '
            f'obj={epoch_losses["obj_loss"]/n_steps:.4f}{bnd_str}{giou_str}{tube_cls_str}{tube_box_str}{dn_cls_str}{dn_box_str}{memory_str}{distill_str}{offline_dense_str}{yowo_distill_str}{query_str} '
            f'avg_fg={epoch_losses["num_fg"]/n_steps:.0f}', rank)

        # Save checkpoint
        if is_main_process(rank) and (epoch + 1) % cfg['output']['save_interval'] == 0:
            ckpt = {
                'epoch': epoch,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'config': cfg,
            }
            if ema is not None:
                ckpt['ema'] = ema.state_dict()
            if cfg['output'].get('epoch_checkpoints', 'full') == 'weights_only':
                # Resume uses latest.pt; periodic files are for evaluation only.
                ckpt = {key: ckpt[key] for key in ('epoch', 'model', 'config', 'ema')
                        if key in ckpt}
            save_path = os.path.join(exp_dir, f'epoch_{epoch}.pt')
            torch.save(ckpt, save_path)
            log(f'Saved checkpoint: {save_path}', rank)
            # Also save EMA-only checkpoint for easy eval
            if ema is not None:
                ema_path = os.path.join(exp_dir, f'ema_epoch_{epoch}.pt')
                torch.save({'epoch': epoch, 'model': ema.ema.state_dict(), 'config': cfg},
                           ema_path)

        # Save latest
        if is_main_process(rank) and cfg['output'].get('save_latest', True):
            ckpt = {
                'epoch': epoch,
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'config': cfg,
            }
            if ema is not None:
                ckpt['ema'] = ema.state_dict()
            torch.save(ckpt, os.path.join(exp_dir, 'latest.pt'))

    if conflict_hook is not None:
        conflict_hook.remove()
        write_conflict_report()
        log(
            f'Gradient-conflict report written to {conflict_report_path}', rank
        )

    # Save final
    if is_main_process(rank):
        torch.save({
            'epoch': epochs - 1,
            'model': raw_model.state_dict(),
            'config': cfg,
        }, os.path.join(exp_dir, 'final.pt'))
        if ema is not None:
            torch.save({
                'epoch': epochs - 1,
                'model': ema.ema.state_dict(),
                'config': cfg,
            }, os.path.join(exp_dir, 'ema_final.pt'))
        log('Training complete.', rank)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
