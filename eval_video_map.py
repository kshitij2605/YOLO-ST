"""Evaluate YOLO-ST video-mAP on UCF101-24.

Runs the full pipeline:
  1. Model inference → per-frame detections
  2. Per-frame NMS
  3. Tube assembly (probabilistic or hard voting)
  4. Video-mAP computation at IoU thresholds 0.2 and 0.5

Usage:
    python eval_video_map.py --config configs/yolost_phase3a_boundary.yaml \
        --checkpoint experiments/phase3a_boundary/final.pt \
        --method probabilistic
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torchvision
import yaml
from config_utils import load_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ucf101_24 import UCF101_24_Dataset, build_clip_starts
from yolost.tube_assembly import (assemble_tubes, assemble_tubes_hard_voting,
                                   interpolate_tube_gaps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--conf_thresh', type=float, default=0.3)
    p.add_argument('--nms_thresh', type=float, default=0.5)
    p.add_argument('--method', choices=['probabilistic', 'hard', 'both'],
                   default='both')
    p.add_argument('--tau_link', type=float, default=0.3)
    p.add_argument('--k_gap', type=int, default=5)
    p.add_argument('--tau_gap', type=float, default=0.2)
    p.add_argument('--cls_momentum', type=float, default=0.3)
    p.add_argument('--min_tube_length', type=int, default=3)
    p.add_argument('--use_boundary', action='store_true')
    p.add_argument('--tau_bnd', type=float, default=0.6)
    p.add_argument('--batch_size', type=int, default=8)
    return p.parse_args()


def load_model(cfg, checkpoint, device):
    """Load the model from config and checkpoint."""
    model_type = cfg['model'].get('type', 'uniform')
    if model_type == 'phase3':
        from yolost.model_phase3 import YOLOST_Phase3
        model = YOLOST_Phase3(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
            target_T=cfg['model'].get('target_T', [64, 32, 16]),
            clip_length=cfg['data']['clip_length'],
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
            keyframe_pyramid=cfg['model'].get('keyframe_pyramid', False),
            keyframe_version=cfg['model'].get('keyframe_version', 'n'),
            keyframe_pretrained_path=cfg['model'].get('keyframe_pretrained_path'),
            keyframe_freeze=cfg['model'].get('keyframe_freeze', True),
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
    else:
        from yolost.model import YOLOST
        model = YOLOST(
            num_classes=cfg['model']['num_classes'],
            img_size=cfg['data']['img_size'],
        ).to(device)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f'Loaded epoch {ckpt["epoch"]}')
    return model


def load_gt_tubes(annot_file, test_videos, clip_length=64, stride=1):
    """Load GT tubes from UCF101-24 annotations.

    Returns: dict video_name -> list of tube dicts:
        {'class': int, 'detections': {frame_id: np.array(4,)}}
    """
    with open(annot_file, 'rb') as f:
        annot = pickle.load(f, encoding='latin1')

    gt_tubes_by_video = {}
    for video_name in test_videos:
        gt_tubes_by_video[video_name] = []
        video_annot = annot.get('gttubes', {}).get(video_name, {})
        resolution = annot['resolution'].get(video_name, (240, 320))
        orig_h, orig_w = resolution

        for class_id, tubes in video_annot.items():
            for tube in tubes:
                tube_dict = {'class': int(class_id), 'detections': {}}
                tube_dict['video'] = video_name
                tube_dict['resolution'] = resolution
                for row in tube:
                    frame_id = int(row[0]) - 1  # 0-indexed
                    box = np.array([
                        row[1] / orig_w, row[2] / orig_h,
                        row[3] / orig_w, row[4] / orig_h
                    ], dtype=np.float32)
                    box = np.clip(box, 0, 1)
                    tube_dict['detections'][frame_id] = box
                if tube_dict['detections']:
                    gt_tubes_by_video[video_name].append(tube_dict)

    return gt_tubes_by_video


@torch.no_grad()
def extract_per_frame_detections(model, video_clips, device, conf_thresh,
                                  nms_thresh, num_classes, clip_length,
                                  start_frames):
    """Extract per-frame detections from a batch of clips.

    Args:
        video_clips: (B, 3, T, H, W) tensor
        start_frames: list of (video_name, start_frame_0indexed) per batch element

    Returns:
        dict: video_name -> {frame_id -> list of det dicts}
    """
    clips = video_clips.to(device)
    B = clips.shape[0]
    outputs = model(clips)

    has_boundary = len(outputs[0]) > 3

    all_dets = defaultdict(lambda: defaultdict(list))

    for b in range(B):
        video_name, start_f = start_frames[b]

        for cf in range(clip_length):
            frame_id = start_f + cf

            all_boxes = []
            all_scores = []
            all_cls_probs = []
            all_boundary = []

            for si, scale_out in enumerate(outputs):
                cls_pred, reg_pred, obj_pred = scale_out[0], scale_out[1], scale_out[2]
                bnd_pred = scale_out[3] if has_boundary else None

                t_stride = model.temporal_strides[si]
                s_stride = model.spatial_strides[si]
                T_det = cls_pred.shape[2]
                S = cls_pred.shape[3]
                step = s_stride / model.img_size

                t_det = cf // t_stride
                if t_det >= T_det:
                    continue

                obj_sig = torch.sigmoid(obj_pred[b, 0, t_det])
                cls_sig = torch.sigmoid(cls_pred[b, :, t_det])  # (nc, S, S)
                reg = reg_pred[b, :, t_det]

                combined = obj_sig.unsqueeze(0) * cls_sig
                max_score, _ = combined.max(dim=0)

                mask = max_score > conf_thresh
                if not mask.any():
                    continue

                h_idx, w_idx = torch.where(mask)
                scores = max_score[mask]
                cls_probs_all = cls_sig[:, h_idx, w_idx].T  # (N, nc)
                box_raw = reg[:, h_idx, w_idx].T

                cx = (w_idx.float() + torch.sigmoid(box_raw[:, 0])) * step
                cy = (h_idx.float() + torch.sigmoid(box_raw[:, 1])) * step
                w = torch.exp(box_raw[:, 2].clamp(max=5.0)) * step
                h = torch.exp(box_raw[:, 3].clamp(max=5.0)) * step
                boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_cls_probs.append(cls_probs_all)

                if bnd_pred is not None:
                    bnd_sig = torch.sigmoid(bnd_pred[b, 0, t_det])
                    all_boundary.append(bnd_sig[h_idx, w_idx])

            if not all_boxes:
                continue

            boxes_cat = torch.cat(all_boxes)
            scores_cat = torch.cat(all_scores)
            cls_probs_cat = torch.cat(all_cls_probs)
            bnd_cat = torch.cat(all_boundary) if all_boundary else None

            # Class-agnostic NMS
            keep = torchvision.ops.nms(boxes_cat, scores_cat, nms_thresh)
            for k in keep:
                det = {
                    'box': boxes_cat[k].cpu().numpy(),
                    'score': scores_cat[k].item(),
                    'class_probs': cls_probs_cat[k].cpu().numpy(),
                }
                if bnd_cat is not None:
                    det['boundary'] = bnd_cat[k].item()
                all_dets[video_name][frame_id].append(det)

    return all_dets


def _iou_np(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def tube_iou(tube_a, tube_b):
    """3D IoU = temporal_IoU × mean spatial IoU over overlapping frames."""
    frames_a = set(tube_a['detections'].keys())
    frames_b = set(tube_b['detections'].keys())
    overlap = frames_a & frames_b

    if not overlap:
        return 0.0

    t_iou = len(overlap) / len(frames_a | frames_b)

    s_ious = []
    for f in overlap:
        s_ious.append(_iou_np(tube_a['detections'][f], tube_b['detections'][f]))

    return t_iou * np.mean(s_ious)


def compute_video_map(pred_tubes, gt_tubes, iou_thresholds=(0.2, 0.5),
                      num_classes=24):
    """Compute video-mAP over action tubes."""
    class_names = UCF101_24_Dataset.CLASSES
    results = {}

    for thr in iou_thresholds:
        per_class_ap = {}

        for c in range(num_classes):
            preds_c = sorted([p for p in pred_tubes if p['class'] == c],
                             key=lambda x: -x['score'])
            gts_c = [g for g in gt_tubes if g['class'] == c]

            if not gts_c:
                continue
            if not preds_c:
                per_class_ap[c] = 0.0
                continue

            gt_matched = [False] * len(gts_c)
            tp = np.zeros(len(preds_c))
            fp = np.zeros(len(preds_c))

            for pi, pred in enumerate(preds_c):
                best_iou = 0
                best_gi = -1
                for gi, gt in enumerate(gts_c):
                    if gt_matched[gi]:
                        continue
                    iou = tube_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi

                if best_iou >= thr and best_gi >= 0:
                    tp[pi] = 1
                    gt_matched[best_gi] = True
                else:
                    fp[pi] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / len(gts_c)
            precision = tp_cum / (tp_cum + fp_cum)

            mrec = np.concatenate(([0.0], recall, [1.0]))
            mpre = np.concatenate(([1.0], precision, [0.0]))
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            px = np.linspace(0, 1, 101)
            ap = np.trapz(np.interp(px, mrec, mpre), px)
            per_class_ap[c] = ap

        mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
        results[thr] = {'mAP': mAP, 'per_class': per_class_ap}

    return results


def resolve_clip_overlaps(all_video_dets, clip_length):
    """Resolve overlapping clip detections using Hann window weighting.

    For overlapping clips (50% overlap), center frames get higher weight.
    When multiple clips predict at the same frame, keep higher-weighted dets.
    """
    # Build Hann window
    hann = np.hanning(clip_length + 2)[1:-1]  # exclude zeros at edges

    resolved = {}
    for video_name, clips_dets in all_video_dets.items():
        # clips_dets: list of (start_frame, {frame_id: [dets]})
        frame_dets = defaultdict(list)

        for start_frame, clip_det in clips_dets:
            for frame_id, dets in clip_det.items():
                cf = frame_id - start_frame
                if 0 <= cf < clip_length:
                    weight = hann[cf]
                    for d in dets:
                        d_weighted = d.copy()
                        d_weighted['score'] = d['score'] * weight
                        frame_dets[frame_id].append(d_weighted)

        # For each frame, run NMS to remove duplicates from overlapping clips
        resolved_dets = {}
        for frame_id, dets in frame_dets.items():
            if not dets:
                continue
            boxes = np.array([d['box'] for d in dets])
            scores = np.array([d['score'] for d in dets])
            boxes_t = torch.from_numpy(boxes).float()
            scores_t = torch.from_numpy(scores).float()
            keep = torchvision.ops.nms(boxes_t, scores_t, 0.5)
            resolved_dets[frame_id] = [dets[k] for k in keep.numpy()]

        resolved[video_name] = resolved_dets

    return resolved


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, args.checkpoint, device)
    model.eval()
    if hasattr(model, 'enable_persistent_memory'):
        model.enable_persistent_memory(True)

    clip_length = cfg['data']['clip_length']
    num_classes = cfg['model']['num_classes']

    # Load annotations for GT tubes
    with open(cfg['data']['annot_file'], 'rb') as f:
        annot = pickle.load(f, encoding='latin1')

    test_videos = annot['test_videos'][int(cfg['data'].get('split_index', 0))]
    gt_tubes_by_video = load_gt_tubes(cfg['data']['annot_file'], test_videos)

    # Run inference per video
    print(f'Running inference on {len(test_videos)} test videos...')
    t0 = time.time()

    all_pred_tubes = {m: [] for m in (['probabilistic', 'hard'] if args.method == 'both'
                                       else [args.method])}
    all_gt_tubes = []

    from train import collate_fn

    for vi, video_name in enumerate(test_videos):
        if hasattr(model, 'reset_memory'):
            model.reset_memory()
        if (vi + 1) % 100 == 0:
            print(f'  Processing video {vi+1}/{len(test_videos)}...')

        video_dir = os.path.join(cfg['data']['root'], video_name)
        if not os.path.isdir(video_dir):
            continue

        num_frames = annot['nframes'][video_name]

        # Build clips for this video
        val_dataset = UCF101_24_Dataset(
            root=cfg['data']['root'],
            annot_file=cfg['data']['annot_file'],
            clip_length=clip_length,
            stride=1,
            split='test',
            img_size=cfg['data']['img_size'],
            augment=False,
        )

        # Collect per-frame detections across all clips for this video
        video_frame_dets = defaultdict(list)

        for start in build_clip_starts(num_frames, clip_length):
            # Load clip
            frame_indices = []
            for i in range(clip_length):
                fi = start + i
                fi = min(fi, num_frames)
                frame_indices.append(fi)

            resolution = annot['resolution'].get(video_name, (240, 320))
            orig_h, orig_w = resolution
            import cv2
            frames = []
            for fi in frame_indices:
                img_path = os.path.join(cfg['data']['root'], video_name, f'{fi:05d}.jpg')
                if not os.path.exists(img_path):
                    img_path = os.path.join(cfg['data']['root'], video_name, f'{fi:05d}.png')
                img = cv2.imread(img_path)
                if img is None:
                    img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (cfg['data']['img_size'], cfg['data']['img_size']))
                frames.append(img)

            clip = np.stack(frames)
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
            clip = (clip - mean) / std
            clip = clip.unsqueeze(0)  # (1, 3, T, H, W)

            dets = extract_per_frame_detections(
                model, clip, device, args.conf_thresh, args.nms_thresh,
                num_classes, clip_length,
                [(video_name, start - 1)])  # 0-indexed frame

            # Collect with Hann weighting
            hann = np.hanning(clip_length + 2)[1:-1]
            for frame_id, frame_dets in dets.get(video_name, {}).items():
                cf = frame_id - (start - 1)
                if 0 <= cf < clip_length:
                    weight = hann[cf]
                    for d in frame_dets:
                        d['score'] *= weight
                        video_frame_dets[frame_id].append(d)

        # NMS to resolve overlapping clip detections per frame
        resolved_dets = {}
        for frame_id, dets_list in video_frame_dets.items():
            if not dets_list:
                continue
            boxes = np.array([d['box'] for d in dets_list])
            scores = np.array([d['score'] for d in dets_list])
            boxes_t = torch.from_numpy(boxes).float()
            scores_t = torch.from_numpy(scores).float()
            keep = torchvision.ops.nms(boxes_t, scores_t, 0.5)
            resolved_dets[frame_id] = [dets_list[k] for k in keep.numpy()]

        # Assemble tubes
        for method in all_pred_tubes:
            if method == 'probabilistic':
                tubes = assemble_tubes(
                    resolved_dets, num_frames,
                    num_classes=num_classes,
                    tau_link=args.tau_link,
                    k_gap=args.k_gap,
                    tau_gap=args.tau_gap,
                    cls_momentum=args.cls_momentum,
                    min_tube_length=args.min_tube_length,
                    use_boundary=args.use_boundary,
                    tau_bnd=args.tau_bnd,
                )
            else:
                tubes = assemble_tubes_hard_voting(
                    resolved_dets, num_frames,
                    num_classes=num_classes,
                    tau_link=args.tau_link,
                    k_gap=args.k_gap,
                    tau_gap=args.tau_gap,
                    min_tube_length=args.min_tube_length,
                )

            # Interpolate gaps
            tubes = interpolate_tube_gaps(tubes)
            all_pred_tubes[method].extend(tubes)

        # Collect GT tubes
        all_gt_tubes.extend(gt_tubes_by_video.get(video_name, []))

    elapsed = time.time() - t0
    print(f'Inference done in {elapsed:.0f}s')

    # Compute video-mAP
    class_names = UCF101_24_Dataset.CLASSES
    for method, tubes in all_pred_tubes.items():
        print(f'\n=== {method.upper()} Tube Assembly ===')
        print(f'Total tubes: {len(tubes)}')

        results = compute_video_map(tubes, all_gt_tubes,
                                    iou_thresholds=(0.2, 0.5),
                                    num_classes=num_classes)

        for thr, res in sorted(results.items()):
            print(f'\nVideo-mAP@{thr}: {res["mAP"]*100:.2f}%')
            for c, ap in sorted(res['per_class'].items(),
                                key=lambda x: x[1], reverse=True):
                print(f'  {class_names[c]:25s}: {ap*100:.1f}%')


if __name__ == '__main__':
    main()
