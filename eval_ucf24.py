"""Evaluate YOLO-ST on UCF101-24 — frame-mAP@0.5.

Follows the reference eval approach: iterate per-frame, match GT per-frame.
Uses 101-point AP interpolation (COCO-style) matching reference metrics.py.

Usage:
    python eval_ucf24.py --config configs/yolost_s_ucf24_v5_100ep.yaml \
        --checkpoint experiments/phase1_v5_100ep/final.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import yaml
from config_utils import load_config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yolost.model import YOLOST
from data.ucf101_24 import UCF101_24_Dataset
from train import collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--conf_thresh', type=float, default=0.005)
    p.add_argument('--nms_thresh', type=float, default=0.5)
    p.add_argument('--batch_size', type=int, default=8)
    return p.parse_args()


def _iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-7)


@torch.no_grad()
def evaluate(model, dataloader, device, conf_thresh, nms_thresh, num_classes, clip_length):
    """Evaluate per-frame mAP, following reference approach."""
    model.eval()

    # Per-class collection
    pred_by_class = {c: [] for c in range(num_classes)}
    gt_by_class = {c: [] for c in range(num_classes)}
    frame_counter = 0

    for clips, targets in dataloader:
        clips = clips.to(device)
        B = clips.shape[0]
        outputs = model(clips)

        for b in range(B):
            gt_boxes = targets['boxes'][b]
            gt_labels = targets['labels'][b]
            valid = gt_boxes.sum(dim=-1) > 0
            gt_boxes = gt_boxes[valid]
            gt_labels = gt_labels[valid]

            # Iterate over every clip frame (0..clip_length-1)
            for cf in range(clip_length):
                fid = frame_counter + cf

                # Register GT at this clip frame
                for i in range(gt_boxes.shape[0]):
                    if int(gt_boxes[i, 0].item()) == cf:
                        c = int(gt_labels[i].item())
                        gt_by_class[c].append((gt_boxes[i, 1:5].cpu().numpy(), fid))

                # Collect predictions from all scales that cover this clip frame
                all_boxes = []
                all_scores = []
                all_classes = []

                for si, scale_out in enumerate(outputs):
                    cls_pred, reg_pred, obj_pred = scale_out[0], scale_out[1], scale_out[2]
                    t_stride = model.temporal_strides[si]
                    s_stride = model.spatial_strides[si]
                    T_det = cls_pred.shape[2]
                    S = cls_pred.shape[3]
                    step = s_stride / model.img_size

                    # Which detection frame covers this clip frame?
                    t_det = cf // t_stride
                    if t_det >= T_det:
                        continue

                    obj_sig = torch.sigmoid(obj_pred[b, 0, t_det])
                    cls_sig = torch.sigmoid(cls_pred[b, :, t_det])
                    reg = reg_pred[b, :, t_det]

                    combined = obj_sig.unsqueeze(0) * cls_sig
                    max_score, max_cls = combined.max(dim=0)

                    mask = max_score > conf_thresh
                    if not mask.any():
                        continue

                    h_idx, w_idx = torch.where(mask)
                    scores = max_score[mask]
                    classes = max_cls[mask]
                    box_raw = reg[:, h_idx, w_idx].T

                    cx = (w_idx.float() + torch.sigmoid(box_raw[:, 0])) * step
                    cy = (h_idx.float() + torch.sigmoid(box_raw[:, 1])) * step
                    w = torch.exp(box_raw[:, 2].clamp(max=5.0)) * step
                    h = torch.exp(box_raw[:, 3].clamp(max=5.0)) * step
                    boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)

                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(classes)

                if not all_boxes:
                    continue

                boxes_cat = torch.cat(all_boxes)
                scores_cat = torch.cat(all_scores)
                classes_cat = torch.cat(all_classes)

                for c in classes_cat.unique():
                    c_mask = classes_cat == c
                    c_boxes = boxes_cat[c_mask]
                    c_scores = scores_cat[c_mask]
                    keep = torchvision.ops.nms(c_boxes, c_scores, nms_thresh)
                    for k in keep:
                        pred_by_class[int(c.item())].append(
                            (c_scores[k].item(), c_boxes[k].cpu().numpy(), fid))

            frame_counter += clip_length

    # Compute per-class AP (101-point interpolation, matching reference)
    per_class_ap = {}
    for c in range(num_classes):
        gts = gt_by_class[c]
        preds = pred_by_class[c]
        n_gt = len(gts)

        if n_gt == 0:
            continue
        if not preds:
            per_class_ap[c] = 0.0
            continue

        preds.sort(key=lambda x: -x[0])

        # Build GT lookup by frame_id
        gt_by_frame = {}
        for gi, (box, fid) in enumerate(gts):
            gt_by_frame.setdefault(fid, []).append((gi, box))

        gt_matched = np.zeros(n_gt, dtype=bool)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for pi, (score, pred_box, pred_fid) in enumerate(preds):
            frame_gts = gt_by_frame.get(pred_fid, [])
            if not frame_gts:
                fp[pi] = 1
                continue

            best_iou = 0.0
            best_gi = -1
            for gi, gt_box in frame_gts:
                if gt_matched[gi]:
                    continue
                iou = _iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= 0.5 and best_gi >= 0:
                tp[pi] = 1
                gt_matched[best_gi] = True
            else:
                fp[pi] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)

        # 101-point AP interpolation (matching reference)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        px = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(px, mrec, mpre), px)
        per_class_ap[c] = ap

    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
    return mAP, per_class_ap


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f'Loaded epoch {ckpt["epoch"]}')

    val_dataset = UCF101_24_Dataset(
        root=cfg['data']['root'],
        annot_file=cfg['data']['annot_file'],
        clip_length=cfg['data']['clip_length'],
        stride=cfg['data']['stride'],
        split='test',
        img_size=cfg['data']['img_size'],
        augment=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=cfg['data']['num_workers'], collate_fn=collate_fn,
    )

    print(f'Evaluating {len(val_dataset)} clips (conf={args.conf_thresh}, nms={args.nms_thresh})...')
    t0 = time.time()
    mAP, per_class = evaluate(
        model, val_loader, device, args.conf_thresh, args.nms_thresh,
        cfg['model']['num_classes'], cfg['data']['clip_length'])
    print(f'Done in {time.time()-t0:.0f}s')
    print(f'\nFrame-mAP@0.5: {mAP*100:.2f}%')
    names = UCF101_24_Dataset.CLASSES
    for cls, ap in sorted(per_class.items(), key=lambda x: x[1], reverse=True):
        print(f'  {names[cls]:25s}: {ap*100:.1f}%')


if __name__ == '__main__':
    main()
