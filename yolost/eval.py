"""Frame-mAP evaluation for YOLO-ST.

Computes frame-level mean Average Precision at various IoU thresholds.
"""

import torch
import numpy as np
from collections import defaultdict


def compute_frame_map(predictions, ground_truths, iou_threshold=0.5, num_classes=24):
    """Compute frame-level mAP.

    Args:
        predictions: List of dicts per frame, each with:
            'boxes': (N, 4) x1y1x2y2 normalized
            'scores': (N,)
            'labels': (N,)
        ground_truths: List of dicts per frame, each with:
            'boxes': (M, 4) x1y1x2y2 normalized
            'labels': (M,)
        iou_threshold: IoU threshold for matching.
        num_classes: Number of classes.

    Returns:
        mAP: Mean AP across classes.
        per_class_ap: Dict of class_id -> AP.
    """
    # Collect all detections and GTs by class
    class_dets = defaultdict(list)  # class_id -> [(score, frame_idx, box)]
    class_gts = defaultdict(lambda: defaultdict(list))  # class_id -> frame_idx -> [box]
    class_ngt = defaultdict(int)  # class_id -> total GT count

    for frame_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Ground truths
        if gt['boxes'].shape[0] > 0:
            for i in range(gt['boxes'].shape[0]):
                cls = gt['labels'][i].item() if torch.is_tensor(gt['labels'][i]) else gt['labels'][i]
                box = gt['boxes'][i].cpu().numpy() if torch.is_tensor(gt['boxes'][i]) else gt['boxes'][i]
                class_gts[cls][frame_idx].append(box)
                class_ngt[cls] += 1

        # Predictions
        if pred['boxes'].shape[0] > 0:
            for i in range(pred['boxes'].shape[0]):
                cls = pred['labels'][i].item() if torch.is_tensor(pred['labels'][i]) else pred['labels'][i]
                score = pred['scores'][i].item() if torch.is_tensor(pred['scores'][i]) else pred['scores'][i]
                box = pred['boxes'][i].cpu().numpy() if torch.is_tensor(pred['boxes'][i]) else pred['boxes'][i]
                class_dets[cls].append((score, frame_idx, box))

    # Compute AP per class
    per_class_ap = {}
    for cls in range(num_classes):
        ngt = class_ngt[cls]
        if ngt == 0:
            continue

        dets = class_dets[cls]
        if not dets:
            per_class_ap[cls] = 0.0
            continue

        # Sort by score descending
        dets.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        matched = defaultdict(set)  # frame_idx -> set of matched GT indices

        for d_idx, (score, frame_idx, box) in enumerate(dets):
            gt_boxes = class_gts[cls].get(frame_idx, [])
            if not gt_boxes:
                fp[d_idx] = 1
                continue

            best_iou = 0
            best_gt = -1
            for g_idx, gt_box in enumerate(gt_boxes):
                iou = _compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = g_idx

            if best_iou >= iou_threshold and best_gt not in matched[frame_idx]:
                tp[d_idx] = 1
                matched[frame_idx].add(best_gt)
            else:
                fp[d_idx] = 1

        # Compute precision-recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / ngt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # AP using 11-point interpolation
        ap = _compute_ap(recall, precision)
        per_class_ap[cls] = ap

    if per_class_ap:
        mAP = np.mean(list(per_class_ap.values()))
    else:
        mAP = 0.0

    return mAP, per_class_ap


def _compute_iou(box1, box2):
    """Compute IoU between two boxes in x1y1x2y2 format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-7)


def _compute_ap(recall, precision):
    """Compute AP using all-point interpolation (PASCAL VOC style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
