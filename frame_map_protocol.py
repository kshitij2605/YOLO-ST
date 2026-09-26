"""Protocol-aware UCF101-24 frame-AP helpers.

The benchmark evaluates unique annotated video frames. Boxes are converted back
to integer image coordinates and scored with the VOC every-point AP integral,
matching the widely used UCF/JHMDB frame evaluator.
"""

from collections import defaultdict

import numpy as np
import torch
import torchvision


def load_frame_ground_truth(annot, videos, drop_tube_terminal=False):
    """Build per-class GT and the unique frame set used by frame evaluation."""
    gt_by_class = defaultdict(list)
    frames_by_video = defaultdict(set)
    for video_name in videos:
        for class_id, tubes in annot.get("gttubes", {}).get(video_name, {}).items():
            for tube in tubes:
                rows = tube[:-1] if drop_tube_terminal else tube
                for row in rows:
                    frame_id = int(row[0]) - 1
                    frame_key = (video_name, frame_id)
                    box = np.asarray(row[1:5], dtype=np.float32)
                    gt_by_class[int(class_id)].append((frame_key, box))
                    frames_by_video[video_name].add(frame_id)
    return gt_by_class, frames_by_video


def normalized_to_pixels(box, resolution):
    """Convert normalized xyxy to the integer inclusive coordinates in UCF GT."""
    height, width = resolution
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    result = np.rint(np.asarray(box, dtype=np.float32) * scale)
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width - 1)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height - 1)
    return result


def resolve_frame_candidates(candidates, nms_thresh=0.5):
    """Apply class-aware NMS to candidates from overlapping input clips."""
    resolved = []
    by_class = defaultdict(list)
    for candidate in candidates:
        by_class[int(candidate["class"])].append(candidate)
    for class_candidates in by_class.values():
        boxes = torch.as_tensor(
            np.stack([item["box"] for item in class_candidates]), dtype=torch.float32
        )
        scores = torch.as_tensor(
            [item["score"] for item in class_candidates], dtype=torch.float32
        )
        keep = torchvision.ops.nms(boxes, scores, nms_thresh).tolist()
        resolved.extend(class_candidates[index] for index in keep)
    return resolved


def add_frame_predictions(pred_by_class, video_name, frame_id, detections,
                          resolution, score_scale=1.0):
    """Append normalized detections to the global per-class AP accumulator."""
    frame_key = (video_name, frame_id)
    for detection in detections:
        class_id = int(detection["class"])
        score = float(detection["score"]) * score_scale
        box = normalized_to_pixels(detection["box"], resolution)
        pred_by_class[class_id].append((score, frame_key, box))


def inclusive_iou(box_a, box_b):
    """Pixel-coordinate IoU used by the common UCF/JHMDB VOC evaluator."""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    if x2 < x1 or y2 < y1:
        return 0.0
    intersection = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0]) + 1.0) * max(
        0.0, float(box_a[3]) - float(box_a[1]) + 1.0
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0]) + 1.0) * max(
        0.0, float(box_b[3]) - float(box_b[1]) + 1.0
    )
    return intersection / max(area_a + area_b - intersection, 1e-7)


def voc_ap(recall, precision):
    """VOC every-point interpolation, as used by the reference frame evaluator."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changed] - mrec[changed - 1]) * mpre[changed]))


def compute_frame_map(pred_by_class, gt_by_class, num_classes=24, iou_thresh=0.5,
                      evaluated_frames=None):
    """Compute independent-frame AP with one-to-one matching per video frame."""
    per_class = {}
    for class_id in range(num_classes):
        ground_truth = gt_by_class.get(class_id, [])
        if not ground_truth:
            continue
        predictions = pred_by_class.get(class_id, [])
        if evaluated_frames is not None:
            predictions = [
                item for item in predictions if item[1] in evaluated_frames
            ]
        predictions = sorted(predictions, key=lambda item: -item[0])
        gt_by_frame = defaultdict(list)
        for gt_index, (frame_key, box) in enumerate(ground_truth):
            gt_by_frame[frame_key].append((gt_index, box))
        matched = np.zeros(len(ground_truth), dtype=bool)
        true_positive = np.zeros(len(predictions), dtype=np.float64)
        false_positive = np.zeros(len(predictions), dtype=np.float64)
        for pred_index, (_, frame_key, pred_box) in enumerate(predictions):
            best_iou = 0.0
            best_gt = -1
            for gt_index, gt_box in gt_by_frame.get(frame_key, []):
                if matched[gt_index]:
                    continue
                overlap = inclusive_iou(pred_box, gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt = gt_index
            if best_gt >= 0 and best_iou >= iou_thresh:
                matched[best_gt] = True
                true_positive[pred_index] = 1.0
            else:
                false_positive[pred_index] = 1.0
        true_positive = np.cumsum(true_positive)
        false_positive = np.cumsum(false_positive)
        recall = true_positive / len(ground_truth)
        precision = true_positive / np.maximum(
            true_positive + false_positive, np.finfo(np.float64).eps
        )
        per_class[class_id] = voc_ap(recall, precision)
    mean_ap = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_ap, per_class
