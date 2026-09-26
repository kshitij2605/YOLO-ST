"""Tube Assembly — Gap-tolerant linking with probabilistic class commitment.

Implements Phase 3B/3C of YOLO-ST:
  - Hungarian matching with IoU + class similarity cost
  - Gap-tolerant linking with linear box extrapolation
  - Exponential moving average class probabilities
  - Boundary-based tube splitting (optional)

Input: per-frame detections (after NMS)
Output: action tubes with final class labels and confidence scores
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _iou(box1, box2):
    """IoU between two (4,) arrays/tensors [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def _cosine_sim(a, b):
    """Cosine similarity between two 1D numpy arrays."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return dot / (na * nb)


def assemble_tubes(frame_detections, num_frames, num_classes=24,
                   tau_link=0.3, k_gap=5, tau_gap=0.2,
                   cls_momentum=0.3, min_tube_length=3,
                   use_boundary=False, tau_bnd=0.6):
    """Assemble per-frame detections into action tubes.

    Args:
        frame_detections: dict frame_id -> list of dicts, each with:
            'box': np.array(4,) [x1,y1,x2,y2] normalized
            'score': float (objectness * max class score)
            'class_probs': np.array(num_classes,) (sigmoid class probabilities)
            'boundary': float (optional, boundary score 0-1)
        num_frames: total number of frames in the video
        num_classes: number of action classes
        tau_link: IoU threshold for linking
        k_gap: max gap frames before closing tube
        tau_gap: IoU threshold for gap recovery
        cls_momentum: alpha for EMA class probability update
        min_tube_length: discard tubes shorter than this
        use_boundary: whether to use boundary scores for tube splitting
        tau_bnd: boundary score threshold for splitting

    Returns:
        tubes: list of dicts with:
            'class': int, 'score': float,
            'detections': dict frame_id -> np.array(4,) box
    """
    INF = 1e6

    # Active tube state
    active = []  # list of tube dicts
    finished = []
    next_id = [0]

    def new_tube(frame, det):
        next_id[0] += 1
        return {
            'id': next_id[0],
            'detections': {frame: det['box'].copy()},
            'last_frame': frame,
            'last_box': det['box'].copy(),
            'velocity': np.zeros(4),
            'gap_count': 0,
            'cls_ema': det['class_probs'].copy(),
            'all_scores': [det['score']],
            'all_cls_probs': [det['class_probs'].copy()],
        }

    for t in range(num_frames):
        dets = frame_detections.get(t, [])

        if not dets and not active:
            continue

        n_tubes = len(active)
        n_dets = len(dets)

        # Predict tube positions (linear extrapolation)
        pred_boxes = []
        for tb in active:
            dt = tb['gap_count'] + 1
            pred_boxes.append(tb['last_box'] + tb['velocity'] * dt)

        matched_tubes = set()
        matched_dets = set()

        if n_tubes > 0 and n_dets > 0:
            # Build cost matrix
            cost = np.full((n_tubes, n_dets), INF)
            for i, tb in enumerate(active):
                threshold = tau_gap if tb['gap_count'] > 0 else tau_link
                for j, det in enumerate(dets):
                    iou = _iou(pred_boxes[i], det['box'])
                    if iou >= threshold:
                        cls_sim = _cosine_sim(tb['cls_ema'], det['class_probs'])
                        score = 0.7 * iou + 0.3 * cls_sim
                        cost[i, j] = -score

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= INF:
                    continue
                tb = active[r]
                det = dets[c]

                # Check boundary-based splitting
                if (use_boundary and tb['gap_count'] == 0 and
                        det.get('boundary', 0) > tau_bnd):
                    # Close current tube and start a new one
                    finished.append(tb)
                    active[r] = new_tube(t, det)
                    matched_tubes.add(r)
                    matched_dets.add(c)
                    continue

                # Normal update
                tb['detections'][t] = det['box'].copy()
                # Update velocity with EMA
                dt_actual = max(t - tb['last_frame'], 1)
                new_vel = (det['box'] - tb['last_box']) / dt_actual
                tb['velocity'] = 0.8 * tb['velocity'] + 0.2 * new_vel
                tb['last_frame'] = t
                tb['last_box'] = det['box'].copy()
                tb['gap_count'] = 0
                tb['all_scores'].append(det['score'])
                tb['all_cls_probs'].append(det['class_probs'].copy())
                # EMA class prob update (probabilistic tube classification)
                tb['cls_ema'] = ((1 - cls_momentum) * tb['cls_ema'] +
                                 cls_momentum * det['class_probs'])
                matched_tubes.add(r)
                matched_dets.add(c)

        # Handle unmatched tubes
        for i in range(n_tubes):
            if i not in matched_tubes:
                active[i]['gap_count'] += 1
                if active[i]['gap_count'] > k_gap:
                    finished.append(active[i])
                    active[i] = None
        active = [tb for tb in active if tb is not None]

        # Start new tubes from unmatched detections
        for j in range(n_dets):
            if j not in matched_dets:
                active.append(new_tube(t, dets[j]))

    # Close remaining active tubes
    finished.extend(active)

    # Filter short tubes and compute final class/score
    tubes = []
    for tb in finished:
        if len(tb['detections']) < min_tube_length:
            continue

        # Final class from EMA (probabilistic deferred commitment)
        final_class = int(np.argmax(tb['cls_ema']))

        # Tube confidence: mean of per-frame scores × EMA confidence
        mean_score = np.mean(tb['all_scores'])
        ema_conf = float(tb['cls_ema'][final_class])
        final_score = mean_score * ema_conf

        tubes.append({
            'class': final_class,
            'score': final_score,
            'detections': tb['detections'],
        })

    return tubes


def assemble_tubes_hard_voting(frame_detections, num_frames, num_classes=24,
                                tau_link=0.3, k_gap=5, tau_gap=0.2,
                                min_tube_length=3):
    """Baseline: hard class voting (majority vote over tube frames).

    Same linking as assemble_tubes but uses argmax per frame + majority vote.
    """
    INF = 1e6
    active = []
    finished = []
    next_id = [0]

    def new_tube(frame, det):
        next_id[0] += 1
        cls = int(np.argmax(det['class_probs']))
        return {
            'id': next_id[0],
            'detections': {frame: det['box'].copy()},
            'last_frame': frame,
            'last_box': det['box'].copy(),
            'velocity': np.zeros(4),
            'gap_count': 0,
            'frame_classes': [cls],
            'all_scores': [det['score']],
        }

    for t in range(num_frames):
        dets = frame_detections.get(t, [])
        if not dets and not active:
            continue

        n_tubes = len(active)
        n_dets = len(dets)
        pred_boxes = [tb['last_box'] + tb['velocity'] * (tb['gap_count'] + 1)
                      for tb in active]

        matched_tubes = set()
        matched_dets = set()

        if n_tubes > 0 and n_dets > 0:
            cost = np.full((n_tubes, n_dets), INF)
            for i, tb in enumerate(active):
                threshold = tau_gap if tb['gap_count'] > 0 else tau_link
                for j, det in enumerate(dets):
                    iou = _iou(pred_boxes[i], det['box'])
                    if iou >= threshold:
                        cost[i, j] = -iou

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= INF:
                    continue
                tb = active[r]
                det = dets[c]
                tb['detections'][t] = det['box'].copy()
                dt_actual = max(t - tb['last_frame'], 1)
                new_vel = (det['box'] - tb['last_box']) / dt_actual
                tb['velocity'] = 0.8 * tb['velocity'] + 0.2 * new_vel
                tb['last_frame'] = t
                tb['last_box'] = det['box'].copy()
                tb['gap_count'] = 0
                tb['all_scores'].append(det['score'])
                tb['frame_classes'].append(int(np.argmax(det['class_probs'])))
                matched_tubes.add(r)
                matched_dets.add(c)

        for i in range(n_tubes):
            if i not in matched_tubes:
                active[i]['gap_count'] += 1
                if active[i]['gap_count'] > k_gap:
                    finished.append(active[i])
                    active[i] = None
        active = [tb for tb in active if tb is not None]

        for j in range(n_dets):
            if j not in matched_dets:
                active.append(new_tube(t, dets[j]))

    finished.extend(active)

    tubes = []
    for tb in finished:
        if len(tb['detections']) < min_tube_length:
            continue
        # Majority vote
        classes = np.array(tb['frame_classes'])
        final_class = int(np.bincount(classes, minlength=num_classes).argmax())
        final_score = float(np.mean(tb['all_scores']))
        tubes.append({
            'class': final_class,
            'score': final_score,
            'detections': tb['detections'],
        })

    return tubes


def interpolate_tube_gaps(tubes):
    """Fill gap frames in tubes with linearly interpolated boxes."""
    for tube in tubes:
        frames = sorted(tube['detections'].keys())
        if len(frames) < 2:
            continue
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            if f2 - f1 <= 1:
                continue
            box1 = tube['detections'][f1]
            box2 = tube['detections'][f2]
            for f in range(f1 + 1, f2):
                alpha = (f - f1) / (f2 - f1)
                tube['detections'][f] = (1 - alpha) * box1 + alpha * box2
    return tubes
