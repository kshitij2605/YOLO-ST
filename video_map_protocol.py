"""UCF101-24 video-mAP and MOC-style causal tube linking.

The benchmark protocol keeps video identity, applies class-wise 3D tube NMS,
uses inclusive pixel-coordinate IoU, and integrates the raw precision/recall
curve as in the public MOC evaluation code.
"""

from collections import defaultdict

import numpy as np


def _box_iou(box_a, box_b, inclusive=False):
    extra = 1.0 if inclusive else 0.0
    left_top = np.maximum(box_a[:2], box_b[:2])
    right_bottom = np.minimum(box_a[2:], box_b[2:])
    extent = np.maximum(right_bottom - left_top + extra, 0.0)
    intersection = float(extent[0] * extent[1])
    area_a = float(np.prod(np.maximum(box_a[2:] - box_a[:2] + extra, 0.0)))
    area_b = float(np.prod(np.maximum(box_b[2:] - box_b[:2] + extra, 0.0)))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def _mean_overlap_iou(candidate_a, candidate_b):
    overlap = sorted(
        set(candidate_a["detections"]) & set(candidate_b["detections"])
    )
    if overlap:
        return float(np.mean([
            _box_iou(
                candidate_a["detections"][frame],
                candidate_b["detections"][frame],
            )
            for frame in overlap
        ]))

    frames_a = sorted(candidate_a["detections"])
    frames_b = sorted(candidate_b["detections"])
    if not frames_a or not frames_b:
        return 0.0
    if frames_a[-1] < frames_b[0]:
        return _box_iou(
            candidate_a["detections"][frames_a[-1]],
            candidate_b["detections"][frames_b[0]],
        )
    if frames_b[-1] < frames_a[0]:
        return _box_iou(
            candidate_a["detections"][frames_a[0]],
            candidate_b["detections"][frames_b[-1]],
        )
    return 0.0


def _moc_tubelet_nms(candidates, overlap_thresh=0.6, top_k=10):
    """Port MOC's score-decay tubelet NMS to dictionary tubelets."""
    if not candidates:
        return []
    scores = np.asarray([item["score"] for item in candidates], dtype=np.float64)
    weights = np.ones_like(scores)
    order = np.argsort(scores)[::-1]
    while order.size:
        selected = int(order[0])
        rest = order[1:]
        ious = np.asarray([
            _mean_overlap_iou(candidates[selected], candidates[int(index)])
            for index in rest
        ])
        suppressed = np.where(ious > overlap_thresh)[0]
        weights[rest[suppressed]] = 1.0 - ious[suppressed]
        order = rest[np.where(ious <= overlap_thresh)[0]]

    adjusted = scores * weights
    keep = np.argsort(adjusted)[::-1][:top_k]
    output = []
    for index in keep:
        candidate = dict(candidates[int(index)])
        candidate["detections"] = {
            frame: np.asarray(box, dtype=np.float32).copy()
            for frame, box in candidate["detections"].items()
        }
        candidate["score"] = float(adjusted[int(index)])
        candidate["scores"] = [candidate["score"]]
        candidate["boundary_scores"] = dict(
            candidate.get("boundary_scores", {})
        )
        output.append(candidate)
    return output


def _endpoint_compatible(candidate_a, candidate_b, tolerance,
                         transition_margin=2, min_confidence=0.0):
    """Reject links whose learned interval endpoints imply separate actions."""
    reliable = {}
    for name, candidate in (("a", candidate_a), ("b", candidate_b)):
        for endpoint in ("start", "end"):
            frame = candidate.get(f"{endpoint}_frame")
            confidence = float(candidate.get(f"{endpoint}_confidence", 0.0))
            censored = bool(candidate.get(f"{endpoint}_censored", True))
            reliable[(name, endpoint)] = (
                frame is not None and not censored and
                confidence >= min_confidence
            )

    for endpoint in ("start", "end"):
        if reliable[("a", endpoint)] and reliable[("b", endpoint)]:
            if abs(
                int(candidate_a[f"{endpoint}_frame"])
                - int(candidate_b[f"{endpoint}_frame"])
            ) > tolerance:
                return False

    if reliable[("a", "end")] and reliable[("b", "start")]:
        if (int(candidate_b["start_frame"]) >=
                int(candidate_a["end_frame"]) - transition_margin):
            return False
    if reliable[("b", "end")] and reliable[("a", "start")]:
        if (int(candidate_b["end_frame"]) <=
                int(candidate_a["start_frame"]) + transition_margin):
            return False
    return True


def _dense_detections(detections):
    if not detections:
        return {}
    frames = np.asarray(sorted(detections), dtype=np.int64)
    if frames.size == 1:
        return {int(frames[0]): np.asarray(detections[int(frames[0])]).copy()}
    boxes = np.stack([detections[int(frame)] for frame in frames])
    dense_frames = np.arange(frames[0], frames[-1] + 1)
    dense_boxes = np.stack([
        np.interp(dense_frames, frames, boxes[:, coordinate])
        for coordinate in range(4)
    ], axis=1)
    return {
        int(frame): dense_boxes[index].astype(np.float32)
        for index, frame in enumerate(dense_frames)
    }


def link_tubelets_moc(candidates, clip_starts, clip_length, link_iou=0.5,
                      tubelet_nms=0.6, top_k=10, score_thresh=0.005,
                      min_length=15, split_gap=None,
                      boundary_split_thresh=None, endpoint_tolerance=None,
                      endpoint_transition_margin=2,
                      endpoint_min_confidence=0.0):
    """Greedily link query tubelets using the public MOC BuildTubes policy.

    Our queries have visibility-masked, variable temporal support, so this is
    an architecture-aware adaptation of MOC's fixed-K linker. The ordering,
    NMS, assignment, aging, score, and overlap averaging match that policy.
    """
    candidates_by_class_start = defaultdict(list)
    classes = set()
    for candidate in candidates:
        key = (candidate["class"], candidate["clip_start"])
        candidates_by_class_start[key].append(candidate)
        classes.add(candidate["class"])

    linked = []
    for class_id in sorted(classes):
        active = []
        finished = []
        for clip_start in clip_starts:
            tubelets = _moc_tubelet_nms(
                candidates_by_class_start[(class_id, clip_start)],
                overlap_thresh=tubelet_nms,
                top_k=top_k,
            )
            active.sort(
                key=lambda tube: -float(np.mean([item["score"] for item in tube]))
            )
            expired = []
            for tube_index, tube in enumerate(active):
                last = tube[-1]
                ious = np.asarray([
                    _mean_overlap_iou(last, tubelet) for tubelet in tubelets
                ])
                compatible = np.ones(len(tubelets), dtype=bool)
                if endpoint_tolerance is not None:
                    compatible = np.asarray([
                        _endpoint_compatible(
                            last,
                            tubelet,
                            tolerance=endpoint_tolerance,
                            transition_margin=endpoint_transition_margin,
                            min_confidence=endpoint_min_confidence,
                        )
                        for tubelet in tubelets
                    ], dtype=bool)
                valid = np.where((ious >= link_iou) & compatible)[0]
                if valid.size:
                    selected = int(valid[np.argmax([
                        tubelets[int(index)]["score"] for index in valid
                    ])])
                    tube.append(tubelets.pop(selected))
                elif clip_start - last["clip_start"] >= clip_length:
                    expired.append(tube_index)
            for tube_index in reversed(expired):
                finished.append(active.pop(tube_index))
            active.extend([[tubelet] for tubelet in tubelets])
        finished.extend(active)

        for tubelets in finished:
            score = float(np.mean([item["score"] for item in tubelets]))
            if score < score_thresh:
                continue
            boxes_by_frame = defaultdict(list)
            boundaries_by_frame = defaultdict(list)
            for tubelet in tubelets:
                for frame, box in tubelet["detections"].items():
                    boxes_by_frame[frame].append(np.asarray(box, dtype=np.float32))
                for frame, value in tubelet.get("boundary_scores", {}).items():
                    boundaries_by_frame[frame].append(float(value))
            observed = {
                frame: np.mean(boxes, axis=0).astype(np.float32)
                for frame, boxes in boxes_by_frame.items()
            }
            observed_frames = sorted(observed)
            frame_segments = []
            if observed_frames:
                segment = [observed_frames[0]]
                for frame in observed_frames[1:]:
                    missing = frame - segment[-1] - 1
                    if split_gap is not None and missing > split_gap:
                        frame_segments.append(segment)
                        segment = []
                    segment.append(frame)
                frame_segments.append(segment)
            action_segments = []
            for segment in frame_segments:
                if boundary_split_thresh is None:
                    action_segments.append(segment)
                    continue
                active = [
                    frame for frame in segment
                    if boundaries_by_frame.get(frame)
                    and np.mean(boundaries_by_frame[frame]) >= boundary_split_thresh
                ]
                events = []
                for frame in active:
                    if not events or frame > events[-1][-1] + 1:
                        events.append([])
                    events[-1].append(frame)
                cuts = []
                segment_start, segment_end = segment[0], segment[-1]
                for event in events:
                    peak = max(
                        event,
                        key=lambda frame: np.mean(boundaries_by_frame[frame]),
                    )
                    previous = cuts[-1] + 1 if cuts else segment_start
                    if (peak - previous + 1 >= min_length and
                            segment_end - peak >= min_length):
                        cuts.append(peak)
                begin = 0
                for cut in cuts:
                    end = segment.index(cut) + 1
                    action_segments.append(segment[begin:end])
                    begin = end
                action_segments.append(segment[begin:])

            for segment in action_segments:
                detections = _dense_detections({frame: observed[frame] for frame in segment})
                duration = max(detections) - min(detections) + 1
                if duration < min_length:
                    continue
                contributing_scores = [
                    item["score"] for item in tubelets
                    if set(item["detections"]).intersection(segment)
                ]
                segment_score = float(np.mean(contributing_scores))
                linked.append({
                    "video": tubelets[0]["video"],
                    "resolution": tubelets[0]["resolution"],
                    "class": class_id,
                    "score": segment_score,
                    "detections": detections,
                })
    return linked


def _maximum_weight_assignment(weights, valid):
    """Solve active-track/tubelet assignment, allowing unmatched rows."""
    rows, columns = weights.shape
    if not rows or not columns:
        return ()
    try:
        from scipy.optimize import linear_sum_assignment

        padded = np.zeros((rows, columns + rows), dtype=np.float64)
        padded[:, :columns] = np.where(valid, weights, -1e6)
        row_indices, column_indices = linear_sum_assignment(
            padded, maximize=True
        )
        return tuple(
            (int(row), int(column))
            for row, column in zip(row_indices, column_indices)
            if column < columns and valid[row, column]
        )
    except ImportError:
        edges = sorted(
            (
                (float(weights[row, column]), row, column)
                for row in range(rows) for column in range(columns)
                if valid[row, column]
            ),
            reverse=True,
        )
        used_rows, used_columns, pairs = set(), set(), []
        for _, row, column in edges:
            if row in used_rows or column in used_columns:
                continue
            used_rows.add(row)
            used_columns.add(column)
            pairs.append((row, column))
        return tuple(pairs)


def _consensus_boundary_cuts(tubelets, segment, clip_length, min_length,
                             threshold=0.5, radius=6, min_votes=2,
                             edge_window=8):
    """Find action transitions supported by tubelet end/start evidence.

    A single clip edge is not sufficient evidence: a cut needs both an ending
    and a starting endpoint vote (or additional nearby votes). This avoids the
    severe over-segmentation caused by thresholding every boundary logit.
    """
    if threshold is None or not segment:
        return []
    segment_set = set(segment)
    endpoints = []
    for tubelet_index, tubelet in enumerate(tubelets):
        frames = sorted(set(tubelet["detections"]) & segment_set)
        boundaries = tubelet.get("boundary_scores", {})
        if not frames or not boundaries:
            continue
        clip_start = int(tubelet["clip_start"])
        windows = []
        if frames[0] > clip_start + 2:
            windows.append(("start", frames[:edge_window]))
        if frames[-1] < clip_start + clip_length - 3:
            windows.append(("end", frames[-edge_window:]))
        for kind, window in windows:
            scored = [
                (float(boundaries.get(frame, 0.0)), frame) for frame in window
            ]
            score, frame = max(scored, default=(0.0, -1))
            if score >= threshold:
                endpoints.append((frame, kind, score, tubelet_index))

    proposals = []
    ends = [event for event in endpoints if event[1] == "end"]
    starts = [event for event in endpoints if event[1] == "start"]
    for end in ends:
        nearby_starts = [
            start for start in starts
            if start[3] != end[3] and abs(start[0] - end[0]) <= radius
        ]
        for start in nearby_starts:
            midpoint = int(round((end[0] + start[0]) / 2))
            center = min(segment, key=lambda frame: abs(frame - midpoint))
            nearby = [
                event for event in endpoints
                if abs(event[0] - center) <= radius
            ]
            unique_votes = {(event[1], event[3]) for event in nearby}
            if len(unique_votes) < min_votes:
                continue
            confidence = float(np.mean([event[2] for event in nearby]))
            proposals.append((confidence, center))

    cuts = []
    segment_start, segment_end = segment[0], segment[-1]
    for _, center in sorted(proposals, reverse=True):
        if center - segment_start + 1 < min_length:
            continue
        if segment_end - center < min_length:
            continue
        if any(abs(center - selected) < min_length for selected in cuts):
            continue
        cuts.append(center)
    return sorted(cuts)


def _render_global_track(tubelets, clip_length, min_length, split_gap,
                         boundary_split_thresh, boundary_radius,
                         boundary_votes, boundary_edge_window):
    boxes_by_frame = defaultdict(list)
    for tubelet in tubelets:
        for frame, box in tubelet["detections"].items():
            weight = float(tubelet.get("frame_weights", {}).get(frame, 1.0))
            weight *= max(float(tubelet["score"]), 1e-6)
            boxes_by_frame[frame].append((np.asarray(box, dtype=np.float32), weight))
    observed = {
        frame: (
            sum(box * weight for box, weight in boxes) /
            max(sum(weight for _, weight in boxes), 1e-12)
        ).astype(np.float32)
        for frame, boxes in boxes_by_frame.items()
    }
    observed_frames = sorted(observed)
    frame_segments = []
    if observed_frames:
        current = [observed_frames[0]]
        for frame in observed_frames[1:]:
            missing = frame - current[-1] - 1
            if split_gap is not None and missing > split_gap:
                frame_segments.append(current)
                current = []
            current.append(frame)
        frame_segments.append(current)

    action_segments = []
    for segment in frame_segments:
        cuts = _consensus_boundary_cuts(
            tubelets,
            segment,
            clip_length,
            min_length,
            threshold=boundary_split_thresh,
            radius=boundary_radius,
            min_votes=boundary_votes,
            edge_window=boundary_edge_window,
        )
        begin = 0
        for cut in cuts:
            end = segment.index(cut) + 1
            action_segments.append(segment[begin:end])
            begin = end
        action_segments.append(segment[begin:])

    rendered = []
    for segment in action_segments:
        detections = _dense_detections({frame: observed[frame] for frame in segment})
        if not detections:
            continue
        duration = max(detections) - min(detections) + 1
        if duration < min_length:
            continue
        contributors = [
            item for item in tubelets
            if set(item["detections"]).intersection(segment)
        ]
        score = float(np.mean([item["score"] for item in contributors]))
        rendered.append({
            "video": tubelets[0]["video"],
            "resolution": tubelets[0]["resolution"],
            "class": tubelets[0]["class"],
            "score": score,
            "detections": detections,
        })
    return rendered


def link_tubelets_global(candidates, clip_starts, clip_length, link_iou=0.5,
                         tubelet_nms=0.6, top_k=10, score_thresh=0.005,
                         min_length=15, split_gap=None,
                         boundary_split_thresh=None, boundary_radius=6,
                         boundary_votes=2, boundary_edge_window=8,
                         association_score_weight=0.05):
    """Link tubelets with joint assignment and consensus action boundaries.

    Unlike the public MOC greedy policy, this assigns all active actor tracks
    jointly at each overlapping clip. Box fusion uses predicted visibility,
    while action splitting requires agreement between ending and starting
    tubelets. The method remains causal at the tubelet-assignment stage.
    """
    candidates_by_class_start = defaultdict(list)
    classes = set()
    for candidate in candidates:
        key = (candidate["class"], candidate["clip_start"])
        candidates_by_class_start[key].append(candidate)
        classes.add(candidate["class"])

    linked = []
    for class_id in sorted(classes):
        active = []
        finished = []
        for clip_start in clip_starts:
            tubelets = _moc_tubelet_nms(
                candidates_by_class_start[(class_id, clip_start)],
                overlap_thresh=tubelet_nms,
                top_k=top_k,
            )
            if active and tubelets:
                overlaps = np.asarray([
                    [_mean_overlap_iou(track[-1], tubelet) for tubelet in tubelets]
                    for track in active
                ], dtype=np.float64)
                scores = np.asarray(
                    [tubelet["score"] for tubelet in tubelets], dtype=np.float64
                )[None, :]
                weights = overlaps + association_score_weight * scores
                pairs = _maximum_weight_assignment(weights, overlaps >= link_iou)
            else:
                pairs = ()
            matched_tracks = set()
            matched_tubelets = set()
            for track_index, tubelet_index in pairs:
                active[track_index].append(tubelets[tubelet_index])
                matched_tracks.add(track_index)
                matched_tubelets.add(tubelet_index)

            expired = [
                index for index, track in enumerate(active)
                if index not in matched_tracks and
                clip_start - track[-1]["clip_start"] >= clip_length
            ]
            for index in reversed(expired):
                finished.append(active.pop(index))
            active.extend([
                [tubelet] for index, tubelet in enumerate(tubelets)
                if index not in matched_tubelets
            ])
        finished.extend(active)

        for tubelets in finished:
            score = float(np.mean([item["score"] for item in tubelets]))
            if score < score_thresh:
                continue
            linked.extend(_render_global_track(
                tubelets,
                clip_length=clip_length,
                min_length=min_length,
                split_gap=split_gap,
                boundary_split_thresh=boundary_split_thresh,
                boundary_radius=boundary_radius,
                boundary_votes=boundary_votes,
                boundary_edge_window=boundary_edge_window,
            ))
    return linked


def _to_pixel_box(box, resolution):
    height, width = resolution
    scale = np.asarray([width, height, width, height], dtype=np.float64)
    return np.asarray(box, dtype=np.float64) * scale


def official_tube_iou_components(tube_a, tube_b):
    """Return temporal, mean spatial, and product IoU under MOC semantics."""
    detections_a = _dense_detections(tube_a["detections"])
    detections_b = _dense_detections(tube_b["detections"])
    if not detections_a or not detections_b:
        return 0.0, 0.0, 0.0
    begin = max(min(detections_a), min(detections_b))
    end = min(max(detections_a), max(detections_b))
    if end < begin:
        return 0.0, 0.0, 0.0
    temporal_intersection = end - begin + 1
    temporal_union = (
        max(max(detections_a), max(detections_b))
        - min(min(detections_a), min(detections_b)) + 1
    )
    resolution = tube_a.get("resolution", tube_b.get("resolution"))
    if resolution is None:
        raise ValueError("Official tube IoU requires the video resolution")
    spatial_ious = [
        _box_iou(
            _to_pixel_box(detections_a[frame], resolution),
            _to_pixel_box(detections_b[frame], resolution),
            inclusive=True,
        )
        for frame in range(begin, end + 1)
    ]
    temporal_iou = float(temporal_intersection / temporal_union)
    spatial_iou = float(np.mean(spatial_ious))
    return temporal_iou, spatial_iou, temporal_iou * spatial_iou


def official_tube_iou(tube_a, tube_b):
    """MOC iou3dt semantics for normalized dictionary tubes."""
    return official_tube_iou_components(tube_a, tube_b)[2]


def _tube_nms(tubes, overlap_thresh=0.3):
    if not tubes:
        return []
    order = np.argsort([tube["score"] for tube in tubes])
    keep = []
    while order.size:
        selected = int(order[-1])
        keep.append(tubes[selected])
        rest = order[:-1]
        ious = np.asarray([
            official_tube_iou(tubes[int(index)], tubes[selected]) for index in rest
        ])
        order = rest[np.where(ious <= overlap_thresh)[0]]
    return keep


def _moc_pr_to_ap(precision, recall):
    # ACT stores its PR table as float32; preserving that dtype makes the
    # public evaluator and this implementation bit-level comparable.
    precision = np.concatenate((
        np.asarray([1.0], dtype=np.float32),
        np.asarray(precision, dtype=np.float32),
    ))
    recall = np.concatenate((
        np.asarray([0.0], dtype=np.float32),
        np.asarray(recall, dtype=np.float32),
    ))
    return float(np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) * 0.5))


def compute_video_map_official(pred_tubes, gt_tubes,
                               iou_thresholds=(0.2, 0.5), num_classes=24,
                               tube_nms=0.3):
    """Compute video-mAP with the public MOC/YOWO UCF101-24 protocol."""
    for tube in list(pred_tubes) + list(gt_tubes):
        if "video" not in tube:
            raise ValueError("Official video mAP requires a video key on every tube")

    predictions_by_class = defaultdict(list)
    predictions_by_video_class = defaultdict(list)
    for prediction in pred_tubes:
        predictions_by_video_class[
            (prediction["video"], prediction["class"])
        ].append(prediction)
    for (_, class_id), predictions in predictions_by_video_class.items():
        predictions_by_class[class_id].extend(
            _tube_nms(predictions, overlap_thresh=tube_nms)
        )

    ground_truth_by_video_class = defaultdict(list)
    for ground_truth in gt_tubes:
        ground_truth_by_video_class[
            (ground_truth["video"], ground_truth["class"])
        ].append(ground_truth)

    results = {}
    for threshold in iou_thresholds:
        per_class = {}
        for class_id in range(num_classes):
            ground_truth_count = sum(
                len(tubes) for (video, label), tubes
                in ground_truth_by_video_class.items() if label == class_id
            )
            if not ground_truth_count:
                continue
            raw_predictions = predictions_by_class[class_id]
            # Match ACT's NumPy argsort, including its deterministic ordering
            # for tied scores. Python's stable sort changes one UCF24 match.
            order = np.argsort(-np.asarray([
                item["score"] for item in raw_predictions
            ]))
            predictions = [raw_predictions[int(index)] for index in order]
            matched = {
                key: np.zeros(len(tubes), dtype=bool)
                for key, tubes in ground_truth_by_video_class.items()
                if key[1] == class_id
            }
            true_positive = np.zeros(len(predictions), dtype=np.float64)
            false_positive = np.zeros(len(predictions), dtype=np.float64)
            for index, prediction in enumerate(predictions):
                key = (prediction["video"], class_id)
                candidates = ground_truth_by_video_class.get(key, [])
                available = np.where(~matched.get(
                    key, np.ones(len(candidates), dtype=bool)
                ))[0]
                if available.size:
                    ious = np.asarray([
                        official_tube_iou(prediction, candidates[int(gt_index)])
                        for gt_index in available
                    ])
                    best_local = int(np.argmax(ious))
                    if ious[best_local] >= threshold:
                        matched[key][int(available[best_local])] = True
                        true_positive[index] = 1.0
                        continue
                false_positive[index] = 1.0

            tp_cumulative = np.cumsum(true_positive)
            fp_cumulative = np.cumsum(false_positive)
            precision = tp_cumulative / np.maximum(
                tp_cumulative + fp_cumulative, 1e-12
            )
            recall = tp_cumulative / ground_truth_count
            per_class[class_id] = _moc_pr_to_ap(precision, recall)
        results[float(threshold)] = {
            "mAP": float(np.mean(list(per_class.values()))) if per_class else 0.0,
            "per_class": per_class,
        }
    return results
