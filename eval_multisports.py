"""Native MultiSports evaluation for YOLO-ST.

This is a faithful reimplementation of the official
``MCG-NJU/MultiSports/evaluate_multisports.py`` conventions. It is deliberately
**not** built on ``video_map_protocol.py``: the MultiSports protocol differs
from the MOC/YOWO UCF101-24 protocol in four ways that change the numbers.

======================  ==================================  ===================
Aspect                  UCF101-24 (MOC/ACT, this repo)      MultiSports
======================  ==================================  ===================
Box area                inclusive, ``(x2-x1+1)(y2-y1+1)``   VOC, ``(x2-x1)(y2-y1)``
Temporal extent         inclusive, ``tmax-tmin+1``          ``tmax-tmin``
Tube NMS                applied, threshold 0.30             **not applied**
AP integration          trapezoidal over the raw PR curve   VOC envelope
Evaluated classes       all 24                              60 of 66
======================  ==================================  ===================

Six classes are excluded from evaluation by the official script because their
annotations are unreliable: ``aerobic kick jump``, ``aerobic off axis jump``,
``aerobic butterfly jump``, ``aerobic balance turn``, ``basketball save`` and
``basketball jump ball``.

Reported metrics
----------------
* ``frameAP@0.5``
* ``videoAP`` at a single threshold, conventionally 0.2 and 0.5
* ``videoAP_all``, the mean over three official threshold ranges:
  ``low`` 0.05:0.45 step 0.05, ``all`` 0.10:0.90 step 0.10, and
  ``high`` 0.50:0.95 step 0.05. Published MultiSports tables quote the
  ``all`` (0.1:0.9) column, **not** 0.5:0.95. Do not mix them.

Detection formats follow the official script:

* frame detections: array ``(N, 8)`` of
  ``[video_index, frame_id, label, score, x1, y1, x2, y2]`` in pixels
* video detections: mapping ``label -> [(video_name, score, tube), ...]``
  where ``tube`` is ``(T, 5)`` of ``[frame_id, x1, y1, x2, y2]`` in pixels

Note the asymmetry, which is in the official script and is easy to get wrong:
``frameAP`` identifies a video by its **index** into ``GT['test_videos'][0]``,
while ``videoAP`` identifies it by its **name**.
"""

import argparse
import json
import os
import pickle

import numpy as np


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GT = os.path.join(
    REPO_ROOT, 'data', 'multisports', 'hf', 'data', 'trainval',
    'multisports_GT.pkl',
)

#: Classes the official evaluator skips.
EXCLUDED_LABELS = (
    'aerobic kick jump',
    'aerobic off axis jump',
    'aerobic butterfly jump',
    'aerobic balance turn',
    'basketball save',
    'basketball jump ball',
)

#: Official threshold ranges used by ``videoAP_all``.
THRESHOLD_RANGES = {
    'low': np.arange(0.05, 0.50, 0.05),
    'all': np.arange(0.10, 0.95, 0.10),
    'high': np.arange(0.50, 1.00, 0.05),
}


# ---------------------------------------------------------------------------
# Geometry, matching the official *_voc helpers exactly.
# ---------------------------------------------------------------------------

def area2d_voc(boxes):
    """Areas of ``(N, 4)`` xyxy boxes, VOC convention (no ``+1``)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def overlap2d_voc(boxes, box):
    """Intersection areas between ``(N, 4)`` boxes and one ``(4,)`` box."""
    boxes = np.asarray(boxes, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64).reshape(-1, 4)
    xmin = np.maximum(boxes[:, 0], box[:, 0])
    ymin = np.maximum(boxes[:, 1], box[:, 1])
    xmax = np.minimum(boxes[:, 2], box[:, 2])
    ymax = np.minimum(boxes[:, 3], box[:, 3])
    width = np.maximum(0.0, xmax - xmin)
    height = np.maximum(0.0, ymax - ymin)
    return width * height


def iou2d_voc(boxes, box):
    """IoU between ``(N, 4)`` boxes and one ``(4,)`` box."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float64).reshape(-1, 4)
    overlap = overlap2d_voc(boxes, box)
    return overlap / (area2d_voc(boxes) + area2d_voc(box) - overlap)


def iou3d_voc(tube_a, tube_b):
    """Mean per-frame IoU of two tubes covering identical frames."""
    tube_a = np.asarray(tube_a, dtype=np.float64)
    tube_b = np.asarray(tube_b, dtype=np.float64)
    if tube_a.shape[0] != tube_b.shape[0]:
        raise ValueError('iou3d_voc requires equal temporal extent')
    if not np.all(tube_a[:, 0] == tube_b[:, 0]):
        raise ValueError('iou3d_voc requires identical frame indices')
    overlap = np.empty(tube_a.shape[0], dtype=np.float64)
    for index in range(tube_a.shape[0]):
        overlap[index] = overlap2d_voc(
            tube_a[index:index + 1, 1:5], tube_b[index, 1:5]
        )[0]
    return float(np.mean(
        overlap / (area2d_voc(tube_a[:, 1:5]) + area2d_voc(tube_b[:, 1:5])
                   - overlap)
    ))


def iou3dt_voc(tube_a, tube_b, spatialonly=False, temporalonly=False):
    """Spatio-temporal IoU: mean spatial IoU times temporal IoU.

    Temporal extent is ``tmax - tmin`` with no ``+1``, unlike the MOC
    convention used for UCF101-24 in :mod:`video_map_protocol`.
    """
    tube_a = np.asarray(tube_a, dtype=np.float64)
    tube_b = np.asarray(tube_b, dtype=np.float64)
    tmin = max(tube_a[0, 0], tube_b[0, 0])
    tmax = min(tube_a[-1, 0], tube_b[-1, 0])
    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin
    temporal_union = (max(tube_a[-1, 0], tube_b[-1, 0])
                      - min(tube_a[0, 0], tube_b[0, 0]))
    if temporalonly:
        return float(temporal_inter / temporal_union) if temporal_union else 0.0

    start_a = int(np.where(tube_a[:, 0] == tmin)[0][0])
    end_a = int(np.where(tube_a[:, 0] == tmax)[0][0])
    start_b = int(np.where(tube_b[:, 0] == tmin)[0][0])
    end_b = int(np.where(tube_b[:, 0] == tmax)[0][0])
    spatial = iou3d_voc(
        tube_a[start_a:end_a + 1, :], tube_b[start_b:end_b + 1, :]
    )
    if spatialonly:
        return float(spatial)
    if temporal_union == 0:
        return 0.0
    return float(spatial * temporal_inter / temporal_union)


def pr_to_ap_voc(pr):
    """VOC average precision from an ``(N, 2)`` ``[precision, recall]`` table."""
    pr = np.asarray(pr, dtype=np.float64)
    precision = np.concatenate([[0.0], pr[:, 0], [0.0]])
    recall = np.concatenate([[0.0], pr[:, 1], [1.0]])
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = np.maximum(precision[index], precision[index + 1])
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices]
    ))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_groundtruth(path=DEFAULT_GT):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')


def evaluated_labels(groundtruth):
    """Return ``[(label_index, label_name), ...]`` excluding the skipped six."""
    return [
        (index, name)
        for index, name in enumerate(groundtruth['labels'])
        if name not in EXCLUDED_LABELS
    ]


def frame_ap(groundtruth, detections, thr=0.5, videos=None):
    """Frame mAP at IoU ``thr``.

    Args:
        groundtruth: the MultiSports GT dictionary.
        detections: ``(N, 8)`` array of
            ``[video_index, frame_id, label, score, x1, y1, x2, y2]``.
        thr: IoU threshold, official default 0.5.
        videos: optional video list; defaults to ``GT['test_videos'][0]``.

    Returns:
        ``(mean_ap, {label_name: ap})`` with values in percent.
    """
    video_list = videos if videos is not None \
        else groundtruth['test_videos'][0]
    detections = np.asarray(detections, dtype=np.float64).reshape(-1, 8)

    per_class = {}
    for ilabel, label in evaluated_labels(groundtruth):
        class_detections = detections[detections[:, 2] == ilabel, :]

        gt = {}
        for video_index, video in enumerate(video_list):
            tubes = groundtruth['gttubes'].get(video, {})
            if ilabel not in tubes:
                continue
            for tube in tubes[ilabel]:
                for row in range(tube.shape[0]):
                    key = (video_index, int(tube[row, 0]))
                    gt.setdefault(key, []).append(tube[row, 1:5].tolist())
        gt = {key: np.asarray(value) for key, value in gt.items()}
        gt_num = sum(boxes.shape[0] for boxes in gt.values())
        if gt_num == 0:
            continue

        pr = np.empty((class_detections.shape[0], 2), dtype=np.float64)
        true_positive = 0
        false_positive = 0
        detected = {}
        order = np.argsort(-class_detections[:, 3])
        for rank, index in enumerate(order):
            key = (int(class_detections[index, 0]),
                   int(class_detections[index, 1]))
            box = class_detections[index, 4:8]
            positive = False
            if key in gt:
                if key not in detected:
                    detected[key] = np.zeros(gt[key].shape[0], dtype=bool)
                ious = iou2d_voc(gt[key], box)
                best = int(np.argmax(ious))
                if ious[best] >= thr and not detected[key][best]:
                    positive = True
                    detected[key][best] = True
            if positive:
                true_positive += 1
            else:
                false_positive += 1
            pr[rank, 0] = true_positive / float(true_positive + false_positive)
            pr[rank, 1] = true_positive / float(gt_num)

        per_class[label] = 100.0 * pr_to_ap_voc(pr)

    mean_ap = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_ap, per_class


def video_ap(groundtruth, detections, thr=0.5, videos=None):
    """Video mAP at spatio-temporal IoU ``thr``.

    Args:
        groundtruth: the MultiSports GT dictionary.
        detections: mapping ``label_index -> [(video_name, score, tube), ...]``
            with ``tube`` shaped ``(T, 5)`` as ``[frame, x1, y1, x2, y2]``.
            Videos are named here, not indexed; this matches the official
            ``videoAP`` and differs from ``frameAP``.
        thr: tube IoU threshold.
        videos: optional video list; defaults to ``GT['test_videos'][0]``.

    Returns:
        ``(mean_ap, {label_name: ap})`` with values in percent.
    """
    video_list = videos if videos is not None \
        else groundtruth['test_videos'][0]

    per_class = {}
    for ilabel, label in evaluated_labels(groundtruth):
        class_detections = list(detections.get(ilabel, []))

        gt = {}
        for video in video_list:
            tubes = groundtruth['gttubes'].get(video, {})
            if ilabel not in tubes or len(tubes[ilabel]) == 0:
                continue
            gt[video] = list(tubes[ilabel])
        gt_num = sum(len(tubes) for tubes in gt.values())
        if gt_num == 0:
            continue

        pr = np.empty((len(class_detections), 2), dtype=np.float64)
        true_positive = 0
        false_positive = 0
        detected = {}
        scores = np.asarray(
            [item[1] for item in class_detections], dtype=np.float64
        ) if class_detections else np.zeros(0)
        order = np.argsort(-scores) if class_detections else []
        for rank, index in enumerate(order):
            video, _, tube = class_detections[int(index)]
            positive = False
            if video in gt:
                if video not in detected:
                    detected[video] = np.zeros(len(gt[video]), dtype=bool)
                ious = np.asarray([
                    iou3dt_voc(np.asarray(candidate), np.asarray(tube))
                    for candidate in gt[video]
                ])
                best = int(np.argmax(ious))
                if ious[best] >= thr and not detected[video][best]:
                    positive = True
                    detected[video][best] = True
            if positive:
                true_positive += 1
            else:
                false_positive += 1
            pr[rank, 0] = true_positive / float(true_positive + false_positive)
            pr[rank, 1] = true_positive / float(gt_num)

        per_class[label] = 100.0 * pr_to_ap_voc(pr)

    mean_ap = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return mean_ap, per_class


def video_ap_all(groundtruth, detections, videos=None):
    """Video mAP averaged over the three official threshold ranges."""
    results = {}
    for name, thresholds in THRESHOLD_RANGES.items():
        values = []
        for threshold in thresholds:
            mean_ap, _ = video_ap(
                groundtruth, detections, thr=float(threshold), videos=videos
            )
            values.append(mean_ap)
            results[f'videoAP@{threshold:.2f}'] = mean_ap
        results[f'videoAP_{name}'] = float(np.mean(values))
    return results


def evaluate_multisports(groundtruth, frame_detections=None,
                         video_detections=None, frame_thr=0.5,
                         video_thresholds=(0.2, 0.5), videos=None,
                         include_all_ranges=False, verbose=False):
    """Run the MultiSports protocol and return a metrics dictionary."""
    if isinstance(groundtruth, str):
        groundtruth = load_groundtruth(groundtruth)

    metrics = {
        'num_labels_total': len(groundtruth['labels']),
        'num_labels_evaluated': len(evaluated_labels(groundtruth)),
        'excluded_labels': list(EXCLUDED_LABELS),
    }

    if frame_detections is not None:
        mean_ap, per_class = frame_ap(
            groundtruth, frame_detections, thr=frame_thr, videos=videos
        )
        metrics[f'frameAP@{frame_thr}'] = mean_ap
        metrics['frameAP_per_class'] = per_class

    if video_detections is not None:
        for threshold in video_thresholds:
            mean_ap, per_class = video_ap(
                groundtruth, video_detections, thr=float(threshold),
                videos=videos,
            )
            metrics[f'videoAP@{threshold}'] = mean_ap
            metrics[f'videoAP@{threshold}_per_class'] = per_class
        if include_all_ranges:
            metrics.update(
                video_ap_all(groundtruth, video_detections, videos=videos)
            )

    if verbose:
        for key in sorted(metrics):
            if key.endswith('per_class') or key == 'excluded_labels':
                continue
            value = metrics[key]
            if isinstance(value, float):
                print(f'{key:24s} {value:8.2f}')
            else:
                print(f'{key:24s} {value}')
    return metrics


# ---------------------------------------------------------------------------
# Converters from the repository's internal tube format.
# ---------------------------------------------------------------------------

def tubes_to_video_detections(tubes, video_list):
    """Convert internal tube dicts to the official video-detection mapping.

    Internal tubes use ``{'video', 'class', 'score', 'detections', 'resolution'}``
    with normalised boxes keyed by frame id, as produced by
    :mod:`video_map_protocol`. MultiSports expects pixel coordinates and, for
    video AP, video names rather than indices.
    """
    allowed = set(video_list)
    detections = {}
    for tube in tubes:
        video = tube['video']
        if video not in allowed:
            continue
        resolution = tube['resolution']
        height, width = resolution
        frames = sorted(tube['detections'])
        rows = []
        for frame in frames:
            box = np.asarray(tube['detections'][frame], dtype=np.float64)
            rows.append([
                float(frame),
                box[0] * width, box[1] * height,
                box[2] * width, box[3] * height,
            ])
        detections.setdefault(int(tube['class']), []).append(
            (video, float(tube['score']), np.asarray(rows))
        )
    return detections


def groundtruth_as_frame_detections(groundtruth, videos=None, score=1.0):
    """Replay ground truth as perfect frame detections. Used for parity tests."""
    video_list = videos if videos is not None \
        else groundtruth['test_videos'][0]
    rows = []
    for video_index, video in enumerate(video_list):
        for ilabel, tubes in groundtruth['gttubes'].get(video, {}).items():
            for tube in tubes:
                for row in tube:
                    rows.append([
                        video_index, float(row[0]), float(ilabel), score,
                        float(row[1]), float(row[2]),
                        float(row[3]), float(row[4]),
                    ])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def groundtruth_as_video_detections(groundtruth, videos=None, score=1.0):
    """Replay ground truth as perfect tube detections. Used for parity tests."""
    video_list = videos if videos is not None \
        else groundtruth['test_videos'][0]
    detections = {}
    for video in video_list:
        for ilabel, tubes in groundtruth['gttubes'].get(video, {}).items():
            for tube in tubes:
                detections.setdefault(int(ilabel), []).append(
                    (video, float(score), np.asarray(tube, dtype=np.float64))
                )
    return detections


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate MultiSports with the official protocol.'
    )
    parser.add_argument('--groundtruth', default=DEFAULT_GT)
    parser.add_argument(
        '--frame-detections', default=None,
        help='pickle holding an (N, 8) frame detection array',
    )
    parser.add_argument(
        '--video-detections', default=None,
        help='pickle holding {label: [(video_index, score, tube), ...]}',
    )
    parser.add_argument('--frame-thr', type=float, default=0.5)
    parser.add_argument(
        '--video-thr', type=float, nargs='*', default=[0.2, 0.5]
    )
    parser.add_argument('--all-ranges', action='store_true')
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args(argv)

    groundtruth = load_groundtruth(args.groundtruth)
    frame_detections = None
    if args.frame_detections:
        with open(args.frame_detections, 'rb') as handle:
            frame_detections = pickle.load(handle)
    video_detections = None
    if args.video_detections:
        with open(args.video_detections, 'rb') as handle:
            video_detections = pickle.load(handle)

    metrics = evaluate_multisports(
        groundtruth,
        frame_detections=frame_detections,
        video_detections=video_detections,
        frame_thr=args.frame_thr,
        video_thresholds=args.video_thr,
        include_all_ranges=args.all_ranges,
        verbose=True,
    )
    if args.json_out:
        with open(args.json_out, 'w') as handle:
            json.dump(
                {key: value for key, value in metrics.items()
                 if not key.endswith('per_class')},
                handle, indent=2, sort_keys=True,
            )
        print(f'wrote {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
