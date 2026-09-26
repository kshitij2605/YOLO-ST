"""Native AVA v2.2 evaluation for YOLO-ST.

The AP computation is the official ActivityNet/AVA evaluation code, vendored
untouched at ``references/YOWOv3/evaluator/Evaluation``. This module loads that
code by file path (the vendored tree has no ``__init__.py``, so it cannot be
imported normally) and adds only three things the vendored copy is missing:

1. **Excluded timestamps.** The upstream Google script takes an ``--exclusions``
   file and drops those ``video_id,timestamp`` keys from both ground truth and
   detections. The vendored copy dropped that argument. AVA v2.2 validation has
   37 excluded timestamps, so leaving them in changes the reported mAP.
2. **A return value.** ``run_evaluation`` only pretty-prints. This module
   returns the metrics dictionary so results can be logged, compared and
   registered.
3. **Writing detections.** ``write_ava_detections`` serialises model output into
   the AVA CSV format the evaluator expects.

Protocol notes
--------------
* The evaluated label set is the 60 classes in
  ``ava_action_list_v2.2_for_activitynet_2019.pbtxt``. Label ids run 1..80 with
  gaps; 60 of them are evaluated. Our model head is 0-indexed over the 60
  valid classes, so :func:`model_index_to_ava_id` converts.
* Boxes are normalised ``[x1, y1, x2, y2]`` in ``[0, 1]``, matching the AVA CSV.
  The official reader converts to ``[y1, x1, y2, x2]`` internally.
* Detections are capped at 50 boxes per keyframe, which is the official
  ``capacity`` used by ``run_evaluation``.
* The headline number is ``PascalBoxes_Precision/mAP@0.5IOU``.

Usage::

    python eval_ava.py \
        --groundtruth data/ava/annotations/ava_val_v2.2.csv \
        --detections  experiments/<run>/ava_val_detections.csv \
        --labelmap    data/ava/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt \
        --exclusions  data/ava/annotations/ava_val_excluded_timestamps_v2.2.csv
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import types

import numpy as np


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_EVAL_DIR = os.path.join(
    REPO_ROOT, 'references', 'YOWOv3', 'evaluator', 'Evaluation'
)

DEFAULT_LABELMAP = os.path.join(
    REPO_ROOT, 'data', 'ava', 'annotations',
    'ava_action_list_v2.2_for_activitynet_2019.pbtxt',
)
DEFAULT_GROUNDTRUTH = os.path.join(
    REPO_ROOT, 'data', 'ava', 'annotations', 'ava_val_v2.2.csv'
)
DEFAULT_EXCLUSIONS = os.path.join(
    REPO_ROOT, 'data', 'ava', 'annotations',
    'ava_val_excluded_timestamps_v2.2.csv',
)

MAP_METRIC = 'PascalBoxes_Precision/mAP@0.5IOU'
PER_CLASS_PREFIX = 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/'

#: Official capacity used by the upstream evaluator for detections.
DETECTION_CAPACITY = 50

_official_cache = {}


def load_official(eval_dir=OFFICIAL_EVAL_DIR):
    """Import the vendored official AVA evaluation code without modifying it.

    Returns a tuple ``(get_ava_performance, object_detection_evaluation,
    standard_fields)``.
    """
    eval_dir = os.path.abspath(eval_dir)
    if eval_dir in _official_cache:
        return _official_cache[eval_dir]
    if not os.path.isdir(os.path.join(eval_dir, 'ava')):
        raise FileNotFoundError(
            f'official AVA evaluation code not found under {eval_dir}'
        )

    package_name = '_ava_official'
    package = sys.modules.get(package_name)
    if package is None or getattr(package, '__path__', None) != [eval_dir]:
        package = types.ModuleType(package_name)
        package.__path__ = [eval_dir]
        sys.modules[package_name] = package

    def _load(rel_name, filename):
        full_name = f'{package_name}.{rel_name}'
        if full_name in sys.modules:
            return sys.modules[full_name]
        spec = importlib.util.spec_from_file_location(
            full_name, os.path.join(eval_dir, filename)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    get_ava_performance = _load(
        'get_ava_performance', 'get_ava_performance.py'
    )
    ava_package = importlib.import_module(f'{package_name}.ava')
    object_detection_evaluation = importlib.import_module(
        f'{package_name}.ava.object_detection_evaluation'
    )
    standard_fields = importlib.import_module(
        f'{package_name}.ava.standard_fields'
    )
    del ava_package
    result = (
        get_ava_performance, object_detection_evaluation, standard_fields
    )
    _official_cache[eval_dir] = result
    return result


def _as_file(path_or_file, mode='r'):
    if hasattr(path_or_file, 'read'):
        return path_or_file, False
    return open(path_or_file, mode), True


def read_exclusions(path_or_file):
    """Return the set of ``video_id,timestamp`` keys to exclude.

    Mirrors the upstream ``read_exclusions``: a two-column CSV of
    ``video_id,timestamp``. An empty or missing file excludes nothing.
    """
    excluded = set()
    if path_or_file is None:
        return excluded
    if isinstance(path_or_file, str) and not os.path.isfile(path_or_file):
        raise FileNotFoundError(path_or_file)
    handle, should_close = _as_file(path_or_file)
    try:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f'expected only 2 columns, got: {row}')
            excluded.add(make_image_key(row[0], row[1]))
    finally:
        if should_close:
            handle.close()
    return excluded


def make_image_key(video_id, timestamp):
    """Official keyframe identifier, ``"<video_id>,<timestamp:.6f>"``."""
    get_ava_performance = load_official()[0]
    return get_ava_performance.make_image_key(video_id, timestamp)


def read_labelmap(path_or_file=DEFAULT_LABELMAP):
    """Return ``(categories, class_ids)`` using the official parser."""
    get_ava_performance = load_official()[0]
    handle, should_close = _as_file(path_or_file)
    try:
        return get_ava_performance.read_labelmap(handle)
    finally:
        if should_close:
            handle.close()


def model_index_to_ava_id(labelmap=None):
    """Map a 0-indexed model class to its AVA label id.

    ``data/ava_dataset.py`` compacts the 60 evaluated AVA classes into
    ``0..59``; the evaluator expects the original sparse ids.
    """
    categories = labelmap if labelmap is not None else read_labelmap()[0]
    return [entry['id'] for entry in categories]


def evaluate_ava(
    groundtruth=DEFAULT_GROUNDTRUTH,
    detections=None,
    labelmap=DEFAULT_LABELMAP,
    exclusions=DEFAULT_EXCLUSIONS,
    capacity=DETECTION_CAPACITY,
    verbose=False,
):
    """Evaluate AVA detections and return the official metrics dictionary.

    Args:
        groundtruth: path or file object, AVA ground-truth CSV.
        detections: path or file object, AVA detection CSV
            (``video_id,timestamp,x1,y1,x2,y2,action_id,score``).
        labelmap: path or file object, the 60-class pbtxt.
        exclusions: path or file object of ``video_id,timestamp`` rows to drop
            from both ground truth and detections. Pass ``None`` to disable.
        capacity: maximum detections retained per keyframe (official: 50).
        verbose: print progress and the metrics dictionary.

    Returns:
        dict with the official metric names, plus the convenience keys
        ``mAP@0.5IOU``, ``num_classes``, ``num_groundtruth_keys`` and
        ``num_excluded_keys``.
    """
    if detections is None:
        raise ValueError('detections is required')

    get_ava_performance, object_detection_evaluation, standard_fields = (
        load_official()
    )

    categories, class_whitelist = read_labelmap(labelmap)
    excluded_keys = read_exclusions(exclusions)

    evaluator = object_detection_evaluation.PascalDetectionEvaluator(categories)

    gt_handle, close_gt = _as_file(groundtruth)
    try:
        boxes, labels, _, included_keys = get_ava_performance.read_csv(
            gt_handle, class_whitelist, 0
        )
    finally:
        if close_gt:
            gt_handle.close()

    num_gt_keys = 0
    for image_key in boxes:
        if image_key in excluded_keys:
            continue
        num_gt_keys += 1
        evaluator.add_single_ground_truth_image_info(
            image_key,
            {
                standard_fields.InputDataFields.groundtruth_boxes:
                    np.array(boxes[image_key], dtype=float),
                standard_fields.InputDataFields.groundtruth_classes:
                    np.array(labels[image_key], dtype=int),
                standard_fields.InputDataFields.groundtruth_difficult:
                    np.zeros(len(boxes[image_key]), dtype=bool),
            },
        )

    det_handle, close_det = _as_file(detections)
    try:
        boxes, labels, scores, _ = get_ava_performance.read_csv(
            det_handle, class_whitelist, capacity
        )
    finally:
        if close_det:
            det_handle.close()

    num_det_keys = 0
    for image_key in boxes:
        if image_key in excluded_keys:
            continue
        if image_key not in included_keys:
            # Official behaviour: detections for keyframes with no ground truth
            # entry are ignored rather than counted as false positives.
            continue
        num_det_keys += 1
        evaluator.add_single_detected_image_info(
            image_key,
            {
                standard_fields.DetectionResultFields.detection_boxes:
                    np.array(boxes[image_key], dtype=float),
                standard_fields.DetectionResultFields.detection_classes:
                    np.array(labels[image_key], dtype=int),
                standard_fields.DetectionResultFields.detection_scores:
                    np.array(scores[image_key], dtype=float),
            },
        )

    metrics = dict(evaluator.evaluate())
    metrics['mAP@0.5IOU'] = float(metrics[MAP_METRIC])
    metrics['num_classes'] = len(categories)
    metrics['num_groundtruth_keys'] = num_gt_keys
    metrics['num_detection_keys'] = num_det_keys
    metrics['num_excluded_keys'] = len(excluded_keys)

    if verbose:
        print(f'classes evaluated       : {len(categories)}')
        print(f'ground-truth keyframes  : {num_gt_keys}')
        print(f'detection keyframes     : {num_det_keys}')
        print(f'excluded timestamps     : {len(excluded_keys)}')
        print(f'mAP@0.5IOU              : {100.0 * metrics["mAP@0.5IOU"]:.2f}')
    return metrics


def per_class_ap(metrics):
    """Extract ``{class_name: AP}`` from an :func:`evaluate_ava` result."""
    return {
        key[len(PER_CLASS_PREFIX):]: float(value)
        for key, value in metrics.items()
        if key.startswith(PER_CLASS_PREFIX)
    }


def write_ava_detections(path, records, labelmap=None):
    """Write detections in the AVA CSV format.

    Args:
        path: output CSV path.
        records: iterable of
            ``(video_id, timestamp, x1, y1, x2, y2, class_index, score)``.
            Boxes are normalised to ``[0, 1]``. ``class_index`` is a 0-indexed
            model class when ``labelmap`` is given, otherwise a raw AVA id.
        labelmap: optional list of categories from :func:`read_labelmap`; when
            given, ``class_index`` is mapped through
            :func:`model_index_to_ava_id`.

    Returns:
        Number of rows written.
    """
    ids = model_index_to_ava_id(labelmap) if labelmap is not None else None
    written = 0
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        for record in records:
            video_id, timestamp, x1, y1, x2, y2, class_index, score = record
            action_id = ids[int(class_index)] if ids else int(class_index)
            writer.writerow([
                video_id,
                f'{int(timestamp):04d}',
                f'{float(x1):.6f}', f'{float(y1):.6f}',
                f'{float(x2):.6f}', f'{float(y2):.6f}',
                action_id,
                f'{float(score):.6f}',
            ])
            written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate AVA v2.2 detections with the official code.'
    )
    parser.add_argument('--groundtruth', default=DEFAULT_GROUNDTRUTH)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--labelmap', default=DEFAULT_LABELMAP)
    parser.add_argument(
        '--exclusions', default=DEFAULT_EXCLUSIONS,
        help='pass "none" to evaluate without excluded timestamps',
    )
    parser.add_argument('--capacity', type=int, default=DETECTION_CAPACITY)
    parser.add_argument(
        '--json-out', default=None, help='write the metrics dict here'
    )
    args = parser.parse_args(argv)

    exclusions = None if str(args.exclusions).lower() == 'none' \
        else args.exclusions
    metrics = evaluate_ava(
        groundtruth=args.groundtruth,
        detections=args.detections,
        labelmap=args.labelmap,
        exclusions=exclusions,
        capacity=args.capacity,
        verbose=True,
    )
    print()
    for name, value in sorted(per_class_ap(metrics).items()):
        print(f'{100.0 * value:6.2f}  {name}')
    print()
    print(f'AVA v2.2 mAP@0.5 = {100.0 * metrics["mAP@0.5IOU"]:.2f}')

    if args.json_out:
        with open(args.json_out, 'w') as handle:
            json.dump(
                {key: (float(value) if isinstance(value, (int, float, np.floating))
                       else value)
                 for key, value in metrics.items()},
                handle, indent=2, sort_keys=True,
            )
        print(f'wrote {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
