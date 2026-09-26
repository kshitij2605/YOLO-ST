#!/usr/bin/env python3
"""Score YOLO-ST outputs with the official MultiSports protocol.

Inputs come from eval_tube_queries.py (optionally sharded):
  --frame-dumps  pickles written with --frame_dump (dense detections on every
                 frame, 1-indexed frames, pixel boxes)
  --caches       tube-candidate caches written with --candidate_cache

Tubes are linked per video with the MOC-style linker (defaults are the frozen
v2 UCF linker; recalibrate on MultiSports dev before test use), converted to
the MultiSports convention at this boundary only:
  * internal frames are 0-indexed; MultiSports GT frames are 1-indexed (+1),
  * missing frames inside a tube are filled by linear interpolation, because
    the official 3D IoU requires identical frame indices,
  * boxes are converted to float pixels.
Frame AP identifies videos by index into GT['test_videos'][0]; video AP by name.

    python3 tools/export_multisports.py --gt data/multisports/dev_splits/multisports_dev_GT.pkl \
        --frame-dumps OUT/frames_shard*.pkl --caches OUT/tubes_shard*.pkl --out OUT/multisports_metrics.json
"""

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_multisports  # noqa: E402
from video_map_protocol import link_tubelets_moc  # noqa: E402

FROZEN_LINKER = dict(link_iou=0.45, tubelet_nms=0.6, top_k=10, min_length=16, split_gap=2)


def frame_rows_to_detections(rows, video_list):
    """(video, frame, class, score, x1, y1, x2, y2) rows -> official (N, 8) array."""
    index = {name: i for i, name in enumerate(video_list)}
    out = [[index[r[0]], r[1], r[2], r[3], r[4], r[5], r[6], r[7]] for r in rows if r[0] in index]
    return np.asarray(out, dtype=np.float64).reshape(-1, 8)


def fill_tube_gaps(detections):
    """Linearly interpolate missing integer frames between a tube's first and last frame."""
    frames = sorted(int(f) for f in detections)
    if not frames:
        return {}
    known = np.asarray(frames, dtype=np.float64)
    boxes = np.asarray([detections[f] for f in frames], dtype=np.float64)
    full = np.arange(frames[0], frames[-1] + 1)
    filled = np.stack([np.interp(full, known, boxes[:, k]) for k in range(4)], axis=1)
    return {int(f): filled[i] for i, f in enumerate(full)}


def to_multisports_tube(tube):
    """Internal tube (0-indexed frames, normalised boxes) -> +1 frames, gap-filled."""
    filled = fill_tube_gaps(tube['detections'])
    return dict(tube, detections={frame + 1: box for frame, box in filled.items()})


def link_caches(caches, linker):
    tubes = []
    for cache in caches:
        for video_data in cache['videos'].values():
            tubes.extend(link_tubelets_moc(
                video_data['candidates'], video_data['clip_starts'], cache['clip_length'],
                link_iou=linker['link_iou'], tubelet_nms=linker['tubelet_nms'],
                top_k=linker['top_k'], min_length=linker['min_length'],
                split_gap=linker['split_gap'] if linker['split_gap'] >= 0 else None,
            ))
    return tubes


def _load_all(patterns):
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    loaded = []
    for path in paths:
        with open(path, 'rb') as handle:
            loaded.append(pickle.load(handle))
    return paths, loaded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--gt', required=True)
    parser.add_argument('--frame-dumps', nargs='*', default=[])
    parser.add_argument('--caches', nargs='*', default=[])
    parser.add_argument('--out', required=True)
    for key, value in FROZEN_LINKER.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
    args = parser.parse_args(argv)

    groundtruth = eval_multisports.load_groundtruth(args.gt)
    video_list = groundtruth['test_videos'][0]
    result = {'gt': args.gt, 'videos': len(video_list)}

    frame_detections = None
    if args.frame_dumps:
        paths, dumps = _load_all(args.frame_dumps)
        rows = [row for dump in dumps for row in dump['rows']]
        frame_detections = frame_rows_to_detections(rows, video_list)
        result.update(frame_dumps=paths, frame_rows=int(frame_detections.shape[0]))

    video_detections = None
    if args.caches:
        paths, caches = _load_all(args.caches)
        linker = {k: getattr(args, k) for k in FROZEN_LINKER}
        tubes = [to_multisports_tube(t) for t in link_caches(caches, linker)]
        video_detections = eval_multisports.tubes_to_video_detections(tubes, video_list)
        result.update(caches=paths, linker=linker, tubes=len(tubes))

    metrics = eval_multisports.evaluate_multisports(
        groundtruth, frame_detections=frame_detections, video_detections=video_detections,
        video_thresholds=(0.2, 0.5), include_all_ranges=video_detections is not None,
    )
    result['metrics'] = {k: v for k, v in metrics.items() if not k.endswith('per_class')}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(result, handle, indent=2, default=float)
    for key, value in result['metrics'].items():
        if isinstance(value, (int, float)):
            print(f'{key}: {value:.2f}' if isinstance(value, float) else f'{key}: {value}')
    return result


if __name__ == '__main__':
    main()
