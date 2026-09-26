#!/usr/bin/env python3
"""Build action tubes by linking dense per-frame detections (CPU only).

For crowded MultiSports scenes the tube-query decoder leaves most actors
without a tube. This alternative tube source links the dense head's per-frame
detections (a --frame_dump pickle from eval_tube_queries.py) with a greedy
online linker per class, in the spirit of Singh et al. (ROAD) / ACT:

  * each frame keeps the top --per-frame detections of each class after NMS,
  * active tubes (sorted by mean score) claim the best unclaimed detection in
    the next frame with IoU >= --link-iou,
  * a tube may miss up to --max-gap frames; missed frames are interpolated,
  * unclaimed detections start new tubes; tubes shorter than --min-length are
    dropped; tube score = mean of its top --score-top-k detection scores.

Output tubes use the internal convention of video_map_protocol (0-indexed
frames, normalised boxes) so tools/export_multisports.py scores them exactly
like query tubes. Frame dumps are 1-indexed pixel rows; they are converted
here.

    python3 tools/link_dense_frame_tubes.py --gt data/multisports/dev_splits/multisports_dev_GT.pkl \
        --frame-dump OUT/frames_ema.pkl --out OUT/dense_link_metrics.json
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_multisports as ems  # noqa: E402
from tools.export_multisports import to_multisports_tube  # noqa: E402


def iou(box, boxes):
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=1)
    area = (box[2:] - box[:2]).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return inter / np.maximum(area + areas - inter, 1e-9)


def link_class(frames, link_iou, max_gap, min_length, score_top_k):
    """frames: {frame: (boxes (N,4), scores (N,))} for one video and class."""
    active, done = [], []
    for frame in sorted(frames):
        boxes, scores = frames[frame]
        claimed = np.zeros(len(boxes), dtype=bool)
        active.sort(key=lambda t: -np.mean(t['scores']))
        still = []
        for tube in active:
            if frame - tube['last'] > max_gap + 1:
                done.append(tube)
                continue
            if len(boxes):
                overlaps = iou(tube['boxes'][tube['last']], boxes)
                overlaps[claimed] = -1
                best = int(np.argmax(overlaps))
                if overlaps[best] >= link_iou:
                    claimed[best] = True
                    tube['boxes'][frame] = boxes[best]
                    tube['scores'].append(float(scores[best]))
                    tube['last'] = frame
            still.append(tube)
        for index in np.flatnonzero(~claimed):
            still.append({'boxes': {frame: boxes[index]}, 'scores': [float(scores[index])], 'last': frame})
        active = still
    done.extend(active)
    tubes = []
    for tube in done:
        span = max(tube['boxes']) - min(tube['boxes']) + 1
        if span < min_length:
            continue
        top = sorted(tube['scores'], reverse=True)[:score_top_k]
        tubes.append({'detections': tube['boxes'], 'score': float(np.mean(top))})
    return tubes


def build_tubes(rows, resolution, per_frame, link_iou, max_gap, min_length, score_top_k):
    grouped = defaultdict(lambda: defaultdict(list))
    for video, frame, label, score, x1, y1, x2, y2 in rows:
        height, width = resolution[video]
        grouped[(video, int(label))][int(frame) - 1].append(
            (float(score), [x1 / width, y1 / height, x2 / width, y2 / height]))
    tubes = []
    for (video, label), by_frame in grouped.items():
        frames = {}
        for frame, dets in by_frame.items():
            dets.sort(key=lambda d: -d[0])
            dets = dets[:per_frame]
            frames[frame] = (np.asarray([d[1] for d in dets]), np.asarray([d[0] for d in dets]))
        for tube in link_class(frames, link_iou, max_gap, min_length, score_top_k):
            tubes.append(dict(tube, video=video, **{'class': label}, resolution=resolution[video]))
    return tubes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--gt', required=True)
    parser.add_argument('--frame-dump', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-frame', type=int, nargs='+', default=[5])
    parser.add_argument('--link-iou', type=float, nargs='+', default=[0.3, 0.5])
    parser.add_argument('--max-gap', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--min-length', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--score-top-k', type=int, default=20)
    parser.add_argument('--min-score', type=float, default=0.05)
    args = parser.parse_args()

    gt = ems.load_groundtruth(args.gt)
    videos = gt['test_videos'][0]
    with open(args.frame_dump, 'rb') as handle:
        rows = [r for r in pickle.load(handle)['rows'] if r[3] >= args.min_score]
    results = []
    for per_frame in args.per_frame:
        for link_iou in args.link_iou:
            for max_gap in args.max_gap:
                for min_length in args.min_length:
                    tubes = build_tubes(rows, gt['resolution'], per_frame, link_iou, max_gap,
                                        min_length, args.score_top_k)
                    detections = ems.tubes_to_video_detections([to_multisports_tube(t) for t in tubes], videos)
                    row = dict(per_frame=per_frame, link_iou=link_iou, max_gap=max_gap, min_length=min_length,
                               tubes=len(tubes),
                               vap20=ems.video_ap(gt, detections, thr=0.2)[0],
                               vap50=ems.video_ap(gt, detections, thr=0.5)[0])
                    results.append(row)
                    print(json.dumps(row), flush=True)
    results.sort(key=lambda r: (r['vap50'], r['vap20']), reverse=True)
    with open(args.out, 'w') as handle:
        json.dump({'frame_dump': args.frame_dump, 'min_score': args.min_score, 'rows': results}, handle, indent=2)
    print('best:', json.dumps(results[0]))


if __name__ == '__main__':
    main()
