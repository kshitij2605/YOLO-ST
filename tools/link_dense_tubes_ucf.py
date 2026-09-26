#!/usr/bin/env python3
"""Video mAP from dense-detection tubes under the UCF101-24 / JHMDB-21 protocol.

Query tubes underperform the dense head on MultiSports and JHMDB (JHMDB split 1
dev: dense frame mAP 95.9, query-tube video mAP@0.5 51.5). This scores tubes
built by linking the dense per-frame detections of a --frame_dump pickle
(eval_tube_queries.py) with the greedy per-class linker of
tools/link_dense_frame_tubes.py, using the same official MOC/ACT video-mAP code
as every UCF video number in the ledger (video_map_protocol.compute_video_map_official,
tube NMS 0.3, thresholds 0.2 and 0.5:0.95).

Settings are selected on a tune partition and re-scored unchanged on a confirm
partition; the split-1 test set is never used for selection.

    python3 tools/link_dense_tubes_ucf.py --config CFG_TUNE --frame-dump tune.pkl \
        [--confirm-config CFG_CONFIRM --confirm-frame-dump confirm.pkl] --out result.json
"""

import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import load_config  # noqa: E402
from eval_video_map import load_gt_tubes  # noqa: E402
from tools.link_dense_frame_tubes import build_tubes  # noqa: E402
from video_map_protocol import compute_video_map_official  # noqa: E402

THRESHOLDS = (0.2,) + tuple(np.arange(0.5, 0.951, 0.05))


def load_partition(config_path, dump_path, min_score):
    cfg = load_config(config_path)
    with open(cfg['data']['annot_file'], 'rb') as handle:
        annot = pickle.load(handle, encoding='latin1')
    split = int(cfg['data'].get('split_index', 0))
    videos = list(annot['test_videos'][split])
    resolution = {v: tuple(annot['resolution'][v]) for v in videos}
    gt = load_gt_tubes(cfg['data']['annot_file'], videos)
    ground_truth = [tube for video in videos for tube in gt[video]]
    with open(dump_path, 'rb') as handle:
        rows = [r for r in pickle.load(handle)['rows'] if r[0] in resolution and r[3] >= min_score]
    return dict(videos=videos, resolution=resolution, ground_truth=ground_truth, rows=rows,
                num_classes=int(cfg['model']['num_classes']))


def score(partition, params, score_top_k):
    tubes = build_tubes(partition['rows'], partition['resolution'], params['per_frame'], params['link_iou'],
                        params['max_gap'], params['min_length'], score_top_k)
    results = compute_video_map_official(tubes, partition['ground_truth'], iou_thresholds=THRESHOLDS,
                                         num_classes=partition['num_classes'], tube_nms=0.3)
    return dict(params, tubes=len(tubes), ap20=100 * results[0.2]['mAP'], ap50=100 * results[0.5]['mAP'],
                strict=100 * float(np.mean([results[float(t)]['mAP'] for t in THRESHOLDS[1:]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--frame-dump', required=True)
    parser.add_argument('--confirm-config')
    parser.add_argument('--confirm-frame-dump')
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-frame', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--link-iou', type=float, nargs='+', default=[0.2, 0.3, 0.5])
    parser.add_argument('--max-gap', type=int, nargs='+', default=[2, 5])
    parser.add_argument('--min-length', type=int, nargs='+', default=[8, 16])
    parser.add_argument('--score-top-k', type=int, default=40)
    parser.add_argument('--min-score', type=float, default=0.05)
    parser.add_argument('--top-confirm', type=int, default=5)
    args = parser.parse_args()

    tune = load_partition(args.config, args.frame_dump, args.min_score)
    grid = [dict(per_frame=p, link_iou=l, max_gap=g, min_length=m)
            for p, l, g, m in itertools.product(args.per_frame, args.link_iou, args.max_gap, args.min_length)]
    rows = []
    for params in grid:
        row = score(tune, params, args.score_top_k)
        rows.append(row)
        print('tune', json.dumps(row), flush=True)
    rows.sort(key=lambda r: (r['ap50'], r['strict']), reverse=True)
    result = {'config': args.config, 'frame_dump': args.frame_dump, 'min_score': args.min_score,
              'score_top_k': args.score_top_k, 'tune': rows}
    if args.confirm_config and args.confirm_frame_dump:
        confirm = load_partition(args.confirm_config, args.confirm_frame_dump, args.min_score)
        keys = ('per_frame', 'link_iou', 'max_gap', 'min_length')
        result['confirm'] = [score(confirm, {k: r[k] for k in keys}, args.score_top_k)
                             for r in rows[:args.top_confirm]]
        for row in result['confirm']:
            print('confirm', json.dumps(row), flush=True)
    with open(args.out, 'w') as handle:
        json.dump(result, handle, indent=2)
    print('best tune:', json.dumps(rows[0]))


if __name__ == '__main__':
    main()
