#!/usr/bin/env python3
"""B0: recalibrate the MOC-style tube linker on cached candidates (CPU only).

The linker parameters were tuned for v1 models. This grid-searches them on the
v2ho tune cache (g24/g25), then re-scores the best tune configurations and the
current defaults on the confirm cache (g22/g23). Nothing is selected on the
split-1 test set.

Selection rule (plan 08, B0): rank tune configurations by AP50, then AP50:95;
evaluate the top K on confirm; adopt the best-ranked one whose confirm AP50 is
>= default + 0.5 with AP20 and AP50:95 not below default. Otherwise keep the
defaults. The cache already applied visibility >= 0.35 and min_length >= 8, so
min_length is only searched at or above 8.

    python3 tools/sweep_linker_b0.py \
        --tune research_new/experiments/V2HO-HO0-s17_RESULT/tube_cache_v2ho-tune-g24g25_ema.pkl \
        --confirm research_new/experiments/V2HO-HO0-s17_RESULT/tube_cache_v2ho-confirm-g22g23_ema.pkl \
        --out research_new/experiments/B0_LINKER_SWEEP --workers 32
"""

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_video_map import load_gt_tubes  # noqa: E402
from video_map_protocol import compute_video_map_official, link_tubelets_moc  # noqa: E402

DEFAULTS = dict(link_iou=0.5, tubelet_nms=0.6, top_k=10, min_length=8, split_gap=-1, tube_nms=0.3)
GRID = dict(
    link_iou=[0.3, 0.4, 0.5, 0.6],
    tubelet_nms=[0.5, 0.6, 0.7],
    top_k=[5, 10, 20],
    min_length=[8, 12, 16],
    split_gap=[-1, 4, 8],
    tube_nms=[0.2, 0.3, 0.5],
)
THRESHOLDS = (0.2,) + tuple(np.arange(0.5, 0.951, 0.05))

_STATE = {}


def _load(path):
    with open(path, 'rb') as handle:
        cache = pickle.load(handle)
    videos = list(cache['videos'])
    gt = load_gt_tubes(cache['annot_file'], videos)
    ground_truth = [tube for video in videos for tube in gt[video]]
    return cache, ground_truth


def _init(path):
    _STATE['cache'], _STATE['gt'] = _load(path)


def score(params):
    cache, ground_truth = _STATE['cache'], _STATE['gt']
    predictions = []
    for video_data in cache['videos'].values():
        predictions.extend(link_tubelets_moc(
            video_data['candidates'], video_data['clip_starts'], cache['clip_length'],
            link_iou=params['link_iou'], tubelet_nms=params['tubelet_nms'],
            top_k=params['top_k'], min_length=params['min_length'],
            split_gap=params['split_gap'] if params['split_gap'] >= 0 else None,
        ))
    results = compute_video_map_official(
        predictions, ground_truth, iou_thresholds=THRESHOLDS,
        num_classes=cache['num_classes'], tube_nms=params['tube_nms'],
    )
    return dict(params,
                ap20=100 * results[0.2]['mAP'],
                ap50=100 * results[0.5]['mAP'],
                strict=100 * float(np.mean([results[float(t)]['mAP'] for t in THRESHOLDS[1:]])),
                tubes=len(predictions))


def run(path, configs, workers):
    with Pool(workers, initializer=_init, initargs=(path,)) as pool:
        return pool.map(score, configs, chunksize=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--tune', required=True)
    parser.add_argument('--confirm', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--top-k-confirm', type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    keys = list(GRID)
    configs = [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]
    if DEFAULTS not in configs:
        configs.append(dict(DEFAULTS))
    started = time.time()
    tune = run(args.tune, configs, args.workers)
    print(f'tune: {len(tune)} configurations in {time.time() - started:.0f}s', flush=True)
    tune.sort(key=lambda r: (r['ap50'], r['strict']), reverse=True)
    default_tune = next(r for r in tune if all(r[k] == DEFAULTS[k] for k in keys))

    shortlist = [{k: r[k] for k in keys} for r in tune[:args.top_k_confirm]]
    confirm = run(args.confirm, [dict(DEFAULTS)] + shortlist, min(args.workers, len(shortlist) + 1))
    default_confirm, confirm_rows = confirm[0], confirm[1:]

    chosen = None
    for row in confirm_rows:
        if (row['ap50'] >= default_confirm['ap50'] + 0.5
                and row['ap20'] >= default_confirm['ap20']
                and row['strict'] >= default_confirm['strict']):
            chosen = row
            break

    summary = {
        'tune_cache': args.tune, 'confirm_cache': args.confirm, 'grid': GRID, 'defaults': DEFAULTS,
        'default_tune': default_tune, 'default_confirm': default_confirm,
        'tune_top': tune[:args.top_k_confirm], 'confirm_shortlist': confirm_rows,
        'chosen': chosen,
        'rule': 'confirm AP50 >= default + 0.5, AP20 and AP50:95 not below default; first in tune rank',
    }
    with open(os.path.join(args.out, 'b0_linker_sweep.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(args.out, 'b0_tune_all.json'), 'w') as handle:
        json.dump(tune, handle)

    def fmt(r):
        return (f"link {r['link_iou']} tnms {r['tubelet_nms']} topk {r['top_k']} minlen {r['min_length']} "
                f"gap {r['split_gap']} tubenms {r['tube_nms']} | AP20 {r['ap20']:.2f} AP50 {r['ap50']:.2f} "
                f"AP50:95 {r['strict']:.2f}")
    print('default tune   :', fmt(default_tune))
    print('best tune      :', fmt(tune[0]))
    print('default confirm:', fmt(default_confirm))
    for row in confirm_rows[:5]:
        print('confirm        :', fmt(row))
    print('CHOSEN         :', fmt(chosen) if chosen else 'none (keep defaults)')


if __name__ == '__main__':
    main()
