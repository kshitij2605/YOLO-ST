#!/usr/bin/env python3
"""Export AVA v2.2 detections from a dense YOLO-ST checkpoint and score them.

For every annotated keyframe in --annot (the ground-truth CSV of the evaluated
partition), the clip is built exactly as in training (data.keyframe_position,
data.frame_stride), the dense head is decoded at the keyframe's clip frame with
yolost.decode.decode_dense_multilabel, and one AVA CSV row is written per
(person box, action) with score >= --min-score.

Scores: 'cls_x_actor' (default) multiplies each action probability by the
box's actor score (objectness); 'cls' uses the action probability alone.

Sharding: run --num-shards N --shard-index k on N GPUs, then --merge to
concatenate shard CSVs and evaluate once with the official AVA code.

    python3 tools/export_ava_detections.py --config C --checkpoint K \
        --annot data/ava/dev_splits_v2/ava_dev_val.csv --out-dir OUT \
        --num-shards 2 --shard-index 0
    python3 tools/export_ava_detections.py --config C --annot ... --out-dir OUT \
        --merge --num-shards 2 --exclusions data/ava/dev_splits_v2/ava_dev_excluded.csv
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_ava  # noqa: E402
from config_utils import load_config  # noqa: E402
from data.ava_dataset import AVADataset  # noqa: E402
from yolost.decode import decode_dense_multilabel  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--annot', required=True, help='ground-truth CSV of the evaluated partition')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--actor-thresh', type=float, default=0.2)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    parser.add_argument('--max-detections', type=int, default=20)
    parser.add_argument('--min-score', type=float, default=0.001)
    parser.add_argument('--score-mode', choices=('cls_x_actor', 'cls'), default='cls_x_actor')
    parser.add_argument('--labelmap', default=eval_ava.DEFAULT_LABELMAP)
    parser.add_argument('--exclusions', default=None)
    parser.add_argument('--merge', action='store_true')
    return parser.parse_args(argv)


class _Indexed(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        clip, _ = self.dataset[index]
        return clip, index


def rows_for_detections(video_id, timestamp, boxes, actors, classes, score_mode, min_score):
    """AVA records (video, sec, x1, y1, x2, y2, model_class, score)."""
    scores = classes * actors.unsqueeze(1) if score_mode == 'cls_x_actor' else classes
    rows = []
    for box, box_scores in zip(boxes.tolist(), scores.tolist()):
        for class_index, score in enumerate(box_scores):
            if score >= min_score:
                rows.append((video_id, timestamp, *box, class_index, score))
    return rows


def shard_path(out_dir, index, count):
    return os.path.join(out_dir, f'detections_shard{index:02d}of{count:02d}.csv')


@torch.no_grad()
def export_shard(args, cfg):
    from eval_video_map import load_model

    data = cfg['data']
    dataset = AVADataset(
        frames_root=data['frames_root'], annot_file=args.annot,
        clip_length=data['clip_length'], img_size=data['img_size'], split='val',
        augment=False, multi_label=True,
        keyframe_position=data.get('keyframe_position', 'end'),
        frame_stride=data.get('frame_stride'),
    )
    indices = list(range(len(dataset.samples)))[args.shard_index::args.num_shards]
    loader = DataLoader(_Indexed(dataset, indices), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True)
    key_clip_t = (data['clip_length'] - 1 if data.get('keyframe_position', 'end') == 'end'
                  else data['clip_length'] // 2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(cfg, args.checkpoint, device)
    model.eval()
    if hasattr(model, 'enable_tube_query_output'):
        model.enable_tube_query_output(False)

    rows, started = [], time.time()
    for batch_number, (clips, sample_indices) in enumerate(loader):
        if hasattr(model, 'reset_memory'):
            model.reset_memory()
        output = model(clips.to(device, non_blocking=True))
        dense = output['dense'] if isinstance(output, dict) else output
        for b, sample_index in enumerate(sample_indices.tolist()):
            video_id, timestamp = dataset.samples[sample_index]
            boxes, actors, classes = decode_dense_multilabel(
                dense, model.temporal_strides, model.spatial_strides, model.img_size,
                batch_index=b, clip_frame=key_clip_t, actor_thresh=args.actor_thresh,
                nms_thresh=args.nms_thresh, max_detections=args.max_detections)
            rows.extend(rows_for_detections(video_id, timestamp, boxes.cpu(), actors.cpu(),
                                            classes.cpu(), args.score_mode, args.min_score))
        if batch_number % 50 == 0:
            done = min((batch_number + 1) * args.batch_size, len(indices))
            print(f'shard {args.shard_index}: {done}/{len(indices)} keyframes, '
                  f'{time.time() - started:.0f}s', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    labelmap = eval_ava.read_labelmap(args.labelmap)[0]
    path = shard_path(args.out_dir, args.shard_index, args.num_shards)
    written = eval_ava.write_ava_detections(path, rows, labelmap=labelmap)
    print(f'wrote {written} rows for {len(indices)} keyframes to {path}')


def merge_and_evaluate(args):
    shards = [shard_path(args.out_dir, i, args.num_shards) for i in range(args.num_shards)]
    missing = [path for path in shards if not os.path.isfile(path)]
    if missing:
        sys.exit(f'missing shards: {missing}')
    merged = os.path.join(args.out_dir, 'detections.csv')
    with open(merged, 'w') as out:
        for path in shards:
            with open(path) as handle:
                out.write(handle.read())
    metrics = eval_ava.evaluate_ava(groundtruth=args.annot, detections=merged,
                                    labelmap=args.labelmap, exclusions=args.exclusions)
    result = {
        'map': float(metrics['mAP@0.5IOU']),
        'per_class': {k: float(v) for k, v in eval_ava.per_class_ap(metrics).items()},
        'annot': args.annot, 'config': args.config, 'checkpoint': args.checkpoint,
        'actor_thresh': args.actor_thresh, 'nms_thresh': args.nms_thresh,
        'score_mode': args.score_mode, 'min_score': args.min_score,
        'max_detections': args.max_detections,
    }
    with open(os.path.join(args.out_dir, 'ava_metrics.json'), 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"AVA frame mAP@0.5: {100 * result['map']:.2f}")
    return result


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.merge:
        merge_and_evaluate(args)
    else:
        if not args.checkpoint:
            sys.exit('--checkpoint is required unless --merge')
        export_shard(args, cfg)


if __name__ == '__main__':
    main()
