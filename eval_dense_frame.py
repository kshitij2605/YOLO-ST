"""Protocol-audited frame mAP for a dense-only YOLO-ST model.

Why this exists
---------------
``eval_ucf24.py`` computes frame mAP with its own inline logic. The v1 protocol
audit of 2026-07-15 found that path double-counts frames from overlapping clips
under synthetic ids and omits tail-aligned windows, and replaced it with a
unique-frame VOC evaluator in ``frame_map_protocol.py``. Every v1 number quoted
as "corrected dense frame mAP", including the 89.23 dense owner, comes from
that corrected path, not from ``eval_ucf24.py``.

``eval_tube_queries.py --frame_map`` implements the corrected path, but it
requires ``model_output["tube_queries"]`` and calls
``model.enable_tube_query_output(True)``, so it cannot score a model configured
with ``apt_tube_queries: false``. The WS1 dense-owner experiments are exactly
that.

This module runs the same pipeline for a dense-only model, importing the
decoding and scoring functions from the existing evaluators rather than
reimplementing them, so the numbers stay comparable:

* clips are enumerated with ``build_clip_starts``, including the tail-aligned
  window;
* each clip frame is decoded with ``eval_tube_queries.decode_dense_frame``;
* scores are Hann-weighted across overlapping clips, exactly as
  ``eval_tube_queries`` does, so a frame near a clip edge is down-weighted;
* duplicates are resolved once per unique frame with
  ``resolve_frame_candidates``;
* AP is computed by ``compute_frame_map`` restricted to annotated frames.

Both conventions are reported, because they are not interchangeable:

``corrected``
    every annotated frame is evaluated.
``YOWO``
    the terminal row of each tube is dropped, which is the YOWO-lineage
    convention and reads roughly 0.3 points higher.

Usage::

    python eval_dense_frame.py --config <cfg> --checkpoint <ckpt> [--max-videos N]
"""

import argparse
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import torch

from config_utils import load_config
from data.ucf101_24 import build_clip_starts
from yolost.clip_weighting import MODES as CLIP_WEIGHTING_MODES, clip_frame_weights
from eval_video_map import load_model
from eval_tube_queries import decode_dense_frame, load_clip
from frame_map_protocol import (
    add_frame_predictions,
    compute_frame_map,
    load_frame_ground_truth,
    resolve_frame_candidates,
)


def dense_outputs(model_output):
    """Accept either the dense list or the full output dict."""
    if isinstance(model_output, dict):
        if 'dense' not in model_output:
            raise KeyError(
                'model output dict has no "dense" key; keys are '
                f'{sorted(model_output)}'
            )
        return model_output['dense']
    return model_output


def evaluate_dense_frame_map(config, checkpoint, max_videos=None,
                             conf_thresh=0.005, nms_thresh=0.5,
                             device=None, progress_every=100,
                             clip_weighting='hann', clip_overlap=0.5):
    cfg = load_config(config)
    device = device or torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    model = load_model(cfg, checkpoint, device)
    model.eval()
    if hasattr(model, 'enable_tube_query_output'):
        model.enable_tube_query_output(False)

    with open(cfg['data']['annot_file'], 'rb') as handle:
        annot = pickle.load(handle, encoding='latin1')
    videos = annot['test_videos'][int(cfg['data'].get('split_index', 0))]
    if max_videos is not None:
        videos = videos[:max_videos]

    corrected_gt, corrected_frames = load_frame_ground_truth(annot, videos)
    yowo_gt, yowo_frames = load_frame_ground_truth(
        annot, videos, drop_tube_terminal=True
    )

    clip_length = cfg['data']['clip_length']
    hann = np.hanning(clip_length + 2)[1:-1]
    predictions = defaultdict(list)
    started = time.time()

    for video_index, video_name in enumerate(videos):
        if hasattr(model, 'reset_memory'):
            model.reset_memory()
        num_frames = annot['nframes'][video_name]
        resolution = annot['resolution'].get(video_name, (240, 320))
        candidates = defaultdict(list)

        clip_starts = build_clip_starts(
            num_frames, clip_length, overlap=clip_overlap
        )
        clip_weights = clip_frame_weights(
            clip_starts, clip_length, num_frames, clip_weighting
        )
        for start in clip_starts:
            clip = load_clip(
                cfg['data']['root'], video_name, start, num_frames,
                clip_length, cfg['data']['img_size'], resolution,
            ).to(device)
            with torch.no_grad():
                dense = dense_outputs(model(clip))
            for local_frame in range(clip_length):
                global_frame = min(start + local_frame, num_frames) - 1
                if global_frame not in corrected_frames[video_name]:
                    continue
                for detection in decode_dense_frame(
                    dense, model, local_frame, conf_thresh, nms_thresh,
                ):
                    detection['score'] *= float(clip_weights[start][local_frame])
                    candidates[global_frame].append(detection)

        for frame_id in corrected_frames[video_name]:
            resolved = resolve_frame_candidates(
                candidates.get(frame_id, []), nms_thresh
            )
            add_frame_predictions(
                predictions, video_name, frame_id, resolved, resolution
            )

        if progress_every and (video_index + 1) % progress_every == 0:
            print(f'Processed {video_index + 1}/{len(videos)}', flush=True)

    corrected_eval_frames = {
        (video_name, frame_id)
        for video_name, frame_ids in corrected_frames.items()
        for frame_id in frame_ids
    }
    yowo_eval_frames = {
        (video_name, frame_id)
        for video_name, frame_ids in yowo_frames.items()
        for frame_id in frame_ids
    }
    num_classes = cfg['model']['num_classes']
    corrected_map, corrected_per_class = compute_frame_map(
        predictions, corrected_gt, num_classes=num_classes,
        evaluated_frames=corrected_eval_frames,
    )
    yowo_map, _ = compute_frame_map(
        predictions, yowo_gt, num_classes=num_classes,
        evaluated_frames=yowo_eval_frames,
    )
    return {
        'corrected_frame_map': 100.0 * corrected_map,
        'yowo_frame_map': 100.0 * yowo_map,
        'per_class': {
            str(key): 100.0 * value
            for key, value in corrected_per_class.items()
        },
        'videos': len(videos),
        'evaluated_frames': len(corrected_eval_frames),
        'elapsed_s': round(time.time() - started, 1),
        'checkpoint': checkpoint,
        'config': config,
        'clip_weighting': clip_weighting,
        'clip_overlap': clip_overlap,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--max-videos', type=int, default=None)
    parser.add_argument('--conf-thresh', type=float, default=0.005)
    parser.add_argument('--nms-thresh', type=float, default=0.5)
    parser.add_argument('--json-out', default=None)
    parser.add_argument('--clip-weighting', default='hann',
                        choices=CLIP_WEIGHTING_MODES)
    parser.add_argument('--clip-overlap', type=float, default=0.5)
    args = parser.parse_args(argv)

    result = evaluate_dense_frame_map(
        args.config, args.checkpoint, max_videos=args.max_videos,
        conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh,
        clip_weighting=args.clip_weighting,
        clip_overlap=args.clip_overlap,
    )
    print()
    print(f"videos evaluated        : {result['videos']}")
    print(f"annotated frames        : {result['evaluated_frames']}")
    print(f"elapsed                 : {result['elapsed_s']}s")
    print()
    print(
        'Dense unique-frame VOC mAP@0.5 (corrected/YOWO): '
        f"{result['corrected_frame_map']:.2f}% / "
        f"{result['yowo_frame_map']:.2f}%"
    )
    if args.json_out:
        with open(args.json_out, 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(f'wrote {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
