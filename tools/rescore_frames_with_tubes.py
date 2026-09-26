#!/usr/bin/env python3
"""B-F1: rescore dense frame detections using the scores of the tubes that cover them.

Cache-only: it reads an existing dense frame dump and an existing tube cache, so it needs
no GPU and no retraining. The dense head and the tube-query head disagree about which
detections are confident; this blends the tube score into each frame detection score.

Facts about the v2 dense dump that the implementation depends on, all verified:

  * `rows` are (video, frame, class, score, x1, y1, x2, y2) with 1-indexed frames and
    pixel boxes, restricted to annotated frames.
  * They are already clip-weighted and already class-aware NMS-resolved (mean 1.31
    detections per frame; repeated same-class rows on a frame are non-overlapping NMS
    survivors). Re-applying NMS would move the control, so it is not re-applied.
  * The boxes are stored before rounding, while `add_frame_predictions` scores against
    `rint`-ed and clipped pixel coordinates, so the same rounding is applied here.

Tube detections are normalised and 0-indexed, so a row at frame f is matched against tube
boxes at f-1 after scaling the row box by 1/(W, H).

Scoring reproduces eval_dense_frame exactly: predictions are (score, (video, frame0),
pixel_box) triples fed to compute_frame_map, under both the corrected convention and the
YOWO convention (drop_tube_terminal=True).

Parity: --blend 0.0 must reproduce the control frame mAP for the partition.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_map_protocol import (  # noqa: E402
    compute_frame_map,
    inclusive_iou,
    load_frame_ground_truth,
)
from video_map_protocol import link_tubelets_moc  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-cache", required=True)
    parser.add_argument("--frame-dump", required=True)
    parser.add_argument("--annot", default=None,
                        help="defaults to the annot_file recorded in the dump")
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--match-iou", type=float, nargs="+", default=[0.3])
    parser.add_argument("--blend", type=float, nargs="+", default=[0.0],
                        help="weight on the covering tube score; 0.0 is the control")
    # frozen v2 UCF linker
    parser.add_argument("--link-iou", type=float, default=0.45)
    parser.add_argument("--tubelet-nms", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--split-gap", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=24)
    parser.add_argument("--mode", default="blend", choices=["blend", "multiply"],
                        help="blend mixes frame and tube scores; multiply scales the frame score by tube**w, preserving dense ordering within a tube")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def round_box(box, resolution):
    """Match normalized_to_pixels: round to integers and clip inside the image."""
    height, width = resolution
    result = np.rint(np.asarray(box, dtype=np.float32))
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width - 1)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height - 1)
    return result


def main():
    args = parse_args()
    with open(args.tube_cache, "rb") as handle:
        cache = pickle.load(handle)
    with open(args.frame_dump, "rb") as handle:
        dump = pickle.load(handle)

    base = int(dump.get("frame_index_base", 1))
    if dump.get("box_units") != "pixels":
        raise ValueError("unexpected box_units %r" % dump.get("box_units"))

    annot_path = args.annot or dump.get("annot_file") or cache.get("annot_file")
    with open(annot_path, "rb") as handle:
        annot = pickle.load(handle, encoding="latin1")
    videos = annot["test_videos"][args.split_index]
    corrected_gt, corrected_frames = load_frame_ground_truth(annot, videos)
    yowo_gt, yowo_frames = load_frame_ground_truth(annot, videos, drop_tube_terminal=True)
    corrected_eval = {(v, f) for v, ids in corrected_frames.items() for f in ids}
    yowo_eval = {(v, f) for v, ids in yowo_frames.items() for f in ids}

    resolutions = {v: tuple(d["resolution"]) for v, d in cache["videos"].items()}

    # Link tubes once with the frozen protocol, then index them by (video, frame0, class).
    tube_index = defaultdict(list)
    linked_total = 0
    for video, data in cache["videos"].items():
        tubes = link_tubelets_moc(
            data["candidates"], data["clip_starts"], cache["clip_length"],
            link_iou=args.link_iou, tubelet_nms=args.tubelet_nms, top_k=args.top_k,
            min_length=args.min_length, split_gap=args.split_gap,
        )
        linked_total += len(tubes)
        for tube in tubes:
            class_id = int(tube["class"])
            score = float(tube["score"])
            for frame0, box in tube["detections"].items():
                tube_index[(video, int(frame0), class_id)].append(
                    (np.asarray(box, dtype=np.float32), score)
                )
    print("linked tubes: %d, covered (video,frame,class) keys: %d"
          % (linked_total, len(tube_index)))

    rows = dump["rows"]
    print("dump rows: %d, videos: %d" % (len(rows), len(videos)))

    results = []
    for match_iou in args.match_iou:
        for blend in args.blend:
            predictions = defaultdict(list)
            rescored = 0
            for video, frame, class_id, score, x1, y1, x2, y2 in rows:
                resolution = resolutions.get(video)
                if resolution is None:
                    continue
                height, width = resolution
                frame0 = int(frame) - base
                new_score = float(score)
                if blend > 0.0:
                    covering = tube_index.get((video, frame0, int(class_id)))
                    if covering:
                        normalised = np.asarray(
                            [x1 / width, y1 / height, x2 / width, y2 / height],
                            dtype=np.float32,
                        )
                        best_iou = 0.0
                        best_score = 0.0
                        for tube_box, tube_score in covering:
                            overlap = inclusive_iou(normalised, tube_box)
                            if overlap > best_iou:
                                best_iou = overlap
                                best_score = tube_score
                        if best_iou >= match_iou:
                            if args.mode == "multiply":
                                new_score = float(score) * (max(best_score, 1e-6) ** blend)
                            else:
                                new_score = (1.0 - blend) * float(score) + blend * best_score
                            rescored += 1
                predictions[int(class_id)].append(
                    (new_score, (video, frame0),
                     round_box([x1, y1, x2, y2], resolution))
                )
            corrected_map, _ = compute_frame_map(
                predictions, corrected_gt, num_classes=args.num_classes,
                evaluated_frames=corrected_eval,
            )
            yowo_map, _ = compute_frame_map(
                predictions, yowo_gt, num_classes=args.num_classes,
                evaluated_frames=yowo_eval,
            )
            row = {
                "mode": args.mode,
                "match_iou": match_iou,
                "blend": blend,
                "rescored_rows": rescored,
                "rescored_frac": round(rescored / max(len(rows), 1), 4),
                "corrected": round(100.0 * corrected_map, 2),
                "yowo": round(100.0 * yowo_map, 2),
            }
            results.append(row)
            print("match_iou=%.2f blend=%.2f rescored=%.3f | corrected=%.2f yowo=%.2f"
                  % (match_iou, blend, row["rescored_frac"], row["corrected"], row["yowo"]))

    results.sort(key=lambda r: -r["corrected"])
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({
            "tube_cache": os.path.abspath(args.tube_cache),
            "frame_dump": os.path.abspath(args.frame_dump),
            "annot": os.path.abspath(annot_path),
            "rows": results,
        }, handle, indent=2)
    print("best:", json.dumps(results[0]))


if __name__ == "__main__":
    main()
