#!/usr/bin/env python3
"""B2: snap linked tube boxes toward the model own dense frame detections.

Ports tools/eval_cached_tubes_with_frame_geometry.py (v1) to the v2 caches. Two
format differences and one protocol trap made a direct reuse impossible:

  * v1 wants --frame-predictions to be a directory of per-video pickles, each
    {"video", "frames"}. The v2 dense dump is a single file whose "rows" is a flat
    list of (video, frame, class, score, x1, y1, x2, y2) tuples.
  * v2 rows are in pixels and 1-indexed; tube detections are normalised and
    0-indexed (see export_moc_tubes, which multiplies by width/height and adds 1).
    Rows are therefore converted with frame-1 and box/(W,H,W,H).
  * v1 links with min_length taken from the cache, which stores 8 here, while the
    frozen v2 protocol mandates 16. Taking the cache value would silently retune the
    linker, so min_length comes from the freeze and defaults to 16.

Linking does not depend on the refinement parameters, so tubes are linked once and
every grid point re-scores the same linked tubes.

Selection happens on the tune partition (g24/g25) and is confirmed on g22/g23; the
control is the frozen-B0 protocol (tune 93.75/80.21/43.20, confirm 89.60/77.29/41.51).
Run with --box-blend 0.0 alone as a parity check: it must reproduce the control.
"""

import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_video_map import load_gt_tubes  # noqa: E402
from video_map_protocol import compute_video_map_official, link_tubelets_moc  # noqa: E402

OFFICIAL_THRESHOLDS = (0.2,) + tuple(np.arange(0.5, 0.951, 0.05))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tube-cache", required=True)
    parser.add_argument("--frame-dump", required=True)
    parser.add_argument("--match-iou", type=float, nargs="+", default=[0.3])
    parser.add_argument("--box-blend", type=float, nargs="+", default=[0.0])
    parser.add_argument("--score-blend", type=float, nargs="+", default=[0.0])
    parser.add_argument("--smooth-radius", type=int, nargs="+", default=[0])
    # frozen v2 UCF protocol
    parser.add_argument("--link-iou", type=float, default=0.45)
    parser.add_argument("--tubelet-nms", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--split-gap", type=int, default=2)
    parser.add_argument("--tube-nms", type=float, default=0.3)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def box_iou(box_a, box_b):
    left_top = np.maximum(box_a[:2], box_b[:2])
    right_bottom = np.minimum(box_a[2:], box_b[2:])
    extent = np.maximum(right_bottom - left_top, 0.0)
    intersection = float(extent[0] * extent[1])
    area_a = float(np.prod(np.maximum(box_a[2:] - box_a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(box_b[2:] - box_b[:2], 0.0)))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def load_frame_predictions(path, resolutions):
    """Flat v2 rows -> {video: {frame0: (N,6) [x1,y1,x2,y2,score,class] normalised}}."""
    with open(path, "rb") as handle:
        dump = pickle.load(handle)
    base = int(dump.get("frame_index_base", 1))
    units = dump.get("box_units", "pixels")
    if units != "pixels":
        raise ValueError("unexpected box_units %r" % units)
    staged = {}
    skipped = 0
    for row in dump["rows"]:
        video, frame, class_id, score, x1, y1, x2, y2 = row
        resolution = resolutions.get(video)
        if resolution is None:
            skipped += 1
            continue
        height, width = resolution
        frame0 = int(frame) - base
        staged.setdefault(video, {}).setdefault(frame0, []).append(
            [x1 / width, y1 / height, x2 / width, y2 / height, float(score), int(class_id)]
        )
    predictions = {
        video: {frame: np.asarray(values, dtype=np.float32) for frame, values in frames.items()}
        for video, frames in staged.items()
    }
    return predictions, dump, skipped


def refine_tube(tube, frame_predictions, match_iou, box_blend, score_blend):
    output = copy.copy(tube)
    output["detections"] = {
        frame: np.asarray(box, dtype=np.float32).copy()
        for frame, box in tube["detections"].items()
    }
    matched_scores = []
    matched = 0
    for frame, primary_box in output["detections"].items():
        detections = frame_predictions.get(frame)
        if detections is None or not len(detections):
            continue
        same_class = detections[detections[:, 5].astype(int) == int(tube["class"])]
        if not len(same_class):
            continue
        overlaps = np.asarray([box_iou(primary_box, value[:4]) for value in same_class])
        selected = int(np.argmax(overlaps))
        if float(overlaps[selected]) < match_iou:
            continue
        geometry = same_class[selected]
        output["detections"][frame] = (
            (1.0 - box_blend) * primary_box + box_blend * geometry[:4]
        ).astype(np.float32)
        matched_scores.append(float(geometry[4]))
        matched += 1
    if matched_scores and score_blend > 0:
        output["score"] = float(
            (1.0 - score_blend) * tube["score"] + score_blend * float(np.mean(matched_scores))
        )
    return output, matched, len(output["detections"])


def smooth_tube(tube, radius):
    if radius <= 0 or len(tube["detections"]) < 3:
        return tube
    output = copy.copy(tube)
    frames = sorted(tube["detections"])
    boxes = np.stack([tube["detections"][frame] for frame in frames])
    smoothed = {}
    for index, frame in enumerate(frames):
        begin = max(0, index - radius)
        end = min(len(frames), index + radius + 1)
        valid = [
            offset for offset in range(begin, end)
            if abs(frames[offset] - frame) <= radius
        ]
        if not valid:
            smoothed[frame] = boxes[index].copy()
            continue
        weights = np.asarray(
            [radius + 1 - abs(frames[offset] - frame) for offset in valid], dtype=np.float32
        )
        smoothed[frame] = np.average(boxes[valid], axis=0, weights=weights).astype(np.float32)
    output["detections"] = smoothed
    return output


def main():
    args = parse_args()
    with open(args.tube_cache, "rb") as handle:
        cache = pickle.load(handle)

    video_items = list(cache["videos"].items())
    if args.max_videos is not None:
        video_items = video_items[:args.max_videos]
    videos = [video for video, _ in video_items]
    resolutions = {video: tuple(data["resolution"]) for video, data in video_items}

    frame_predictions, dump, skipped = load_frame_predictions(args.frame_dump, resolutions)
    covered = sum(1 for video in videos if video in frame_predictions)
    print("videos=%d with-frame-predictions=%d rows-skipped=%d clip_weighting=%s"
          % (len(videos), covered, skipped, dump.get("clip_weighting")))
    if covered == 0:
        raise SystemExit("no video in the tube cache appears in the frame dump")

    # Link once with the frozen protocol; refinement does not affect linking.
    linked = {}
    for video, data in video_items:
        tubes = link_tubelets_moc(
            data["candidates"],
            data["clip_starts"],
            cache["clip_length"],
            link_iou=args.link_iou,
            tubelet_nms=args.tubelet_nms,
            top_k=args.top_k,
            min_length=args.min_length,
            split_gap=args.split_gap,
        )
        for tube in tubes:
            tube.setdefault("video", video)
            tube.setdefault("resolution", resolutions[video])
        linked[video] = tubes
    print("linked tubes: %d" % sum(len(v) for v in linked.values()))

    ground_truth_by_video = load_gt_tubes(cache["annot_file"], videos)
    ground_truth = [tube for video in videos for tube in ground_truth_by_video[video]]

    rows = []
    for match_iou in args.match_iou:
        for box_blend in args.box_blend:
            for score_blend in args.score_blend:
                for radius in args.smooth_radius:
                    predictions = []
                    matched = total = 0
                    for video in videos:
                        per_frame = frame_predictions.get(video, {})
                        for tube in linked[video]:
                            refined, hit, seen = refine_tube(
                                tube, per_frame, match_iou, box_blend, score_blend
                            )
                            predictions.append(smooth_tube(refined, radius))
                            matched += hit
                            total += seen
                    results = compute_video_map_official(
                        predictions,
                        ground_truth,
                        iou_thresholds=OFFICIAL_THRESHOLDS,
                        num_classes=cache["num_classes"],
                        tube_nms=args.tube_nms,
                    )
                    row = {
                        "match_iou": match_iou,
                        "box_blend": box_blend,
                        "score_blend": score_blend,
                        "smooth_radius": radius,
                        "matched_frac": round(matched / max(total, 1), 4),
                        "ap20": round(100.0 * results[0.2]["mAP"], 2),
                        "ap50": round(100.0 * results[0.5]["mAP"], 2),
                        "strict": round(100.0 * float(np.mean([
                            results[float(t)]["mAP"] for t in OFFICIAL_THRESHOLDS[1:]
                        ])), 2),
                    }
                    rows.append(row)
                    print("match_iou=%.2f box_blend=%.2f score_blend=%.2f smooth=%d "
                          "matched=%.3f | ap20=%.2f ap50=%.2f strict=%.2f"
                          % (match_iou, box_blend, score_blend, radius,
                             row["matched_frac"], row["ap20"], row["ap50"], row["strict"]))

    rows.sort(key=lambda r: (-r["ap50"], -r["strict"]))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({
            "tube_cache": os.path.abspath(args.tube_cache),
            "frame_dump": os.path.abspath(args.frame_dump),
            "protocol": {
                "link_iou": args.link_iou, "tubelet_nms": args.tubelet_nms,
                "top_k": args.top_k, "min_length": args.min_length,
                "split_gap": args.split_gap, "tube_nms": args.tube_nms,
            },
            "rows": rows,
        }, handle, indent=2)
    print("best:", json.dumps(rows[0]))


if __name__ == "__main__":
    main()
