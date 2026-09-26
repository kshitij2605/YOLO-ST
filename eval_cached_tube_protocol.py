"""Evaluate cached query tubelets without repeating model inference."""

import argparse
import pickle

import numpy as np

from eval_tube_queries import merge_candidates
from eval_video_map import load_gt_tubes
from video_map_protocol import (
    compute_video_map_official,
    link_tubelets_global,
    link_tubelets_moc,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--merge_iou", type=float, default=0.5)
    parser.add_argument("--min_length", type=int, default=None)
    parser.add_argument("--moc_link_iou", type=float, default=0.5)
    parser.add_argument("--moc_tubelet_nms", type=float, default=0.6)
    parser.add_argument("--moc_tube_nms", type=float, default=0.3)
    parser.add_argument("--moc_top_k", type=int, default=10)
    parser.add_argument("--moc_split_gap", type=int, default=-1)
    parser.add_argument("--boundary_split_thresh", type=float, default=None)
    parser.add_argument("--endpoint_tolerance", type=int, default=-1)
    parser.add_argument("--endpoint_transition_margin", type=int, default=2)
    parser.add_argument("--endpoint_min_confidence", type=float, default=0.0)
    parser.add_argument("--global_linker", action="store_true")
    parser.add_argument("--global_only", action="store_true")
    parser.add_argument("--global_boundary_radius", type=int, default=6)
    parser.add_argument("--global_boundary_votes", type=int, default=2)
    parser.add_argument("--global_boundary_edge_window", type=int, default=8)
    parser.add_argument("--global_score_weight", type=float, default=0.05)
    return parser.parse_args()


def summarize(name, results, thresholds):
    sweep = np.mean([
        results[float(threshold)]["mAP"] for threshold in thresholds[1:]
    ])
    print(
        f"{name} official video-mAP@0.2/@0.5/@0.5:0.95: "
        f"{100 * results[0.2]['mAP']:.2f}% / "
        f"{100 * results[0.5]['mAP']:.2f}% / {100 * sweep:.2f}%"
    )


def main():
    args = parse_args()
    if args.global_only and not args.global_linker:
        raise ValueError("--global_only requires --global_linker")
    with open(args.candidate_cache, "rb") as handle:
        cache = pickle.load(handle)
    if cache.get("version") != 1:
        raise ValueError(f"Unsupported candidate cache version: {cache.get('version')}")

    min_length = args.min_length or cache["min_length"]
    videos = list(cache["videos"])
    gt_by_video = load_gt_tubes(cache["annot_file"], videos)
    ground_truth = [
        tube for video_name in videos for tube in gt_by_video[video_name]
    ]
    current_predictions = []
    moc_predictions = []
    global_predictions = []
    for video_name, video_data in cache["videos"].items():
        candidates = video_data["candidates"]
        if not args.global_only:
            moc_predictions.extend(link_tubelets_moc(
                candidates,
                video_data["clip_starts"],
                cache["clip_length"],
                link_iou=args.moc_link_iou,
                tubelet_nms=args.moc_tubelet_nms,
                top_k=args.moc_top_k,
                min_length=min_length,
                split_gap=args.moc_split_gap if args.moc_split_gap >= 0 else None,
                boundary_split_thresh=args.boundary_split_thresh,
                endpoint_tolerance=(
                    args.endpoint_tolerance
                    if args.endpoint_tolerance >= 0 else None
                ),
                endpoint_transition_margin=args.endpoint_transition_margin,
                endpoint_min_confidence=args.endpoint_min_confidence,
            ))
        if args.global_linker:
            global_predictions.extend(link_tubelets_global(
                candidates,
                video_data["clip_starts"],
                cache["clip_length"],
                link_iou=args.moc_link_iou,
                tubelet_nms=args.moc_tubelet_nms,
                top_k=args.moc_top_k,
                min_length=min_length,
                split_gap=(
                    args.moc_split_gap if args.moc_split_gap >= 0 else None
                ),
                boundary_split_thresh=args.boundary_split_thresh,
                boundary_radius=args.global_boundary_radius,
                boundary_votes=args.global_boundary_votes,
                boundary_edge_window=args.global_boundary_edge_window,
                association_score_weight=args.global_score_weight,
            ))
        if not args.global_only:
            current_predictions.extend(merge_candidates(
                candidates, args.merge_iou, min_length
            ))

    thresholds = (0.2,) + tuple(np.arange(0.5, 0.951, 0.05))
    if not args.global_only:
        current_results = compute_video_map_official(
            current_predictions, ground_truth,
            iou_thresholds=thresholds,
            num_classes=cache["num_classes"],
            tube_nms=args.moc_tube_nms,
        )
        moc_results = compute_video_map_official(
            moc_predictions, ground_truth,
            iou_thresholds=thresholds,
            num_classes=cache["num_classes"],
            tube_nms=args.moc_tube_nms,
        )
        print(f"Current-merger predicted tubes: {len(current_predictions)}")
        print(f"MOC-linker predicted tubes: {len(moc_predictions)}")
        summarize("Current merger", current_results, thresholds)
        summarize("MOC-style linker", moc_results, thresholds)
    if args.global_linker:
        global_results = compute_video_map_official(
            global_predictions, ground_truth,
            iou_thresholds=thresholds,
            num_classes=cache["num_classes"],
            tube_nms=args.moc_tube_nms,
        )
        print(f"Global-linker predicted tubes: {len(global_predictions)}")
        summarize("Global consensus linker", global_results, thresholds)


if __name__ == "__main__":
    main()
