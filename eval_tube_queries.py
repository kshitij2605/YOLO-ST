"""Evaluate learned tube-query sequences directly, without frame linking."""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# NumPy 2 pickles refer to numpy._core; NumPy 1.x exposes the same modules
# under numpy.core. Register the old runtime aliases before loading annotations.
try:
    import numpy._core.numeric  # noqa: F401
except ImportError:
    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_numeric

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)

from config_utils import load_config
from data.ucf101_24 import build_clip_starts
from yolost.clip_weighting import MODES as CLIP_WEIGHTING_MODES, clip_frame_weights
from yolost.geometry import image_hw
from eval_video_map import compute_video_map, load_gt_tubes, load_model
from frame_map_protocol import (
    add_frame_predictions,
    compute_frame_map,
    load_frame_ground_truth,
    resolve_frame_candidates,
)
from video_map_protocol import compute_video_map_official, link_tubelets_moc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--visibility", type=float, default=0.35)
    parser.add_argument("--merge_iou", type=float, default=0.3)
    # --min_length has two roles: (1) a query tubelet needs at least this many
    # frames with visibility >= --visibility to become a candidate, and (2) it is
    # the minimum tube length of the in-script MOC linker and merger below. The
    # frozen protocol filters candidates at 8 and links at 16, so run with
    # --min_length 8 --candidate_cache and score the cache with
    # tools/snap_tubes_to_dense_geometry.py or eval_cached_tube_protocol.py, whose
    # linker length comes from the protocol freeze (16). The video metrics this
    # script prints after a --min_length 8 run link at 8 and are not the protocol.
    parser.add_argument(
        "--min_length", type=int, default=8,
        help="candidate filter (visible frames) and in-script linker length; the "
             "frozen protocol filters at 8 and links at 16 (see source)",
    )
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--frame_conf_thresh", type=float, default=0.005)
    parser.add_argument("--frame_nms_thresh", type=float, default=0.5)
    parser.add_argument("--tube_frame_scale", type=float, default=1.0)
    parser.add_argument("--frame_map", action="store_true")
    parser.add_argument(
        "--hflip", action="store_true",
        help="run on horizontally flipped clips and map every box back to the "
             "unflipped frame: one view of flip test-time augmentation "
             "(fuse the two views with tools/fuse_flip_views.py)",
    )
    parser.add_argument("--moc_link_iou", type=float, default=0.5)
    parser.add_argument("--moc_tubelet_nms", type=float, default=0.6)
    parser.add_argument("--moc_tube_nms", type=float, default=0.3)
    parser.add_argument("--moc_top_k", type=int, default=10)
    parser.add_argument("--moc_split_gap", type=int, default=-1)
    parser.add_argument("--boundary_split_thresh", type=float, default=None)
    parser.add_argument("--endpoint_tolerance", type=int, default=-1)
    parser.add_argument("--endpoint_transition_margin", type=int, default=2)
    parser.add_argument("--endpoint_min_confidence", type=float, default=0.0)
    parser.add_argument("--candidate_cache", default=None)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--clip_overlap", type=float, default=0.5)
    parser.add_argument("--frame_dump", default=None,
                        help="pickle dense detections for every frame (MultiSports export)")
    parser.add_argument("--clip_weighting", default="hann",
                        choices=CLIP_WEIGHTING_MODES)
    parser.add_argument(
        "--cache-only", action="store_true",
        help="Write raw candidates from metadata only, without loading labels or metrics.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Label-free video metadata required by --cache-only.",
    )
    return parser.parse_args()


def box_iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    inter = np.maximum(rb - lt, 0).prod()
    area_a = np.maximum(a[2:] - a[:2], 0).prod()
    area_b = np.maximum(b[2:] - b[:2], 0).prod()
    return float(inter / max(area_a + area_b - inter, 1e-7))


def candidate_overlap(candidate, tube):
    overlap = set(candidate["detections"]) & set(tube["detections"])
    if overlap:
        return float(np.mean([
            box_iou(candidate["detections"][frame], tube["detections"][frame])
            for frame in overlap
        ]))
    first = min(candidate["detections"])
    last = max(tube["detections"])
    if 0 < first - last <= 4:
        return box_iou(candidate["detections"][first], tube["detections"][last])
    return 0.0


def merge_candidates(candidates, merge_iou, min_length):
    merged = []
    for source in sorted(candidates, key=lambda item: min(item["detections"])):
        candidate = dict(source)
        candidate["detections"] = {
            frame: np.asarray(box, dtype=np.float32).copy()
            for frame, box in source["detections"].items()
        }
        candidate["frame_weights"] = dict(source["frame_weights"])
        candidate["scores"] = list(source["scores"])
        options = [
            (candidate_overlap(candidate, tube), index)
            for index, tube in enumerate(merged)
            if tube["class"] == candidate["class"]
        ]
        score, index = max(options, default=(0.0, -1))
        if score < merge_iou:
            merged.append(candidate)
            continue
        tube = merged[index]
        for frame, box in candidate["detections"].items():
            weight = candidate["frame_weights"][frame]
            if frame in tube["detections"]:
                old_weight = tube["frame_weights"][frame]
                tube["detections"][frame] = (
                    tube["detections"][frame] * old_weight + box * weight
                ) / max(old_weight + weight, 1e-7)
                tube["frame_weights"][frame] = old_weight + weight
            else:
                tube["detections"][frame] = box
                tube["frame_weights"][frame] = weight
        tube["scores"].append(candidate["score"])
        tube["score"] = float(np.mean(tube["scores"]))
    return [tube for tube in merged if len(tube["detections"]) >= min_length]


def load_clip(root, video_name, start, num_frames, clip_length, img_size, resolution):
    height, width = resolution
    frames = []
    for offset in range(clip_length):
        frame = min(start + offset, num_frames)
        path = os.path.join(root, video_name, f"{frame:05d}.jpg")
        if not os.path.exists(path):
            path = os.path.join(root, video_name, f"{frame:05d}.png")
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _h, _w = image_hw(img_size)
        frames.append(cv2.resize(image, (_w, _h)))
    clip = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    return ((clip - mean) / std).unsqueeze(0)


def unflip_normalized_boxes(boxes):
    """Map normalized x1,y1,x2,y2 boxes predicted on a horizontally flipped
    frame back to the unflipped frame (x -> 1 - x, so x1 and x2 swap)."""
    if isinstance(boxes, torch.Tensor):
        return torch.stack((1.0 - boxes[..., 2], boxes[..., 1],
                            1.0 - boxes[..., 0], boxes[..., 3]), dim=-1)
    boxes = np.asarray(boxes)
    return np.stack((1.0 - boxes[..., 2], boxes[..., 1],
                     1.0 - boxes[..., 0], boxes[..., 3]), axis=-1)


def expand_tubelet(boxes, visibility, clip_length):
    """Interpolate sampled tubelet predictions onto every input clip frame."""
    if boxes.shape[0] == clip_length:
        return boxes, visibility
    boxes = F.interpolate(
        boxes.transpose(0, 1).unsqueeze(0), size=clip_length,
        mode="linear", align_corners=True,
    )[0].transpose(0, 1)
    visibility = F.interpolate(
        visibility.view(1, 1, -1), size=clip_length,
        mode="linear", align_corners=True,
    )[0, 0]
    return boxes, visibility


def decode_dense_frame(outputs, model, clip_frame, conf_thresh, nms_thresh):
    """Decode one clip frame from the dense YOLO head."""
    all_boxes, all_scores, all_classes = [], [], []
    for scale_index, scale_out in enumerate(outputs):
        cls_pred, reg_pred, obj_pred = scale_out[:3]
        temporal_stride = model.temporal_strides[scale_index]
        spatial_stride = model.spatial_strides[scale_index]
        detection_frame = clip_frame // temporal_stride
        if detection_frame >= cls_pred.shape[2]:
            continue
        step_y = spatial_stride / image_hw(model.img_size)[0]
        step_x = spatial_stride / image_hw(model.img_size)[1]
        objectness = torch.sigmoid(obj_pred[0, 0, detection_frame])
        class_prob = torch.sigmoid(cls_pred[0, :, detection_frame])
        regression = reg_pred[0, :, detection_frame]
        score, class_id = (objectness.unsqueeze(0) * class_prob).max(dim=0)
        mask = score > conf_thresh
        if not mask.any():
            continue
        row, column = torch.where(mask)
        raw_box = regression[:, row, column].T
        if raw_box.shape[-1] != 4:
            # Distribution regression: LTRB expectation from the anchor centre.
            from yolost.dfl import distribution_expectation
            reg_max = raw_box.shape[-1] // 4 - 1
            distance = distribution_expectation(
                raw_box.unsqueeze(0), reg_max
            )[0]
            anchor_x = (column.float() + 0.5) * step_x
            anchor_y = (row.float() + 0.5) * step_y
            all_boxes.append(torch.stack([
                anchor_x - distance[:, 0] * step_x,
                anchor_y - distance[:, 1] * step_y,
                anchor_x + distance[:, 2] * step_x,
                anchor_y + distance[:, 3] * step_y,
            ], dim=-1))
        else:
            center_x = (column.float() + torch.sigmoid(raw_box[:, 0])) * step_x
            center_y = (row.float() + torch.sigmoid(raw_box[:, 1])) * step_y
            width = torch.exp(raw_box[:, 2].clamp(max=5.0)) * step_x
            height = torch.exp(raw_box[:, 3].clamp(max=5.0)) * step_y
            all_boxes.append(torch.stack([
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            ], dim=-1))
        all_scores.append(score[mask])
        all_classes.append(class_id[mask])
    if not all_boxes:
        return []
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    classes = torch.cat(all_classes)
    detections = []
    for class_id in classes.unique():
        class_mask = classes == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        keep = torchvision.ops.nms(class_boxes, class_scores, nms_thresh)
        for index in keep:
            detections.append({
                "class": int(class_id),
                "score": float(class_scores[index]),
                "box": class_boxes[index].detach().cpu().numpy(),
            })
    return detections


@torch.no_grad()
def main():
    args = parse_args()
    protocol_freeze = Path(
        "research/backbone_expansion/results/heldout_owner_calibration_g24g25/"
        "PROTOCOL_FREEZE_V2.json"
    )
    if (
        args.candidate_cache
        and "heldout_g24g25" in Path(args.candidate_cache).name
        and protocol_freeze.is_file()
    ):
        args.cache_only = True
        args.frame_map = False
        args.metadata_json = str(
            protocol_freeze.parent / "heldout_g24g25_eval_metadata.json"
        )
        print(f"Post-freeze cache-only guard active: {protocol_freeze}")
    if args.num_shards > 1:
        if not args.candidate_cache:
            raise ValueError("--num_shards > 1 requires --candidate_cache")
        if args.frame_map:
            raise ValueError("--frame_map needs every video; do not shard it")
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("--shard_index must be in [0, num_shards)")
    if args.cache_only and not args.candidate_cache:
        raise ValueError("--cache-only requires --candidate_cache")
    if args.cache_only and not args.metadata_json:
        raise ValueError("--cache-only requires --metadata-json")
    if args.cache_only and args.frame_map:
        raise ValueError("--cache-only and --frame_map are mutually exclusive")
    cfg = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    model.eval()
    model.enable_tube_query_output(True)
    if hasattr(model, "enable_persistent_memory"):
        model.enable_persistent_memory(True)

    if args.cache_only:
        with open(args.metadata_json, encoding="ascii") as handle:
            metadata = json.load(handle)
        annot = {
            "test_videos": [metadata["videos"]],
            "nframes": metadata["nframes"],
            "resolution": metadata["resolution"],
        }
    else:
        with open(cfg["data"]["annot_file"], "rb") as handle:
            annot = pickle.load(handle, encoding="latin1")
    videos = annot["test_videos"][int(cfg["data"].get("split_index", 0))]
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    if args.num_shards > 1:
        videos = videos[args.shard_index::args.num_shards]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(videos)} videos")
    gt_by_video = None
    corrected_gt = corrected_frames = yowo_gt = yowo_frames = None
    if not args.cache_only:
        gt_by_video = load_gt_tubes(cfg["data"]["annot_file"], videos)
        if args.frame_map:
            corrected_gt, corrected_frames = load_frame_ground_truth(annot, videos)
            yowo_gt, yowo_frames = load_frame_ground_truth(
                annot, videos, drop_tube_terminal=True
            )
    clip_length = cfg["data"]["clip_length"]
    all_predictions, all_moc_predictions, all_ground_truth = [], [], []
    frame_predictions = {
        "dense": defaultdict(list),
        "tubelet": defaultdict(list),
        "fused": defaultdict(list),
    }
    cached_videos = {}
    frame_dump_rows = []
    started = time.time()

    for video_index, video_name in enumerate(videos):
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        num_frames = annot["nframes"][video_name]
        resolution = annot["resolution"].get(video_name, (240, 320))
        candidates = []
        dense_frame_candidates = defaultdict(list)
        tube_frame_candidates = defaultdict(list)
        hann = np.hanning(clip_length + 2)[1:-1]
        clip_starts = build_clip_starts(
            num_frames, clip_length, overlap=args.clip_overlap
        )
        clip_weights = clip_frame_weights(
            clip_starts, clip_length, num_frames, args.clip_weighting
        )
        for start in clip_starts:
            clip = load_clip(
                cfg["data"]["root"], video_name, start, num_frames,
                clip_length, cfg["data"]["img_size"], resolution,
            ).to(device)
            if args.hflip:
                clip = torch.flip(clip, dims=[-1])
            model_output = model(clip)
            output = model_output["tube_queries"]
            if args.frame_map or args.frame_dump:
                dense_output = model_output["dense"]
                for local_frame in range(clip_length):
                    global_frame = min(start + local_frame, num_frames) - 1
                    if (not args.frame_dump
                            and global_frame not in corrected_frames[video_name]):
                        continue
                    for detection in decode_dense_frame(
                        dense_output, model, local_frame,
                        args.frame_conf_thresh, args.frame_nms_thresh,
                    ):
                        if args.hflip:
                            detection["box"] = unflip_normalized_boxes(detection["box"])
                        detection["score"] *= float(clip_weights[start][local_frame])
                        dense_frame_candidates[global_frame].append(detection)
            if cfg.get("loss", {}).get("query_class_focal_alpha") is not None:
                class_prob = output["class_logits"][0].sigmoid()
            else:
                class_prob = output["class_logits"][0].softmax(-1)
            visibility = output["visibility_logits"][0].sigmoid()
            boundaries = output["boundary_logits"][0].sigmoid()
            start_logits = output.get("start_logits")
            end_logits = output.get("end_logits")
            quality = output.get("quality_logits")
            if quality is None:
                quality = torch.ones(
                    output["class_logits"].shape[1], device=class_prob.device
                )
            else:
                quality = quality[0].sigmoid()
            boxes = output["boxes"][0]
            if args.hflip:
                boxes = unflip_normalized_boxes(boxes)
            for query in range(boxes.shape[0]):
                query_boxes, query_visibility = expand_tubelet(
                    boxes[query], visibility[query], clip_length
                )
                endpoint_values = {}
                if start_logits is not None and end_logits is not None:
                    for endpoint, logits in (
                            ("start", start_logits[0, query]),
                            ("end", end_logits[0, query])):
                        expanded = F.interpolate(
                            logits.view(1, 1, -1),
                            size=clip_length,
                            mode="linear",
                            align_corners=True,
                        )[0, 0].softmax(dim=-1)
                        local_frame = int(expanded.argmax())
                        endpoint_values.update({
                            f"{endpoint}_frame": (
                                min(start + local_frame, num_frames) - 1
                            ),
                            f"{endpoint}_confidence": float(expanded[local_frame]),
                            f"{endpoint}_censored": (
                                local_frame <= 2 or local_frame >= clip_length - 3
                            ),
                        })
                query_boundary = F.interpolate(
                    boundaries[query].view(1, 1, -1),
                    size=clip_length,
                    mode="linear",
                    align_corners=True,
                )[0, 0]
                selected = query_visibility >= args.visibility
                # Candidate filter (role 1 of --min_length, see the parser).
                if int(selected.sum()) < args.min_length:
                    continue
                label = int(class_prob[query].argmax())
                if label >= cfg["model"]["num_classes"]:
                    continue
                detections, weights, boundary_scores = {}, {}, {}
                for local_frame in range(clip_length):
                    global_frame = min(start + local_frame, num_frames) - 1
                    boundary_scores[global_frame] = float(query_boundary[local_frame])
                for local_frame in selected.nonzero(as_tuple=True)[0].tolist():
                    global_frame = min(start + local_frame, num_frames) - 1
                    detections[global_frame] = query_boxes[local_frame].cpu().numpy()
                    weights[global_frame] = float(query_visibility[local_frame])
                    if (args.frame_map and
                            global_frame in corrected_frames[video_name]):
                        tube_frame_candidates[global_frame].append({
                            "class": label,
                            "score": float(
                                class_prob[query, label]
                                * quality[query]
                                * query_visibility[local_frame]
                                * clip_weights[start][local_frame]
                            ),
                            "box": query_boxes[local_frame].cpu().numpy(),
                        })
                score = float(
                    class_prob[query, label]
                    * quality[query]
                    * query_visibility[selected].mean()
                )
                candidate = {
                    "video": video_name,
                    "resolution": resolution,
                    "clip_start": start - 1,
                    "class": label,
                    "score": score,
                    "scores": [score],
                    "detections": detections,
                    "frame_weights": weights,
                    "boundary_scores": boundary_scores,
                }
                candidate.update(endpoint_values)
                candidates.append(candidate)
        if args.candidate_cache:
            cached_videos[video_name] = {
                "resolution": resolution,
                "clip_starts": [start - 1 for start in clip_starts],
                "candidates": candidates,
            }
        if not args.cache_only:
            all_moc_predictions.extend(link_tubelets_moc(
                candidates,
                [start - 1 for start in clip_starts],
                clip_length,
                link_iou=args.moc_link_iou,
                tubelet_nms=args.moc_tubelet_nms,
                top_k=args.moc_top_k,
                min_length=args.min_length,
                split_gap=args.moc_split_gap if args.moc_split_gap >= 0 else None,
                boundary_split_thresh=args.boundary_split_thresh,
                endpoint_tolerance=(
                    args.endpoint_tolerance if args.endpoint_tolerance >= 0 else None
                ),
                endpoint_transition_margin=args.endpoint_transition_margin,
                endpoint_min_confidence=args.endpoint_min_confidence,
            ))
            merged_predictions = merge_candidates(
                candidates, args.merge_iou, args.min_length
            )
            all_predictions.extend(merged_predictions)
            all_ground_truth.extend(gt_by_video.get(video_name, []))
        if args.frame_dump:
            height, width = resolution
            for frame_id in sorted(dense_frame_candidates):
                for detection in resolve_frame_candidates(
                        dense_frame_candidates[frame_id], args.frame_nms_thresh):
                    x1, y1, x2, y2 = (float(v) for v in detection["box"])
                    frame_dump_rows.append((
                        video_name, int(frame_id) + 1, int(detection["class"]),
                        float(detection["score"]),
                        x1 * width, y1 * height, x2 * width, y2 * height,
                    ))
        if args.frame_map:
            for frame_id in corrected_frames[video_name]:
                dense = resolve_frame_candidates(
                    dense_frame_candidates.get(frame_id, []), args.frame_nms_thresh
                )
                tubelet = resolve_frame_candidates(
                    tube_frame_candidates.get(frame_id, []), args.frame_nms_thresh
                )
                scaled_tubelet = [
                    {**item, "score": item["score"] * args.tube_frame_scale}
                    for item in tubelet
                ]
                fused = resolve_frame_candidates(
                    dense + scaled_tubelet, args.frame_nms_thresh
                )
                add_frame_predictions(
                    frame_predictions["dense"], video_name, frame_id,
                    dense, resolution,
                )
                add_frame_predictions(
                    frame_predictions["tubelet"], video_name, frame_id,
                    tubelet, resolution,
                )
                add_frame_predictions(
                    frame_predictions["fused"], video_name, frame_id,
                    fused, resolution,
                )
        if (video_index + 1) % 100 == 0:
            print(f"Processed {video_index + 1}/{len(videos)}", flush=True)

    if args.frame_dump:
        os.makedirs(os.path.dirname(os.path.abspath(args.frame_dump)), exist_ok=True)
        with open(args.frame_dump, "wb") as handle:
            pickle.dump({
                "version": 1,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "annot_file": cfg["data"]["annot_file"],
                "clip_weighting": args.clip_weighting,
                "clip_overlap": args.clip_overlap,
                "frame_conf_thresh": args.frame_conf_thresh,
                "frame_nms_thresh": args.frame_nms_thresh,
                "shard": [args.shard_index, args.num_shards],
                "frame_index_base": 1,
                "box_units": "pixels",
                "hflip": bool(args.hflip),
                "rows": frame_dump_rows,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(frame_dump_rows)} frame detections: {args.frame_dump}")

    if args.candidate_cache:
        cache_dir = os.path.dirname(os.path.abspath(args.candidate_cache))
        os.makedirs(cache_dir, exist_ok=True)
        with open(args.candidate_cache, "wb") as handle:
            pickle.dump({
                "version": 1,
                "config": args.config,
                "clip_weighting": args.clip_weighting,
                "clip_overlap": args.clip_overlap,
                "shard": [args.shard_index, args.num_shards],
                "checkpoint": args.checkpoint,
                "annot_file": cfg["data"]["annot_file"],
                "metadata_json": args.metadata_json,
                "clip_length": clip_length,
                "num_classes": cfg["model"]["num_classes"],
                "visibility": args.visibility,
                "min_length": args.min_length,
                "hflip": bool(args.hflip),
                "videos": cached_videos,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved raw tubelet candidates: {args.candidate_cache}")

    if args.num_shards > 1:
        print("Sharded run: candidates cached; merge shards before scoring.")
        return

    if args.cache_only:
        print("Cache-only mode: label-free metadata used; metrics were not computed.")
        print(f"Elapsed: {time.time() - started:.0f}s")
        return

    num_classes = cfg["model"]["num_classes"]
    results = compute_video_map(
        all_predictions, all_ground_truth, num_classes=num_classes
    )
    official_thresholds = (0.2,) + tuple(np.arange(0.5, 0.951, 0.05))
    official_merged = compute_video_map_official(
        all_predictions, all_ground_truth,
        iou_thresholds=official_thresholds,
        num_classes=num_classes,
        tube_nms=args.moc_tube_nms,
    )
    official_moc = compute_video_map_official(
        all_moc_predictions, all_ground_truth,
        iou_thresholds=official_thresholds,
        num_classes=num_classes,
        tube_nms=args.moc_tube_nms,
    )
    print(f"Current-merger predicted tubes: {len(all_predictions)}")
    print(f"MOC-linker predicted tubes: {len(all_moc_predictions)}")
    print(
        "Legacy flattened direct-query video-mAP@0.2/@0.5: "
        f"{100 * results[0.2]['mAP']:.2f}% / "
        f"{100 * results[0.5]['mAP']:.2f}%"
    )
    for name, protocol_results in (
        ("Current merger", official_merged),
        ("MOC-style linker", official_moc),
    ):
        sweep = np.mean([
            protocol_results[float(threshold)]["mAP"]
            for threshold in official_thresholds[1:]
        ])
        print(
            f"{name} official video-mAP@0.2/@0.5/@0.5:0.95: "
            f"{100 * protocol_results[0.2]['mAP']:.2f}% / "
            f"{100 * protocol_results[0.5]['mAP']:.2f}% / "
            f"{100 * sweep:.2f}%"
        )
    if args.frame_map:
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
        for source, predictions in frame_predictions.items():
            corrected_map, _ = compute_frame_map(
                predictions, corrected_gt,
                num_classes=cfg["model"]["num_classes"],
                evaluated_frames=corrected_eval_frames,
            )
            yowo_map, _ = compute_frame_map(
                predictions, yowo_gt,
                num_classes=cfg["model"]["num_classes"],
                evaluated_frames=yowo_eval_frames,
            )
            print(
                f"{source.capitalize()} unique-frame VOC mAP@0.5 "
                f"(corrected/YOWO): {100 * corrected_map:.2f}% / "
                f"{100 * yowo_map:.2f}%"
            )
    print(f"Elapsed: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
