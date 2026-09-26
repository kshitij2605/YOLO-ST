"""Fuse the unflipped and horizontally flipped views of one evaluation (flip test-time augmentation).

The inputs are two eval_tube_queries.py runs of the same model on the same videos, the second with --hflip,
so its boxes are already in the unflipped frame. Scores are averaged over the two views, a view that misses a
detection contributing 0: a detection seen in one view only keeps half its score. Matched boxes are averaged
with score weights. Settings are fixed in advance (no sweep): frame IoU 0.55, tube overlap 0.5, frame NMS 0.5.

Frame dumps: per video, frame and class, the detections of both views are taken in descending score order and
each joins the best-overlapping cluster (IoU >= --frame-iou with the cluster's fused box) that has no member
from its view yet, else starts a new cluster. The fused rows then pass the evaluator's class-aware NMS.

Candidate caches: per video, clip start and class, candidates are clustered the same way using the tube overlap
of eval_tube_queries-style candidates: mean per-frame IoU over shared frames times the share of frames in
common. Per-frame boxes of a cluster are score-weighted means; per-frame visibility weights and boundary scores
are averaged over the members that have the frame; endpoint fields come from the highest-scoring member.

Usage: python3 tools/fuse_flip_views.py --dumps PLAIN FLIPPED --caches PLAIN FLIPPED --out-dump D --out-cache C
"""
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
import torchvision

ENDPOINT_KEYS = ("start_frame", "start_confidence", "start_censored",
                 "end_frame", "end_confidence", "end_censored")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def cluster(items, overlap, threshold):
    """items: (score, view, payload) in any order; returns clusters as {view: (score, payload)}."""
    clusters = []
    for score, view, payload in sorted(items, key=lambda item: -item[0]):
        best, best_value = None, threshold
        for members in clusters:
            if view in members:
                continue
            value = overlap(members, payload)
            if value >= best_value:
                best, best_value = members, value
        if best is None:
            clusters.append({view: (score, payload)})
        else:
            best[view] = (score, payload)
    return clusters


def fused_box(members):
    weights = np.array([score for score, _ in members.values()], dtype=np.float64)
    boxes = np.stack([np.asarray(box, dtype=np.float64) for _, box in members.values()])
    return (weights[:, None] * boxes).sum(0) / max(weights.sum(), 1e-12)


def fuse_dumps(plain, flipped, frame_iou, frame_nms):
    groups = defaultdict(list)
    for view, dump in enumerate((plain, flipped)):
        for video, frame, cls, score, x1, y1, x2, y2 in dump["rows"]:
            groups[(video, frame, cls)].append((float(score), view, np.array([x1, y1, x2, y2])))
    rows = []
    for (video, frame, cls), items in groups.items():
        clusters = cluster(items, lambda members, box: iou(fused_box(members), box), frame_iou)
        fused = [(sum(score for score, _ in members.values()) / 2.0, fused_box(members)) for members in clusters]
        keep = torchvision.ops.nms(torch.as_tensor(np.stack([box for _, box in fused]), dtype=torch.float32),
                                   torch.as_tensor([score for score, _ in fused], dtype=torch.float32),
                                   frame_nms).tolist()
        for index in keep:
            score, box = fused[index]
            rows.append((video, frame, cls, float(score)) + tuple(float(v) for v in box))
    rows.sort(key=lambda row: (row[0], row[1], row[2], -row[3]))
    out = dict(plain)
    out["rows"] = rows
    out["tta"] = {"views": ["plain", "hflip"], "frame_iou": frame_iou, "frame_nms": frame_nms,
                  "rows_in": [len(plain["rows"]), len(flipped["rows"])]}
    return out


def tube_overlap(det_a, det_b):
    common = set(det_a) & set(det_b)
    if not common:
        return 0.0
    mean_iou = np.mean([iou(det_a[frame], det_b[frame]) for frame in common])
    return float(mean_iou * len(common) / len(set(det_a) | set(det_b)))


def merge_members(members):
    ordered = sorted(members.values(), key=lambda item: -item[0])
    top = ordered[0][1]
    merged = {key: value for key, value in top.items()}
    score = sum(score for score, _ in ordered) / 2.0
    merged["score"], merged["scores"] = score, [score]
    frames = sorted(set().union(*(candidate["detections"] for _, candidate in ordered)))
    detections, weights = {}, {}
    for frame in frames:
        have = [(s, c) for s, c in ordered if frame in c["detections"]]
        w = np.array([s for s, _ in have], dtype=np.float64)
        boxes = np.stack([np.asarray(c["detections"][frame], dtype=np.float64) for _, c in have])
        detections[frame] = ((w[:, None] * boxes).sum(0) / max(w.sum(), 1e-12)).astype(np.float32)
        weights[frame] = float(np.mean([c["frame_weights"][frame] for _, c in have]))
    boundary = {}
    for frame in sorted(set().union(*(c.get("boundary_scores", {}) for _, c in ordered))):
        values = [c["boundary_scores"][frame] for _, c in ordered if frame in c.get("boundary_scores", {})]
        boundary[frame] = float(np.mean(values))
    merged["detections"], merged["frame_weights"], merged["boundary_scores"] = detections, weights, boundary
    for key in ENDPOINT_KEYS:
        if key in top:
            merged[key] = top[key]
    merged["tta_views"] = sorted(members)
    return merged


def fuse_caches(plain, flipped, tube_iou):
    out = dict(plain)
    videos = {}
    for video in sorted(set(plain["videos"]) | set(flipped["videos"])):
        entry_a = plain["videos"].get(video) or flipped["videos"][video]
        groups = defaultdict(list)
        for view, cache in enumerate((plain, flipped)):
            for candidate in cache["videos"].get(video, {}).get("candidates", []):
                groups[(candidate["clip_start"], candidate["class"])].append(
                    (float(candidate["score"]), view, candidate))
        fused = []
        for key in sorted(groups):
            clusters = cluster(groups[key], lambda members, cand: tube_overlap(
                merge_members(members)["detections"], cand["detections"]), tube_iou)
            fused.extend(merge_members(members) for members in clusters)
        videos[video] = {"resolution": entry_a["resolution"], "clip_starts": entry_a["clip_starts"],
                         "candidates": fused}
    out["videos"] = videos
    out["tta"] = {"views": ["plain", "hflip"], "tube_iou": tube_iou}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dumps", nargs=2, required=True, metavar=("PLAIN", "FLIPPED"))
    parser.add_argument("--caches", nargs=2, required=True, metavar=("PLAIN", "FLIPPED"))
    parser.add_argument("--out-dump", required=True)
    parser.add_argument("--out-cache", required=True)
    parser.add_argument("--frame-iou", type=float, default=0.55)
    parser.add_argument("--tube-iou", type=float, default=0.5)
    parser.add_argument("--frame-nms", type=float, default=0.5)
    args = parser.parse_args()
    dumps = [pickle.load(open(path, "rb")) for path in args.dumps]
    caches = [pickle.load(open(path, "rb")) for path in args.caches]
    for kind, pair in (("dump", dumps), ("cache", caches)):
        if bool(pair[0].get("hflip")) or not bool(pair[1].get("hflip")):
            raise SystemExit("%s views must be PLAIN then --hflip" % kind)
        if pair[0].get("checkpoint") != pair[1].get("checkpoint"):
            raise SystemExit("%s views come from different checkpoints" % kind)
    pickle.dump(fuse_dumps(dumps[0], dumps[1], args.frame_iou, args.frame_nms), open(args.out_dump, "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    fused = fuse_caches(caches[0], caches[1], args.tube_iou)
    pickle.dump(fused, open(args.out_cache, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    counts = [sum(len(v["candidates"]) for v in c["videos"].values()) for c in caches + [fused]]
    print("fused %d + %d dump rows; candidates %d + %d -> %d" % (
        len(dumps[0]["rows"]), len(dumps[1]["rows"]), counts[0], counts[1], counts[2]))


if __name__ == "__main__":
    main()
