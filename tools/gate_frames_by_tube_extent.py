#!/usr/bin/env python3
"""IG1: gate dense frame detections by the temporal extent of linked tubes.

In YOLO-ST the tube queries own temporal extent and the dense head owns frame
geometry. The frozen B2 snap moves dense geometry into tubes; IG1 is the
converse transfer. A dense detection at frame f keeps its score only when a
linked tube (frozen MOC-style linker) with score >= tau covers f, the tube's
extent dilated by delta frames; otherwise its score is multiplied by eps
(eps = 0 removes it). Coverage is taken over tubes of any class (mode "any")
or of the detection's own class (mode "cls").

Detections that the dense head fires outside every predicted action interval
are false positives under the all-frame frame-mAP protocol (ROAD test-ucf24.py
with full_test=True; MOC/ACT frameAP) but are invisible to the annotated-frame
convention. Every setting is therefore scored three ways on the same detections:
  annotated_voc : annotated frames only, VOC every-point AP (reported convention)
  all_voc       : every frame, VOC every-point AP (ROAD full_test)
  all_moc       : every frame, trapezoid on the raw PR curve (MOC/ACT pr_to_ap)
Matching follows MOC ACT.py frameAP: per class, score order, best remaining GT
box in the same frame, inclusive (+1) IoU >= 0.5, matched box removed. Boxes are
np.rint-ed and clipped exactly as frame_map_protocol.normalized_to_pixels does,
so the ungated annotated_voc reproduces the recorded frame mAP.

Tubes are linked once per video with the frozen linker, exactly as
tools/snap_tubes_to_dense_geometry.py does. The tube cache must be built with
candidate min_length 8 (the setting the frozen linker was validated on).

Read-only except for the single --out JSON.
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from frame_map_protocol import load_frame_ground_truth, voc_ap  # noqa: E402
from video_map_protocol import link_tubelets_moc  # noqa: E402

FROZEN_LINKER = dict(link_iou=0.45, tubelet_nms=0.6, top_k=10, min_length=16, split_gap=2)
ANNOTATED_TOLERANCE = 0.5   # pre-registered: annotated_voc may drop at most this much
TIE_WINDOW = 0.05           # pre-registered: settings within this of the best all_voc tie


def parse_args():
    p = argparse.ArgumentParser(description="IG1 tube-extent frame gating")
    p.add_argument("--tube-cache", required=True)
    p.add_argument("--frame-dump", required=True)
    p.add_argument("--mode", nargs="+", default=["any", "cls"], choices=["any", "cls"])
    p.add_argument("--tau", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3])
    p.add_argument("--delta", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    p.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    p.add_argument("--expect-annotated", type=float, default=None,
                   help="recorded ungated frame mAP; abort if not reproduced within 0.02")
    p.add_argument("--out", required=True)
    return p.parse_args()


def iou2d(gt, box):
    """ACT_utils.iou2d: inclusive (+1) pixel extents."""
    xmin = np.maximum(gt[:, 0], box[0])
    ymin = np.maximum(gt[:, 1], box[1])
    xmax = np.minimum(gt[:, 2] + 1, box[2] + 1)
    ymax = np.minimum(gt[:, 3] + 1, box[3] + 1)
    overlap = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    return overlap / (area_gt + area_box - overlap)


def ap_voc(tp, npos):
    if len(tp) == 0 or npos == 0:
        return 0.0
    ctp = np.cumsum(tp).astype(np.float64)
    cfp = np.cumsum(~tp).astype(np.float64)
    return float(voc_ap(ctp / float(npos), ctp / np.maximum(ctp + cfp, 1e-12)))


def ap_moc(tp, npos):
    """ACT pr_to_ap on the raw curve with the (precision 1, recall 0) start point."""
    if len(tp) == 0 or npos == 0:
        return 0.0
    ctp = np.cumsum(tp).astype(np.float32)
    cfp = np.cumsum(~tp).astype(np.float32)
    pr = np.empty((len(tp) + 1, 2), dtype=np.float32)
    pr[0] = (1.0, 0.0)
    pr[1:, 0] = ctp / np.maximum(ctp + cfp, 1)
    pr[1:, 1] = ctp / float(npos)
    return float(np.sum((pr[1:, 1] - pr[:-1, 1]) * (pr[1:, 0] + pr[:-1, 0]) * 0.5))


def main():
    args = parse_args()
    t0 = time.time()
    with open(args.tube_cache, "rb") as handle:
        cache = pickle.load(handle)
    with open(args.frame_dump, "rb") as handle:
        dump = pickle.load(handle)
    if dump.get("box_units", "pixels") != "pixels":
        raise SystemExit("unexpected box_units %r" % dump.get("box_units"))
    if os.path.normpath(dump["annot_file"]) != os.path.normpath(cache["annot_file"]):
        raise SystemExit("tube cache and frame dump use different annotation files")
    if int(cache.get("min_length", 8)) != 8:
        raise SystemExit("tube cache built with candidate min_length %s; the frozen linker "
                         "was validated on caches built with 8" % cache.get("min_length"))
    with open(cache["annot_file"], "rb") as handle:
        annot = pickle.load(handle, encoding="latin1")
    videos = list(annot["test_videos"][0])
    if set(videos) != set(cache["videos"]):
        raise SystemExit("tube-cache videos differ from the evaluation partition")
    nframes = {v: int(annot["nframes"][v]) for v in videos}
    resolution = {v: tuple(annot["resolution"].get(v, (240, 320))) for v in videos}
    num_classes = int(cache["num_classes"])

    # 1. link once with the frozen protocol
    spans = {v: [] for v in videos}
    n_tubes = 0
    for v in videos:
        data = cache["videos"][v]
        tubes = link_tubelets_moc(data["candidates"], data["clip_starts"], cache["clip_length"],
                                  **FROZEN_LINKER)
        n_tubes += len(tubes)
        for tube in tubes:
            frames = sorted(int(f) for f in tube["detections"])
            if frames:
                spans[v].append((int(tube["class"]), float(tube["score"]), frames[0], frames[-1]))

    # 2. ground truth (corrected annotations, 0-based frames)
    gt_by_class, frames_by_video = load_frame_ground_truth(annot, videos)
    annotated = {(v, f) for v, fs in frames_by_video.items() for f in fs}
    gt_index = []
    for c in range(num_classes):
        grouped = defaultdict(list)
        for key, box in gt_by_class.get(c, []):
            grouped[key].append(np.asarray(box, dtype=np.float32))
        gt_index.append({k: np.stack(b) for k, b in grouped.items()})
    npos = [sum(b.shape[0] for b in g.values()) for g in gt_index]

    # 3. detections, boxes as the recorded pipeline scored them
    base = int(dump.get("frame_index_base", 1))
    rows = [r for r in dump["rows"] if r[0] in nframes]
    n = len(rows)
    det_video = [None] * n
    det_frame = np.empty(n, np.int64)
    det_class = np.empty(n, np.int64)
    det_score = np.empty(n, np.float64)
    det_box = np.empty((n, 4), np.float32)
    for i, (v, f, c, s, x1, y1, x2, y2) in enumerate(rows):
        h, w = resolution[v]
        box = np.rint(np.asarray([x1, y1, x2, y2], dtype=np.float32))
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h - 1)
        det_video[i], det_frame[i], det_class[i], det_score[i], det_box[i] = v, int(f) - base, int(c), float(s), box
    keys = list(zip(det_video, det_frame.tolist()))
    on_annotated = np.fromiter((k in annotated for k in keys), dtype=bool, count=n)
    by_class = [np.nonzero(det_class == c)[0] for c in range(num_classes)]

    # 4. tube coverage per detection for every delta
    cov_any, cov_cls = {}, {}
    for d in sorted(set(args.delta)):
        per_any, per_cls = {}, {}
        for v in videos:
            a = np.zeros(nframes[v])
            m = np.zeros((nframes[v], num_classes))
            for c, s, lo, hi in spans[v]:
                lo2, hi2 = max(0, lo - d), min(nframes[v] - 1, hi + d)
                a[lo2:hi2 + 1] = np.maximum(a[lo2:hi2 + 1], s)
                m[lo2:hi2 + 1, c] = np.maximum(m[lo2:hi2 + 1, c], s)
            per_any[v], per_cls[v] = a, m
        cov_any[d] = np.fromiter((per_any[v][f] if f < nframes[v] else 0.0 for v, f in keys),
                                 dtype=np.float64, count=n)
        cov_cls[d] = np.fromiter((per_cls[v][f, c] if f < nframes[v] else 0.0
                                  for (v, f), c in zip(keys, det_class.tolist())),
                                 dtype=np.float64, count=n)

    def evaluate(scores, keep):
        out = {}
        for frame_set in ("annotated", "all"):
            selected = keep & on_annotated if frame_set == "annotated" else keep
            aps_voc, aps_moc = [], []
            for c in range(num_classes):
                idx = by_class[c][selected[by_class[c]]]
                order = idx[np.argsort(-scores[idx], kind="stable")]
                remaining = {k: b.copy() for k, b in gt_index[c].items()}
                tp = np.zeros(len(order), dtype=bool)
                for r, j in enumerate(order):
                    g = remaining.get(keys[j])
                    if g is None:
                        continue
                    ious = iou2d(g, det_box[j])
                    a = int(np.argmax(ious))
                    if ious[a] >= 0.5:
                        tp[r] = True
                        g = np.delete(g, a, 0)
                        if g.shape[0]:
                            remaining[keys[j]] = g
                        else:
                            del remaining[keys[j]]
                aps_voc.append(ap_voc(tp, npos[c]))
                if frame_set == "all":
                    aps_moc.append(ap_moc(tp, npos[c]))
            out[frame_set + "_voc"] = round(100.0 * float(np.mean(aps_voc)), 3)
            if frame_set == "all":
                out["all_moc"] = round(100.0 * float(np.mean(aps_moc)), 3)
        out["kept_unannotated"] = int((keep & ~on_annotated).sum())
        out["kept_annotated"] = int((keep & on_annotated).sum())
        return out

    all_keep = np.ones(n, dtype=bool)
    baseline = evaluate(det_score, all_keep)
    print("tubes %d | detections %d (unannotated %d) | ungated: annotated_voc %.2f  all_voc %.2f  all_moc %.2f"
          % (n_tubes, n, int((~on_annotated).sum()), baseline["annotated_voc"],
             baseline["all_voc"], baseline["all_moc"]), flush=True)
    if args.expect_annotated is not None and abs(baseline["annotated_voc"] - args.expect_annotated) > 0.02:
        raise SystemExit("GATE FAIL: ungated annotated_voc %.3f does not reproduce recorded %.2f"
                         % (baseline["annotated_voc"], args.expect_annotated))

    rows_out = []
    for mode in args.mode:
        for d in args.delta:
            coverage = cov_any[d] if mode == "any" else cov_cls[d]
            for tau in args.tau:
                covered = (coverage > 0) & (coverage >= tau)
                for eps in args.eps:
                    scores = np.where(covered, det_score, det_score * eps)
                    keep = scores > 0
                    res = evaluate(scores, keep)
                    res.update({"mode": mode, "delta": d, "tau": tau, "eps": eps})
                    rows_out.append(res)
                    print("mode=%s delta=%2d tau=%.2f eps=%.1f | annotated_voc %.2f  all_voc %.2f  all_moc %.2f"
                          % (mode, d, tau, eps, res["annotated_voc"], res["all_voc"], res["all_moc"]), flush=True)

    # pre-registered selection rule
    floor = baseline["annotated_voc"] - ANNOTATED_TOLERANCE
    eligible = [r for r in rows_out if r["annotated_voc"] >= floor]
    selected = None
    if eligible:
        best = max(r["all_voc"] for r in eligible)
        tied = [r for r in eligible if r["all_voc"] >= best - TIE_WINDOW]
        tied.sort(key=lambda r: (r["eps"] != 0.0, r["mode"] != "any", r["delta"], r["tau"]))
        selected = tied[0]
    rows_out.sort(key=lambda r: -r["all_voc"])
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({
            "tube_cache": os.path.abspath(args.tube_cache),
            "frame_dump": os.path.abspath(args.frame_dump),
            "annot_file": cache["annot_file"],
            "linker": FROZEN_LINKER,
            "tubes": n_tubes,
            "detections": n,
            "baseline": baseline,
            "selection_rule": {"maximize": "all_voc", "annotated_floor": floor,
                               "tie_window": TIE_WINDOW,
                               "tie_break": "eps==0, mode any, smaller delta, smaller tau"},
            "selected": selected,
            "rows": rows_out,
            "elapsed_s": round(time.time() - t0, 1),
        }, handle, indent=1)
    print("SELECTED:", json.dumps(selected))
    print("elapsed %.0fs -> %s" % (time.time() - t0, args.out))


if __name__ == "__main__":
    main()
