#!/usr/bin/env python3
"""IG2: temporal erosion of dense frame detections (post-hoc, frame metrics only).

For each video and class, frames whose best detection of that class scores at
least s_min form segments (gaps up to GAP frames are bridged). Detections of that
class lying in the outer k frames of a segment have their score multiplied by eps.
The rest is untouched. Frame mAP is then scored on annotated frames and on every
frame (VOC every-point, ACT matching, inclusive IoU, as tools/score_frame_conventions.py).

Modes
  grid   : score every setting of the registered grid on one dump, write JSON
  apply  : score one setting (e.g. the frozen one on confirm or test)
"""
import argparse
import itertools
import json
import pickle
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, ".")
from frame_map_protocol import load_frame_ground_truth, voc_ap  # noqa: E402

GAP = 2
GRID = {"k": [2, 4, 8, 12, 16], "eps": [0.0, 0.3, 0.6], "s_min": [0.05, 0.2]}

_STATE = {}


def load(dump_path):
    d = pickle.load(open(dump_path, "rb"))
    annot = pickle.load(open(d["annot_file"], "rb"), encoding="latin1")
    videos = list(annot["test_videos"][0])
    gt, frames = load_frame_ground_truth(annot, videos)
    annotated = {(v, f) for v, fs in frames.items() for f in fs}
    base = int(d["frame_index_base"])
    res = annot["resolution"]
    rows = []
    for v, f, c, s, x1, y1, x2, y2 in d["rows"]:
        h, w = res.get(v, (240, 320))
        b = np.rint(np.array([x1, y1, x2, y2], dtype=np.float32))
        b[[0, 2]] = np.clip(b[[0, 2]], 0, w - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h - 1)
        rows.append((v, int(f) - base, int(c), float(s), b))
    return rows, gt, annotated, len(annot["labels"])


def eroded_scores(rows, k, eps, s_min):
    best = defaultdict(float)
    for v, f, c, s, _ in rows:
        key = (v, c, f)
        if s > best[key]:
            best[key] = s
    frames_by = defaultdict(list)
    for (v, c, f), s in best.items():
        if s >= s_min:
            frames_by[(v, c)].append(f)
    edge = {}
    for (v, c), fs in frames_by.items():
        fs.sort()
        start = prev = fs[0]
        segments = []
        for f in fs[1:]:
            if f - prev > GAP + 1:
                segments.append((start, prev))
                start = f
            prev = f
        segments.append((start, prev))
        for a, b in segments:
            for f in range(a, b + 1):
                edge[(v, c, f)] = min(f - a, b - f)
    out = []
    for v, f, c, s, _ in rows:
        e = edge.get((v, c, f))
        out.append(s * eps if (e is not None and e < k) else s)
    return np.asarray(out, dtype=np.float64)


def iou2d(g, b):
    xmin = np.maximum(g[:, 0], b[0]); ymin = np.maximum(g[:, 1], b[1])
    xmax = np.minimum(g[:, 2] + 1, b[2] + 1); ymax = np.minimum(g[:, 3] + 1, b[3] + 1)
    ov = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    ag = (g[:, 2] - g[:, 0] + 1) * (g[:, 3] - g[:, 1] + 1)
    ab = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return ov / (ag + ab - ov)


def voc_map(rows, scores, gt, num_classes, frame_filter):
    aps = []
    for cls in range(num_classes):
        g = defaultdict(list)
        for key, box in gt.get(cls, []):
            g[key].append(np.asarray(box, dtype=np.float32))
        g = {key: np.stack(x) for key, x in g.items()}
        npos = sum(x.shape[0] for x in g.values())
        idx = [i for i, r in enumerate(rows) if r[2] == cls and
               (frame_filter is None or (r[0], r[1]) in frame_filter)]
        idx.sort(key=lambda i: -scores[i])
        tp = np.zeros(len(idx), dtype=bool)
        for n, i in enumerate(idx):
            key = (rows[i][0], rows[i][1])
            if key in g:
                ious = iou2d(g[key], rows[i][4]); j = int(np.argmax(ious))
                if ious[j] >= 0.5:
                    tp[n] = True; g[key] = np.delete(g[key], j, 0)
                    if g[key].size == 0:
                        del g[key]
        ctp = np.cumsum(tp).astype(float); cfp = np.cumsum(~tp).astype(float)
        aps.append(voc_ap(ctp / max(npos, 1), ctp / np.maximum(ctp + cfp, 1e-12)) if npos else 0.0)
    return 100.0 * float(np.mean(aps))


def score_setting(setting):
    rows, gt, annotated, nc = _STATE["data"]
    k, eps, s_min = setting
    scores = eroded_scores(rows, k, eps, s_min) if k > 0 else np.asarray([r[3] for r in rows])
    return {"k": k, "eps": eps, "s_min": s_min,
            "annotated_voc": voc_map(rows, scores, gt, nc, annotated),
            "allframe_voc": voc_map(rows, scores, gt, nc, None),
            "affected": int(np.sum(scores != np.asarray([r[3] for r in rows])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grid", "apply"])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int); ap.add_argument("--eps", type=float); ap.add_argument("--s-min", type=float)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    _STATE["data"] = load(args.dump)
    if args.mode == "grid":
        settings = [(0, 1.0, 0.0)] + list(itertools.product(GRID["k"], GRID["eps"], GRID["s_min"]))
        with Pool(args.workers) as pool:  # forked workers inherit _STATE
            results = pool.map(score_setting, settings)
        json.dump({"dump": args.dump, "gap": GAP, "grid": GRID, "baseline": results[0], "rows": results[1:]},
                  open(args.out, "w"), indent=1)
        base = results[0]
        print("@@ baseline annotated %.2f every-frame %.2f" % (base["annotated_voc"], base["allframe_voc"]))
        for r in sorted(results[1:], key=lambda r: -r["allframe_voc"])[:8]:
            print("@@ k=%2d eps=%.1f s_min=%.2f  annotated %.2f  every-frame %.2f  affected %d" % (
                r["k"], r["eps"], r["s_min"], r["annotated_voc"], r["allframe_voc"], r["affected"]))
    else:
        result = score_setting((args.k, args.eps, args.s_min))
        base = score_setting((0, 1.0, 0.0))
        json.dump({"dump": args.dump, "gap": GAP, "setting": result, "baseline": base}, open(args.out, "w"), indent=1)
        print("@@ baseline annotated %.2f every-frame %.2f | eroded annotated %.2f every-frame %.2f" % (
            base["annotated_voc"], base["allframe_voc"], result["annotated_voc"], result["allframe_voc"]))


if __name__ == "__main__":
    main()
