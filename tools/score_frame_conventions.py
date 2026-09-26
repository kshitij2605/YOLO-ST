"""Score one recorded UCF101-24 frame dump under every published frame-mAP convention.

Read-only: takes a frame dump written by eval_tube_queries.py --frame_dump and
prints/writes frame mAP (%) under

  annotated_voc        annotated frames, VOC every-point      (repo compute_frame_map)
  yowo_list_voc        YOWO's own list (last frame of each tube omitted), VOC
  annotated_trapz      annotated frames, MOC/ACT trapezoid on the raw PR curve
  allframe_voc         every frame, VOC every-point            (ROAD full_test protocol)
  allframe_trapz       every frame, MOC/ACT trapezoid          (MOC frameAP)
  yowoformer_style     YOWOv3/YOWOFormer routine re-implemented: exclusive IoU at
                       224x224, IoU-ordered matching, trapezoid with a (1, 0) end
                       point; keeps ground truth on frames without detections
  yowoformer_exact     same, but drops that ground truth as the released loop does

Matching for the annotated/allframe trapz and VOC rows copies ACT frameAP: per
class, detections in score order, best remaining GT box of the frame, inclusive
(+1) IoU >= 0.5. Boxes get np.rint + clip exactly as frame_map_protocol does.
The gate compares annotated_voc from compute_frame_map with the ACT-style
annotated VOC value; they must agree.

Usage: python3 score_frame_conventions.py --dump PKL --out JSON [--expect-annotated X]
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from frame_map_protocol import compute_frame_map, load_frame_ground_truth, voc_ap

EPS = 1e-16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-annotated", type=float, default=None)
    args = ap.parse_args()

    d = pickle.load(open(args.dump, "rb"))
    annot = pickle.load(open(d["annot_file"], "rb"), encoding="latin1")
    videos = list(annot["test_videos"][0])
    res = annot["resolution"]
    num_classes = len(annot["labels"])
    gt, frames = load_frame_ground_truth(annot, videos)
    ygt, yframes = load_frame_ground_truth(annot, videos, drop_tube_terminal=True)
    annotated = {(v, f) for v, fs in frames.items() for f in fs}
    yowo_set = {(v, f) for v, fs in yframes.items() for f in fs}
    base = int(d["frame_index_base"])
    if str(d.get("box_units", "pixels")) != "pixels":
        raise SystemExit("expected pixel boxes, got %r" % d.get("box_units"))
    total_frames = sum(int(annot["nframes"][v]) for v in videos)

    def to_pix(box, v):
        h, w = res.get(v, (240, 320))
        b = np.rint(np.asarray(box, dtype=np.float32))
        b[[0, 2]] = np.clip(b[[0, 2]], 0, w - 1)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h - 1)
        return b

    dets = defaultdict(list)
    raw_by_frame = defaultdict(list)
    for v, f, c, s, x1, y1, x2, y2 in d["rows"]:
        key = (v, int(f) - base)
        dets[int(c)].append((float(s), key, to_pix([x1, y1, x2, y2], v)))
        raw_by_frame[key].append((np.array([x1, y1, x2, y2], dtype=np.float64), float(s), int(c)))

    def iou2d(g, b):
        xmin = np.maximum(g[:, 0], b[0]); ymin = np.maximum(g[:, 1], b[1])
        xmax = np.minimum(g[:, 2] + 1, b[2] + 1); ymax = np.minimum(g[:, 3] + 1, b[3] + 1)
        ov = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        ag = (g[:, 2] - g[:, 0] + 1) * (g[:, 3] - g[:, 1] + 1)
        ab = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return ov / (ag + ab - ov)

    def act_match(cls, frame_filter):
        g = defaultdict(list)
        for key, box in gt.get(cls, []):
            g[key].append(np.asarray(box, dtype=np.float32))
        g = {k: np.stack(v) for k, v in g.items()}
        npos = sum(v.shape[0] for v in g.values())
        ds = dets.get(cls, [])
        if frame_filter is not None:
            ds = [x for x in ds if x[1] in frame_filter]
        order = np.argsort(-np.asarray([x[0] for x in ds], dtype=np.float32))
        tp = np.zeros(len(ds), dtype=bool)
        for i, j in enumerate(order):
            _, k, box = ds[j]
            if k in g:
                ious = iou2d(g[k], box)
                a = int(np.argmax(ious))
                if ious[a] >= 0.5:
                    tp[i] = True
                    g[k] = np.delete(g[k], a, 0)
                    if g[k].size == 0:
                        del g[k]
        return tp, npos

    def ap_trapz(tp, npos):
        ctp = np.cumsum(tp).astype(np.float32); cfp = np.cumsum(~tp).astype(np.float32)
        pr = np.empty((len(tp) + 1, 2), dtype=np.float32)
        pr[0] = (1.0, 0.0)
        pr[1:, 0] = ctp / np.maximum(ctp + cfp, 1)
        pr[1:, 1] = ctp / float(npos)
        return float(np.sum((pr[1:, 1] - pr[:-1, 1]) * (pr[1:, 0] + pr[:-1, 0]) * 0.5))

    def ap_voc(tp, npos):
        ctp = np.cumsum(tp).astype(np.float64); cfp = np.cumsum(~tp).astype(np.float64)
        return float(voc_ap(ctp / float(npos), ctp / np.maximum(ctp + cfp, 1e-12)))

    out = {"dump": args.dump, "num_classes": num_classes, "videos": len(videos),
           "total_frames": total_frames, "annotated_frames": len(annotated),
           "rows": len(d["rows"]), "clip_weighting": d.get("clip_weighting"),
           "frame_conf_thresh": d.get("frame_conf_thresh"),
           "frame_nms_thresh": d.get("frame_nms_thresh")}
    for name, fs in (("annotated", annotated), ("allframe", None)):
        voc, trapz = [], []
        for cls in range(num_classes):
            tp, npos = act_match(cls, fs)
            voc.append(ap_voc(tp, npos)); trapz.append(ap_trapz(tp, npos))
        out[name + "_voc_act"] = 100.0 * float(np.mean(voc))
        out[name + "_trapz"] = 100.0 * float(np.mean(trapz))

    preds = defaultdict(list)
    for c in dets:
        preds[c].extend(dets[c])
    out["annotated_voc"] = 100.0 * compute_frame_map(
        preds, gt, num_classes=num_classes, evaluated_frames=annotated)[0]
    out["yowo_list_voc"] = 100.0 * compute_frame_map(
        preds, ygt, num_classes=num_classes, evaluated_frames=yowo_set)[0]
    out["allframe_voc"] = out.pop("allframe_voc_act")
    gate_gap = abs(out["annotated_voc"] - out["annotated_voc_act"])
    out["gate_repo_vs_act_annotated_voc"] = gate_gap

    # YOWOv3 / YOWOFormer routine
    S = 224.0
    gt_by_frame = defaultdict(list)
    for cid, entries in gt.items():
        for key, box in entries:
            h, w = res.get(key[0], (240, 320))
            gt_by_frame[key].append((np.array([box[0] / w * S, box[1] / h * S,
                                               box[2] / w * S, box[3] / h * S]), cid))

    def excl_iou(g, p):
        g = np.asarray(g, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
        lt = np.maximum(g[:, None, :2], p[None, :, :2]); rb = np.minimum(g[:, None, 2:], p[None, :, 2:])
        wh = np.clip(rb - lt, 0.0, None); inter = wh[..., 0] * wh[..., 1]
        ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]); apd = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        return inter / (ag[:, None] + apd[None, :] - inter + EPS)

    def yolo_match(pb, pc, gb, gc):
        correct = np.zeros(len(pb), dtype=bool)
        if not len(gb) or not len(pb):
            return correct
        iou = excl_iou(gb, pb)
        x = np.where((iou >= 0.5) & (np.asarray(gc)[:, None] == np.asarray(pc)[None, :]))
        if x[0].shape[0]:
            m = np.concatenate([np.stack(x, 1).astype(np.float64), iou[x[0], x[1]][:, None]], 1)
            if x[0].shape[0] > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int)] = True
        return correct

    def yolo_ap(tp, conf, pcls, tcls):
        o = np.argsort(-conf); tp, pcls = tp[o], pcls[o]
        uc, nt = np.unique(tcls, return_counts=True)
        aps = []
        for ci, c in enumerate(uc):
            sel = pcls == c
            if sel.sum() == 0 or nt[ci] == 0:
                aps.append(0.0); continue
            tpc = tp[sel].cumsum(); fpc = (~tp[sel]).cumsum()
            rec = tpc / (nt[ci] + EPS); pre = tpc / (tpc + fpc)
            mr = np.concatenate(([0.0], rec, [1.0])); mp = np.concatenate(([1.0], pre, [0.0]))
            mp = np.flip(np.maximum.accumulate(np.flip(mp)))
            aps.append(float(np.trapz(mp, mr)))
        return 100.0 * float(np.mean(aps))

    empty_frames = empty_gt = 0
    for mode in ("yowoformer_style", "yowoformer_exact"):
        tp, cf, pc, tc = [], [], [], []
        for key in annotated:
            g = gt_by_frame.get(key, []); p = raw_by_frame.get(key, [])
            if not p:
                if g and mode == "yowoformer_style":
                    empty_frames += 1; empty_gt += len(g)
                    tc.extend(e[1] for e in g)
                continue
            tc.extend(e[1] for e in g)
            h, w = res.get(key[0], (240, 320))
            pb = [np.array([b[0] / w * S, b[1] / h * S, b[2] / w * S, b[3] / h * S]) for b, _, _ in p]
            tp.append(yolo_match(pb, [e[2] for e in p], [e[0] for e in g], [e[1] for e in g]))
            cf.extend(e[1] for e in p); pc.extend(e[2] for e in p)
        out[mode] = yolo_ap(np.concatenate(tp), np.asarray(cf), np.asarray(pc), np.asarray(tc))
    out["annotated_frames_without_detections"] = empty_frames
    out["gt_boxes_on_frames_without_detections"] = empty_gt

    unann = [(s, k) for c in dets for s, k, _ in dets[c] if k not in annotated]
    frames_with = {k for c in dets for _, k, _ in dets[c]}
    out["unannotated_frames"] = total_frames - len(annotated)
    out["unannotated_frames_with_detections"] = len(frames_with - annotated)
    out["detections_on_unannotated_frames"] = len(unann)
    out["detections_on_unannotated_frames_ge_0.5"] = sum(s >= 0.5 for s, _ in unann)

    ok = gate_gap < 0.01
    if args.expect_annotated is not None:
        ok = ok and abs(out["annotated_voc"] - args.expect_annotated) < 0.015
    out["gate_pass"] = bool(ok)
    json.dump(out, open(args.out, "w"), indent=2, sort_keys=True)
    for k in sorted(out):
        print("@@ %-45s %s" % (k, ("%.4f" % out[k]) if isinstance(out[k], float) else out[k]))
    if not ok:
        raise SystemExit("GATE FAILED")


if __name__ == "__main__":
    main()
