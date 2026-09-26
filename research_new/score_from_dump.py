"""Score the RECORDED detections under both metrics. No re-inference.

The dense detections that produced the reported 93.21 / 93.32 are already on disk:
V2-WS2.0_RESULT/interim_test_frozen_protocol/frames_ucf_test_ema.pkl, written by
eval_tube_queries.py --frame_dump during the interim test. Rows are
(video, frame_1based, class, score, x1, y1, x2, y2) in pixel units, post-NMS and
already clip-weighted.

A first attempt re-ran inference with research_new/score_yowoformer_style.py and
scored 82.58 instead of 93.21. The cause (found 2026-09-24) was that the script
never called model.eval(), so BatchNorm and dropout ran in training mode; an
earlier note blamed the memory/tube-output flags, which is wrong.
Scoring the recorded rows removes that entire class of error: the VOC gate must
reproduce by construction, and the YOWOFormer-style figure then comes from
identical predictions, which is exactly the claim the appendix makes.
"""
import io
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from frame_map_protocol import compute_frame_map, load_frame_ground_truth

EPS = 1e-16
DUMP = ("research_new/experiments/V2-WS2.0_RESULT/"
        "interim_test_frozen_protocol/frames_ucf_test_ema.pkl")


def exclusive_iou(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)))
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    a_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    a_pr = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    return inter / (a_gt[:, None] + a_pr[None, :] - inter + EPS)


def yolo_match(pb, pc, gb, gc, thr=0.5):
    correct = np.zeros((len(pb), 1), dtype=bool)
    if not len(gb) or not len(pb):
        return correct
    iou = exclusive_iou(gb, pb)
    same = np.asarray(gc)[:, None] == np.asarray(pc)[None, :]
    x = np.where((iou >= thr) & same)
    if x[0].shape[0]:
        m = np.stack(x, 1).astype(np.float64)
        m = np.concatenate([m, iou[x[0], x[1]][:, None]], 1)
        if x[0].shape[0] > 1:
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        for r in m:
            correct[int(r[1]), 0] = True
    return correct


def yolo_ap(tp, conf, pcls, tcls):
    o = np.argsort(-conf)
    tp, conf, pcls = tp[o], conf[o], pcls[o]
    uc, nt = np.unique(tcls, return_counts=True)
    ap = np.zeros((uc.shape[0], tp.shape[1]))
    for ci, c in enumerate(uc):
        sel = pcls == c
        nl = nt[ci]
        if sel.sum() == 0 or nl == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        rec = tpc / (nl + EPS)
        pre = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            mr = np.concatenate(([0.0], rec[:, j], [1.0]))
            mp = np.concatenate(([1.0], pre[:, j], [0.0]))
            mp = np.flip(np.maximum.accumulate(np.flip(mp)))
            ap[ci, j] = np.trapz(mp, mr)
    return float(ap[:, 0].mean())


def main():
    d = pickle.load(open(DUMP, "rb"))
    print("dump: weighting=%s overlap=%s conf=%s nms=%s units=%s base=%s"
          % (d["clip_weighting"], d["clip_overlap"], d["frame_conf_thresh"],
             d["frame_nms_thresh"], d["box_units"], d["frame_index_base"]))
    rows = d["rows"]

    annot = pickle.load(open(d["annot_file"], "rb"), encoding="latin1")
    videos = annot["test_videos"][0]
    corrected_gt, corrected_frames = load_frame_ground_truth(annot, videos)
    yowo_gt, yowo_frames = load_frame_ground_truth(
        annot, videos, drop_tube_terminal=True)

    base = int(d["frame_index_base"])
    predictions = defaultdict(list)
    per_frame = defaultdict(list)
    for v, f, c, s, x1, y1, x2, y2 in rows:
        fid = int(f) - base
        key = (v, fid)
        box = np.array([x1, y1, x2, y2], dtype=np.float64)
        predictions[int(c)].append((float(s), key, box))
        per_frame[key].append((box, float(s), int(c)))

    nc = 24
    c_eval = {(v, f) for v, fs in corrected_frames.items() for f in fs}
    y_eval = {(v, f) for v, fs in yowo_frames.items() for f in fs}
    c_map, _ = compute_frame_map(predictions, corrected_gt, num_classes=nc,
                                 evaluated_frames=c_eval)
    y_map, _ = compute_frame_map(predictions, yowo_gt, num_classes=nc,
                                 evaluated_frames=y_eval)

    S = 224.0
    gt_by_frame = defaultdict(list)
    for cid, entries in corrected_gt.items():
        for key, box in entries:
            h, w = annot["resolution"].get(key[0], (240, 320))
            gt_by_frame[key].append(
                (np.array([box[0] / w * S, box[1] / h * S,
                           box[2] / w * S, box[3] / h * S]), cid))

    tp, cf, pc, tc = [], [], [], []
    for key in c_eval:
        g = gt_by_frame.get(key, [])
        p = per_frame.get(key, [])
        if g:
            tc.extend([e[1] for e in g])
        if not p:
            continue
        h, w = annot["resolution"].get(key[0], (240, 320))
        pb = [np.array([b[0] / w * S, b[1] / h * S,
                        b[2] / w * S, b[3] / h * S]) for b, _, _ in p]
        tp.append(yolo_match(pb, [e[2] for e in p],
                             [e[0] for e in g], [e[1] for e in g]))
        cf.extend([e[1] for e in p])
        pc.extend([e[2] for e in p])

    y = yolo_ap(np.concatenate(tp, 0), np.asarray(cf), np.asarray(pc),
                np.asarray(tc)) if tp else float("nan")

    res = {"voc_corrected": 100.0 * c_map, "voc_yowo": 100.0 * y_map,
           "yowoformer_style": 100.0 * y, "frames": len(c_eval),
           "rows": len(rows)}
    print("")
    print("=== VALIDATION GATE ===")
    print("  VOC corrected : %.2f   (recorded 93.21)" % res["voc_corrected"])
    print("  VOC yowo      : %.2f   (recorded 93.32)" % res["voc_yowo"])
    ok = abs(res["voc_corrected"] - 93.21) < 0.05 and abs(res["voc_yowo"] - 93.32) < 0.05
    print("  GATE: %s" % ("PASS" if ok else "FAIL"))
    print("=== same detections, YOWOFormer evaluator ===")
    print("  YOWOFormer-style : %.2f   (they report 93.35)" % res["yowoformer_style"])
    print("  frames %d  rows %d" % (res["frames"], res["rows"]))
    json.dump(res, io.open("research_new/experiments/yowoformer_style_from_dump.json",
                           "w", encoding="utf-8"), indent=2, sort_keys=True)
    print("SCORING_DONE")


if __name__ == "__main__":
    main()
