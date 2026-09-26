"""Port of ROAD's MATLAB frame-mAP script (gurkirt/realtime-action-detection, online-tubes/frameAp.m, with
actionpath/nms.m and eval/xVOCap.m) applied to one frame dump. ROAD's README recommends this script, not the
Python one, for published frame mAP.

Per video frame and class (dofilter): keep scores > 0.01, the top 50 by score, NMS at 0.45 (IoU with exclusive
areas, suppress if > 0.45), then at most 20. Detection boxes are shifted to 1-based pixels (+1) and clipped to
320x240. Matching per frame follows the script: detections in class order, each against the uncovered ground
truth boxes of the frame (when any has the detection's class), IoU with the detection widened by one pixel
(rectint on [x y w h] rectangles); VOC every-point AP (xVOCap), mean over the 24 classes.

ROAD stores ground truth as 1-based [x y w h] in its own annots.mat; this repository's pickle stores 0-based
x1,y1,x2,y2 with inclusive right edges (max x2 = 319 and max y2 = 239 on 320x240 frames), so the mapping is
  reported:          rect = [x1+1, y1+1, x2-x1+1, y2-y1+1]   (the value in the paper, e.g. 88.47)
  exclusive_widths:  rect = [x1+1, y1+1, x2-x1, y2-y1]       (if the stored widths were exclusive; reference)
With --no-filter the script's detection filter is skipped, a control that should match the inclusive-extent
Python scorer (89.55 against 89.54 for the reported checkpoint).
Usage: python3 tools/road_matlab_frame_map.py DUMP [OUT_JSON] [--no-filter]
"""
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, R)
from frame_map_protocol import load_frame_ground_truth  # noqa: E402

d = pickle.load(open(sys.argv[1], "rb"))
af = d["annot_file"]
af = af if af.startswith("/") else R + af
annot = pickle.load(open(af, "rb"), encoding="latin1")
vids = list(annot["test_videos"][0])
gt, _ = load_frame_ground_truth(annot, vids)
base = int(d["frame_index_base"])

gt_frame = defaultdict(list)  # (video, frame) -> [(cls, x1, y1, x2, y2)]
for cls, items in gt.items():
    for key, box in items:
        gt_frame[key].append((int(cls),) + tuple(float(v) for v in box))
dets = defaultdict(lambda: defaultdict(list))  # (video, frame) -> cls -> [(score, box)]
for v, f, c, s, x1, y1, x2, y2 in d["rows"]:
    dets[(v, int(f) - base)][int(c)].append((float(s), np.array([x1, y1, x2, y2], dtype=np.float64)))


def nms(boxes, scores, overlap):
    order = list(np.argsort(scores))  # ascending, as in nms.m
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    pick = []
    while order:
        i = order[-1]
        pick.append(i)
        keep = []
        for j in order[:-1]:
            w = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
            h = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
            if w > 0 and h > 0:
                inter = w * h
                if inter / (area[j] + area[i] - inter) > overlap:
                    continue
            keep.append(j)
        order = keep
    return pick


def dofilter(items):
    items = [(s, b) for s, b in items if s > 0.01]
    items.sort(key=lambda t: -t[0])
    items = items[:50]
    if not items:
        return []
    boxes = np.stack([b for _, b in items]) + 1.0  # read_detections: +1, then clip to the 320x240 frame
    boxes[:, 0] = np.maximum(boxes[:, 0], 1)
    boxes[:, 1] = np.maximum(boxes[:, 1], 1)
    boxes[:, 2] = np.minimum(boxes[:, 2], 320)
    boxes[:, 3] = np.minimum(boxes[:, 3], 240)
    scores = np.array([s for s, _ in items])
    pick = nms(boxes, scores, 0.45)[:20]
    return [(scores[p], boxes[p]) for p in pick]


def rectint(a, b):  # MATLAB rectint on [x y w h]
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def evaluate(gt_plus):
    scores = defaultdict(list)  # cls -> [(score, tp)]
    npos = defaultdict(int)
    keys = set(gt_frame) | set(dets)
    for key in keys:
        gts = gt_frame.get(key, [])
        for g in gts:
            npos[g[0]] += 1
        gt_rects = [(g[0], (g[1] + 1.0, g[2] + 1.0, g[3] - g[1] + gt_plus, g[4] - g[2] + gt_plus)) for g in gts]
        covered = [False] * len(gt_rects)
        labels = {g[0] for g in gts}
        for cls in sorted(dets.get(key, {})):
            for s, b in dofilter(dets[key][cls]):
                dt = (b[0], b[1], b[2] - b[0] + 1, b[3] - b[1] + 1)
                best, best_g = -np.inf, -1
                if cls in labels:
                    for gi, (_, rect) in enumerate(gt_rects):
                        if covered[gi]:
                            continue
                        inter = rectint(rect, dt)
                        iou = inter / (rect[2] * rect[3] + dt[2] * dt[3] - inter)
                        if iou > best:
                            best, best_g = iou, gi
                tp = best >= 0.5
                if tp:
                    covered[best_g] = True
                scores[cls].append((s, tp))
    aps = []
    for cls in range(24):
        items = sorted(scores.get(cls, []), key=lambda t: -t[0])
        tp = np.array([t for _, t in items], dtype=bool)
        ctp, cfp = np.cumsum(tp).astype(float), np.cumsum(~tp).astype(float)
        if npos.get(cls, 0) == 0 or len(items) == 0:
            aps.append(0.0)
            continue
        rec, prec = ctp / npos[cls], ctp / np.maximum(ctp + cfp, np.finfo(float).eps)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        aps.append(float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])))
    return 100 * float(np.mean(aps))


if "--no-filter" in sys.argv:
    def dofilter(items):  # noqa: F811  (control: shift and clip as the script does, no floor, NMS or cap)
        out = []
        for s, b in sorted(items, key=lambda t: -t[0]):
            b = b + 1.0
            b[0], b[1], b[2], b[3] = max(b[0], 1), max(b[1], 1), min(b[2], 320), min(b[3], 240)
            out.append((s, b))
        return out
    sys.argv.remove("--no-filter")
result = {"reported": evaluate(1.0), "exclusive_widths": evaluate(0.0)}
for name, value in result.items():
    print("@@ ROAD MATLAB frameAp.m port, %s: %.2f" % (name, value))
if len(sys.argv) > 2:
    json.dump(result, open(sys.argv[2], "w"), indent=2, sort_keys=True)
