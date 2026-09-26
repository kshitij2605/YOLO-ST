"""ROAD-exact frame mAP of any frame dump: every frame, per-frame matching in score order to the max-IoU
remaining GT, exclusive IoU (ROAD utils/evaluation.py compute_iou), VOC every-point AP (ROAD voc_ap); also the
inclusive-IoU variant used for the paper's 89.54. Usage: python3 tools/road_exact_frame_map.py DUMP [OUT_JSON]"""
import json
import os, pickle, sys
from collections import defaultdict
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
sys.path.insert(0, R)
from frame_map_protocol import load_frame_ground_truth
d = pickle.load(open(sys.argv[1], "rb"))
af = d["annot_file"]; af = af if af.startswith("/") else R + af
a = pickle.load(open(af, "rb"), encoding="latin1")
vids = list(a["test_videos"][0]); res = a["resolution"]
gt, frames = load_frame_ground_truth(a, vids)
base = int(d["frame_index_base"])
dets = defaultdict(lambda: defaultdict(list))   # cls -> frame -> [(score, box)]
for v, f, c, s, x1, y1, x2, y2 in d["rows"]:
    dets[int(c)][(v, int(f) - base)].append((float(s), np.array([x1, y1, x2, y2], dtype=np.float64)))
def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.])); mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
def iou(g, b, plus):
    xmin = np.maximum(g[:, 0], b[0]); ymin = np.maximum(g[:, 1], b[1])
    xmax = np.minimum(g[:, 2], b[2]); ymax = np.minimum(g[:, 3], b[3])
    iw = np.maximum(xmax - xmin + plus, 0.); ih = np.maximum(ymax - ymin + plus, 0.)
    inter = iw * ih
    ag = (g[:, 2] - g[:, 0] + plus) * (g[:, 3] - g[:, 1] + plus); ab = (b[2] - b[0] + plus) * (b[3] - b[1] + plus)
    return inter / (ag + ab - inter)
RESULT = {}
for name, plus, rint in (("ROAD exact (exclusive IoU, raw float boxes)", 0.0, False),
                         ("exclusive IoU, rint+clip boxes", 0.0, True),
                         ("inclusive IoU, rint+clip boxes (paper 89.54)", 1.0, True)):
    aps = []
    for cls in range(24):
        g = defaultdict(list)
        for key, box in gt.get(cls, []):
            g[key].append(np.asarray(box, dtype=np.float64))
        g = {k: np.stack(x) for k, x in g.items()}
        npos = sum(x.shape[0] for x in g.values())
        scores, istp = [], []
        for key, lst in dets[cls].items():
            h, w = res.get(key[0], (240, 320))
            remaining = g.get(key)
            for s, b in sorted(lst, key=lambda t: -t[0]):
                if rint:
                    b = np.rint(b); b[[0, 2]] = np.clip(b[[0, 2]], 0, w - 1); b[[1, 3]] = np.clip(b[[1, 3]], 0, h - 1)
                ok = False
                if remaining is not None and remaining.shape[0] > 0:
                    ious = iou(remaining, b, plus); m = int(np.argmax(ious))
                    if ious[m] >= 0.5:
                        ok = True; remaining = np.delete(remaining, m, 0)
                scores.append(s); istp.append(ok)
        order = np.argsort(-np.asarray(scores)); tp = np.asarray(istp)[order]
        ctp = np.cumsum(tp).astype(float); cfp = np.cumsum(~tp).astype(float)
        aps.append(voc_ap(ctp / max(npos, 1), ctp / np.maximum(ctp + cfp, np.finfo(float).eps)))
    RESULT[name] = 100 * float(np.mean(aps))
    print("@@ every frame, VOC every-point, %-45s %.2f" % (name, RESULT[name]))
if len(sys.argv) > 2:
    json.dump(RESULT, open(sys.argv[2], "w"), indent=2, sort_keys=True)
