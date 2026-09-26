"""Score the headline checkpoint under three metrics from one inference pass.

YOWOFormer-L reports 93.35 on UCF101-24, but its released code computes frame mAP
with the Ultralytics/YOLO detection-metrics routine, not the community evaluator:

  1. AP integral    np.trapz over a maximum.accumulate-monotonised PR curve
                    (evaluate.py compute_ap) vs our voc_ap every-point integral.
  2. IoU            torchvision-style exclusive, vs our inclusive_iou (+1.0).
  3. Coordinates    a 224x224 anisotropically-resized space, vs native 240x320.

Their testlist.txt holds one line per annotated keyframe. Built from our GT that is
exactly 138,527 lines, matching our corrected frame set (all tube rows), NOT the
yowo set (drop_tube_terminal=True). Their number therefore pairs with our 93.21.

VALIDATION GATE: this must reproduce the recorded 93.21 corrected and 93.32 yowo
from the same detections, or the YOLO-style number is not reported.

Not a bit-exact replay of their inference: our detector and per-class NMS are kept.
It isolates the metric, not the pipeline.

The reported 95.10 comes from research_new/score_from_dump.py, which scores the
recorded detections and needs no inference; tools/score_frame_conventions.py
reproduces it together with every other frame-mAP convention.
"""

import argparse
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import torch

from config_utils import load_config
from data.ucf101_24 import build_clip_starts
from yolost.clip_weighting import clip_frame_weights
from eval_video_map import load_model
from eval_tube_queries import decode_dense_frame, load_clip
from frame_map_protocol import (
    add_frame_predictions,
    compute_frame_map,
    load_frame_ground_truth,
    resolve_frame_candidates,
)
from eval_dense_frame import dense_outputs

EPS = 1e-16


def exclusive_iou_matrix(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    return inter / (area_gt[:, None] + area_pred[None, :] - inter + EPS)


def yolo_match(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr=0.5):
    correct = np.zeros((len(pred_boxes), 1), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return correct
    iou = exclusive_iou_matrix(gt_boxes, pred_boxes)
    same = np.asarray(gt_cls)[:, None] == np.asarray(pred_cls)[None, :]
    x = np.where((iou >= iou_thr) & same)
    if x[0].shape[0]:
        matches = np.stack(x, 1).astype(np.float64)
        matches = np.concatenate([matches, iou[x[0], x[1]][:, None]], 1)
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        for m in matches:
            correct[int(m[1]), 0] = True
    return correct


def yolo_compute_ap(tp, conf, pred_cls, target_cls):
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((unique_classes.shape[0], tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        sel = pred_cls == c
        nl = nt[ci]
        if sel.sum() == 0 or nl == 0:
            continue
        fpc = (1 - tp[sel]).cumsum(0)
        tpc = tp[sel].cumsum(0)
        recall = tpc / (nl + EPS)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            ap[ci, j] = np.trapz(m_pre, m_rec)
    return float(ap[:, 0].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conf-thresh", type=float, default=0.005)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--clip-weighting", default="hann_peak_norm")
    parser.add_argument("--clip-overlap", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    # load_model does not switch modes; without eval() BatchNorm and dropout run
    # in training mode, which is what produced the 82.58 of 2026-09-18.
    model.eval()
    if hasattr(model, "enable_tube_query_output"):
        model.enable_tube_query_output(False)

    with open(cfg["data"]["annot_file"], "rb") as handle:
        annot = pickle.load(handle, encoding="latin1")
    videos = annot["test_videos"][int(cfg["data"].get("split_index", 0))]
    if args.max_videos:
        videos = videos[:args.max_videos]

    corrected_gt, corrected_frames = load_frame_ground_truth(annot, videos)
    yowo_gt, yowo_frames = load_frame_ground_truth(
        annot, videos, drop_tube_terminal=True
    )

    clip_length = cfg["data"]["clip_length"]
    predictions = defaultdict(list)
    per_frame = defaultdict(list)
    started = time.time()

    for index, video_name in enumerate(videos):
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        num_frames = annot["nframes"][video_name]
        resolution = annot["resolution"].get(video_name, (240, 320))
        candidates = defaultdict(list)
        clip_starts = build_clip_starts(
            num_frames, clip_length, overlap=args.clip_overlap
        )
        weights = clip_frame_weights(
            clip_starts, clip_length, num_frames, args.clip_weighting
        )
        for start in clip_starts:
            clip = load_clip(
                cfg["data"]["root"], video_name, start, num_frames,
                clip_length, cfg["data"]["img_size"], resolution,
            ).to(device)
            with torch.no_grad():
                dense = dense_outputs(model(clip))
            for local in range(clip_length):
                gframe = min(start + local, num_frames) - 1
                if gframe not in corrected_frames[video_name]:
                    continue
                for det in decode_dense_frame(
                    dense, model, local, args.conf_thresh, args.nms_thresh
                ):
                    det["score"] *= float(weights[start][local])
                    candidates[gframe].append(det)

        for frame_id in corrected_frames[video_name]:
            resolved = resolve_frame_candidates(
                candidates.get(frame_id, []), args.nms_thresh
            )
            add_frame_predictions(
                predictions, video_name, frame_id, resolved, resolution
            )
            for det in resolved:
                per_frame[(video_name, frame_id)].append(
                    (np.asarray(det["box"], dtype=np.float64),
                     float(det["score"]), int(det["class"]))
                )
        if (index + 1) % 100 == 0:
            print("  %d/%d videos (%.0fs)"
                  % (index + 1, len(videos), time.time() - started), flush=True)

    num_classes = cfg["model"]["num_classes"]
    corrected_eval = {(v, f) for v, fs in corrected_frames.items() for f in fs}
    yowo_eval = {(v, f) for v, fs in yowo_frames.items() for f in fs}

    corrected_map, _ = compute_frame_map(
        predictions, corrected_gt, num_classes=num_classes,
        evaluated_frames=corrected_eval,
    )
    yowo_map, _ = compute_frame_map(
        predictions, yowo_gt, num_classes=num_classes,
        evaluated_frames=yowo_eval,
    )

    scale = float(args.img_size)
    gt_by_frame = defaultdict(list)
    for cid, entries in corrected_gt.items():
        for frame_key, box in entries:
            height, width = annot["resolution"].get(frame_key[0], (240, 320))
            gt_by_frame[frame_key].append((
                np.array([box[0] / width * scale, box[1] / height * scale,
                          box[2] / width * scale, box[3] / height * scale]),
                cid,
            ))

    tp_all, conf_all, pcls_all, tcls_all = [], [], [], []
    for frame_key in corrected_eval:
        gts = gt_by_frame.get(frame_key, [])
        preds = per_frame.get(frame_key, [])
        gt_boxes = [g[0] for g in gts]
        gt_cls = [g[1] for g in gts]
        if gts:
            tcls_all.extend(gt_cls)
        if not preds:
            continue
        pred_boxes = [p[0] * scale for p in preds]
        correct = yolo_match(
            pred_boxes, [p[2] for p in preds], gt_boxes, gt_cls
        )
        tp_all.append(correct)
        conf_all.extend([p[1] for p in preds])
        pcls_all.extend([p[2] for p in preds])

    yolo_map = float("nan")
    if tp_all:
        yolo_map = yolo_compute_ap(
            np.concatenate(tp_all, 0),
            np.asarray(conf_all, dtype=np.float64),
            np.asarray(pcls_all),
            np.asarray(tcls_all),
        )

    result = {
        "checkpoint": args.checkpoint,
        "clip_weighting": args.clip_weighting,
        "videos": len(videos),
        "corrected_frames": len(corrected_eval),
        "voc_corrected_frame_map": 100.0 * corrected_map,
        "voc_yowo_frame_map": 100.0 * yowo_map,
        "yowoformer_style_frame_map": 100.0 * yolo_map,
        "elapsed_s": time.time() - started,
    }
    print("")
    print("=== VALIDATION GATE: these must match the recorded values ===")
    print("  VOC corrected : %.2f  (recorded 93.21)"
          % result["voc_corrected_frame_map"])
    print("  VOC yowo      : %.2f  (recorded 93.32)"
          % result["voc_yowo_frame_map"])
    print("=== our model under the YOWOFormer evaluator ===")
    print("  YOWOFormer-style : %.2f  (they report 93.35)"
          % result["yowoformer_style_frame_map"])
    print("  frames scored    : %d (their testlist: 138527)"
          % result["corrected_frames"])
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print("SCORING_DONE")


if __name__ == "__main__":
    main()
