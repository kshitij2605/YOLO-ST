"""Evaluate a multi-checkpoint YOLO-ST Phase3 ensemble on UCF101-24.

This generalizes eval_ucf24_ensemble.py from two members to N members. It is
intended for quick complementarity tests such as 224px + 320px-20ep + 320px-60ep.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ucf101_24 import UCF101_24_Dataset
from eval_ucf24 import _iou
from eval_ucf24_ensemble import decode_frame, load_model, resize_clip_batch, unflip_boxes
from train import collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--member",
        nargs=2,
        action="append",
        metavar=("CONFIG", "CHECKPOINT"),
        required=True,
        help="Add one ensemble member. Repeat for each config/checkpoint pair.",
    )
    parser.add_argument("--conf_thresh", type=float, default=0.005)
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hflip", action="store_true", help="Add horizontal-flip TTA for all members")
    return parser.parse_args()


@torch.no_grad()
def evaluate_multi_ensemble(members, loader, device, conf_thresh, nms_thresh,
                            num_classes, clip_length, hflip=False):
    pred_by_class = {c: [] for c in range(num_classes)}
    gt_by_class = {c: [] for c in range(num_classes)}
    frame_counter = 0

    for clips, targets in loader:
        clips = clips.to(device)
        outputs = []
        for member in members:
            model = member["model"]
            clips_i = resize_clip_batch(clips, member["img_size"])
            out_i = model(clips_i)
            out_i_flip = model(torch.flip(clips_i, dims=[-1])) if hflip else None
            outputs.append((model, out_i, out_i_flip))

        batch = clips.shape[0]
        for b in range(batch):
            gt_boxes = targets["boxes"][b]
            gt_labels = targets["labels"][b]
            valid = gt_boxes.sum(dim=-1) > 0
            gt_boxes = gt_boxes[valid]
            gt_labels = gt_labels[valid]

            for cf in range(clip_length):
                fid = frame_counter + cf
                for i in range(gt_boxes.shape[0]):
                    if int(gt_boxes[i, 0].item()) == cf:
                        c = int(gt_labels[i].item())
                        gt_by_class[c].append((gt_boxes[i, 1:5].cpu().numpy(), fid))

                decoded = []
                for model, out_i, out_i_flip in outputs:
                    pred = decode_frame(model, out_i, b, cf, conf_thresh)
                    if pred is not None:
                        decoded.append(pred)
                    if hflip:
                        pred_flip = decode_frame(model, out_i_flip, b, cf, conf_thresh)
                        if pred_flip is not None:
                            decoded.append((unflip_boxes(pred_flip[0]), pred_flip[1], pred_flip[2]))

                if not decoded:
                    continue

                boxes = torch.cat([d[0] for d in decoded]).clamp(0, 1)
                scores = torch.cat([d[1] for d in decoded])
                classes = torch.cat([d[2] for d in decoded])

                for c in classes.unique():
                    c_mask = classes == c
                    c_boxes = boxes[c_mask]
                    c_scores = scores[c_mask]
                    keep = torchvision.ops.nms(c_boxes, c_scores, nms_thresh)
                    for k in keep:
                        pred_by_class[int(c.item())].append(
                            (c_scores[k].item(), c_boxes[k].cpu().numpy(), fid)
                        )

            frame_counter += clip_length

    per_class_ap = {}
    for c in range(num_classes):
        gts = gt_by_class[c]
        preds = pred_by_class[c]
        n_gt = len(gts)
        if n_gt == 0:
            continue
        if not preds:
            per_class_ap[c] = 0.0
            continue

        preds.sort(key=lambda x: -x[0])
        gt_by_frame = {}
        for gi, (box, fid) in enumerate(gts):
            gt_by_frame.setdefault(fid, []).append((gi, box))

        gt_matched = np.zeros(n_gt, dtype=bool)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for pi, (score, pred_box, pred_fid) in enumerate(preds):
            frame_gts = gt_by_frame.get(pred_fid, [])
            if not frame_gts:
                fp[pi] = 1
                continue
            best_iou = 0.0
            best_gi = -1
            for gi, gt_box in frame_gts:
                if gt_matched[gi]:
                    continue
                iou = _iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                tp[pi] = 1
                gt_matched[best_gi] = True
            else:
                fp[pi] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        px = np.linspace(0, 1, 101)
        per_class_ap[c] = np.trapz(np.interp(px, mrec, mpre), px)

    mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
    return mAP, per_class_ap


def main():
    args = parse_args()
    if len(args.member) < 2:
        raise ValueError("At least two --member CONFIG CHECKPOINT pairs are required")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    members = []
    cfgs = []
    for config_path, checkpoint_path in args.member:
        cfg = yaml.safe_load(open(config_path))
        model = load_model(cfg, checkpoint_path, device)
        cfgs.append(cfg)
        members.append({
            "config": config_path,
            "checkpoint": checkpoint_path,
            "cfg": cfg,
            "model": model,
            "img_size": cfg["data"]["img_size"],
        })

    eval_size = max(member["img_size"] for member in members)
    num_workers = min(cfg["data"].get("num_workers", 4) for cfg in cfgs)
    base_cfg = cfgs[0]
    dataset = UCF101_24_Dataset(
        root=base_cfg["data"]["root"],
        annot_file=base_cfg["data"]["annot_file"],
        clip_length=base_cfg["data"]["clip_length"],
        stride=base_cfg["data"]["stride"],
        split="test",
        img_size=eval_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    print("Members:")
    for member in members:
        print(f"  {member['config']} :: {member['checkpoint']}")
    print(
        f"Evaluating {len(members)}-member ensemble on {len(dataset)} clips "
        f"(conf={args.conf_thresh}, nms={args.nms_thresh}, hflip={args.hflip})"
    )
    t0 = time.time()
    mAP, per_class = evaluate_multi_ensemble(
        members,
        loader,
        device,
        args.conf_thresh,
        args.nms_thresh,
        base_cfg["model"]["num_classes"],
        base_cfg["data"]["clip_length"],
        hflip=args.hflip,
    )
    print(f"Done in {time.time() - t0:.0f}s")
    print(f"\nFrame-mAP@0.5: {mAP * 100:.2f}%")
    names = UCF101_24_Dataset.CLASSES
    for cls, ap in sorted(per_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  {names[cls]:25s}: {ap * 100:.1f}%")


if __name__ == "__main__":
    main()
