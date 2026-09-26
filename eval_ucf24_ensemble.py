"""Evaluate a two-checkpoint YOLO-ST ensemble on UCF101-24 frame-mAP@0.5.

The first intended use is Phase3A 224px + Phase3A 320px. The script loads each
model with its own config, resizes the same test clips to each model's input
size, merges per-frame detections, and applies class-wise NMS.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ucf101_24 import UCF101_24_Dataset
from eval_ucf24 import _iou
from train import collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_a", required=True)
    p.add_argument("--checkpoint_a", required=True)
    p.add_argument("--config_b", required=True)
    p.add_argument("--checkpoint_b", required=True)
    p.add_argument("--conf_thresh", type=float, default=0.005)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--hflip", action="store_true", help="Add horizontal-flip TTA for both models")
    return p.parse_args()


def load_model(cfg, checkpoint, device):
    model_type = cfg["model"].get("type", "uniform")
    if model_type == "phase3":
        from yolost.model_phase3 import YOLOST_Phase3
        model = YOLOST_Phase3(
            num_classes=cfg["model"]["num_classes"],
            img_size=cfg["data"]["img_size"],
            target_T=cfg["model"].get("target_T", [64, 32, 16]),
            clip_length=cfg["data"]["clip_length"],
        ).to(device)
    elif model_type == "phase3_temporal_context":
        from yolost.model_phase3_temporal_context import YOLOST_Phase3TemporalContext
        model = YOLOST_Phase3TemporalContext(
            num_classes=cfg["model"]["num_classes"],
            img_size=cfg["data"]["img_size"],
            target_T=cfg["model"].get("target_T", [64, 32, 16]),
            clip_length=cfg["data"]["clip_length"],
            context_dim=cfg["model"].get("context_dim", 128),
            context_heads=cfg["model"].get("context_heads", 4),
            context_depth=cfg["model"].get("context_depth", 1),
            context_dropout=cfg["model"].get("context_dropout", 0.0),
        ).to(device)
    elif model_type == "videomae":
        from yolost.model_videomae import YOLOST_VideoMAE
        model = YOLOST_VideoMAE(
            num_classes=cfg["model"]["num_classes"],
            img_size=cfg["data"]["img_size"],
            clip_length=cfg["data"]["clip_length"],
            model_id=cfg["model"].get("model_id", "MCG-NJU/videomae-base-finetuned-kinetics"),
            freeze_backbone=cfg["model"].get("freeze_backbone", True),
            unfreeze_last_n_blocks=cfg["model"].get("unfreeze_last_n_blocks", 0),
            backbone_frames=cfg["model"].get("backbone_frames", 16),
            dtype=cfg["model"].get("dtype", "float16"),
            temporal_adapter=cfg["model"].get("temporal_adapter", True),
            temporal_adapter_depth=cfg["model"].get("temporal_adapter_depth", 2),
            temporal_adapter_kernel=cfg["model"].get("temporal_adapter_kernel", 5),
            temporal_adapter_expansion=cfg["model"].get("temporal_adapter_expansion", 2),
            temporal_adapter_dropout=cfg["model"].get("temporal_adapter_dropout", 0.0),
            spatial_query_adapter=cfg["model"].get("spatial_query_adapter", False),
            spatial_query_grid=cfg["model"].get("spatial_query_grid", 7),
            spatial_query_heads=cfg["model"].get("spatial_query_heads", 8),
            spatial_query_dropout=cfg["model"].get("spatial_query_dropout", 0.1),
        ).to(device)
    elif model_type == "mvit_bmvit":
        from yolost.model_mvit_bmvit import YOLOST_MViTBMViT
        model = YOLOST_MViTBMViT(
            num_classes=cfg["model"]["num_classes"],
            img_size=cfg["data"]["img_size"],
            clip_length=cfg["data"]["clip_length"],
            backbone_frames=cfg["model"].get("backbone_frames", 16),
            freeze_backbone=cfg["model"].get("freeze_backbone", True),
            unfreeze_last_n_blocks=cfg["model"].get("unfreeze_last_n_blocks", 0),
            pretrained=cfg["model"].get("pretrained", True),
            stop_before_final_pool=cfg["model"].get("stop_before_final_pool", True),
            hidden_dim=cfg["model"].get("hidden_dim", 384),
            head_depth=cfg["model"].get("head_depth", 3),
            action_temporal_pool=cfg["model"].get("action_temporal_pool", True),
        ).to(device)
    else:
        raise ValueError(f"Unsupported ensemble model type: {model_type}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {checkpoint} epoch {ckpt['epoch']} img_size={cfg['data']['img_size']}")
    return model


def resize_clip_batch(clips, img_size):
    """Resize normalized clips to img_size in tensor space.

    Args:
        clips: (B, 3, T, H, W)
    """
    if clips.shape[-1] == img_size and clips.shape[-2] == img_size:
        return clips
    b, c, t, h, w = clips.shape
    frames = clips.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    frames = torch.nn.functional.interpolate(
        frames,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return frames.reshape(b, t, c, img_size, img_size).permute(0, 2, 1, 3, 4).contiguous()


def decode_frame(model, outputs, b, cf, conf_thresh):
    all_boxes = []
    all_scores = []
    all_classes = []

    for si, scale_out in enumerate(outputs):
        cls_pred, reg_pred, obj_pred = scale_out[0], scale_out[1], scale_out[2]
        t_stride = model.temporal_strides[si]
        s_stride = model.spatial_strides[si]
        t_det = cf // t_stride
        if t_det >= cls_pred.shape[2]:
            continue

        step = s_stride / model.img_size
        obj_sig = torch.sigmoid(obj_pred[b, 0, t_det])
        cls_sig = torch.sigmoid(cls_pred[b, :, t_det])
        reg = reg_pred[b, :, t_det]

        combined = obj_sig.unsqueeze(0) * cls_sig
        max_score, max_cls = combined.max(dim=0)
        mask = max_score > conf_thresh
        if not mask.any():
            continue

        h_idx, w_idx = torch.where(mask)
        scores = max_score[mask]
        classes = max_cls[mask]
        box_raw = reg[:, h_idx, w_idx].T
        cx = (w_idx.float() + torch.sigmoid(box_raw[:, 0])) * step
        cy = (h_idx.float() + torch.sigmoid(box_raw[:, 1])) * step
        bw = torch.exp(box_raw[:, 2].clamp(max=5.0)) * step
        bh = torch.exp(box_raw[:, 3].clamp(max=5.0)) * step
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)

    if not all_boxes:
        return None
    return torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_classes)


def unflip_boxes(boxes):
    flipped = boxes.clone()
    flipped[:, 0] = 1.0 - boxes[:, 2]
    flipped[:, 2] = 1.0 - boxes[:, 0]
    return flipped


@torch.no_grad()
def evaluate_ensemble(model_a, model_b, loader, device, conf_thresh, nms_thresh,
                      num_classes, clip_length, img_size_a, img_size_b, hflip=False):
    pred_by_class = {c: [] for c in range(num_classes)}
    gt_by_class = {c: [] for c in range(num_classes)}
    frame_counter = 0

    for clips, targets in loader:
        clips = clips.to(device)
        clips_a = resize_clip_batch(clips, img_size_a)
        clips_b = resize_clip_batch(clips, img_size_b)
        out_a = model_a(clips_a)
        out_b = model_b(clips_b)
        if hflip:
            out_a_flip = model_a(torch.flip(clips_a, dims=[-1]))
            out_b_flip = model_b(torch.flip(clips_b, dims=[-1]))
        else:
            out_a_flip = None
            out_b_flip = None
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
                da = decode_frame(model_a, out_a, b, cf, conf_thresh)
                db = decode_frame(model_b, out_b, b, cf, conf_thresh)
                if da is not None:
                    decoded.append(da)
                if db is not None:
                    decoded.append(db)
                if hflip:
                    da_flip = decode_frame(model_a, out_a_flip, b, cf, conf_thresh)
                    db_flip = decode_frame(model_b, out_b_flip, b, cf, conf_thresh)
                    if da_flip is not None:
                        decoded.append((unflip_boxes(da_flip[0]), da_flip[1], da_flip[2]))
                    if db_flip is not None:
                        decoded.append((unflip_boxes(db_flip[0]), db_flip[1], db_flip[2]))
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
    cfg_a = yaml.safe_load(open(args.config_a))
    cfg_b = yaml.safe_load(open(args.config_b))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_a = load_model(cfg_a, args.checkpoint_a, device)
    model_b = load_model(cfg_b, args.checkpoint_b, device)

    eval_size = max(cfg_a["data"]["img_size"], cfg_b["data"]["img_size"])
    dataset = UCF101_24_Dataset(
        root=cfg_a["data"]["root"],
        annot_file=cfg_a["data"]["annot_file"],
        clip_length=cfg_a["data"]["clip_length"],
        stride=cfg_a["data"]["stride"],
        split="test",
        img_size=eval_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(cfg_a["data"].get("num_workers", 4), cfg_b["data"].get("num_workers", 4)),
        collate_fn=collate_fn,
    )

    print(
        f"Evaluating ensemble on {len(dataset)} clips "
        f"(conf={args.conf_thresh}, nms={args.nms_thresh}, hflip={args.hflip})"
    )
    t0 = time.time()
    mAP, per_class = evaluate_ensemble(
        model_a,
        model_b,
        loader,
        device,
        args.conf_thresh,
        args.nms_thresh,
        cfg_a["model"]["num_classes"],
        cfg_a["data"]["clip_length"],
        cfg_a["data"]["img_size"],
        cfg_b["data"]["img_size"],
        hflip=args.hflip,
    )
    print(f"Done in {time.time() - t0:.0f}s")
    print(f"\nFrame-mAP@0.5: {mAP * 100:.2f}%")
    names = UCF101_24_Dataset.CLASSES
    for cls, ap in sorted(per_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  {names[cls]:25s}: {ap * 100:.1f}%")


if __name__ == "__main__":
    main()
