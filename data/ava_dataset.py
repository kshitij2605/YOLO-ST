"""AVA v2.2 Dataset for spatiotemporal action detection.

Annotation format (ava_train_v2.2.csv):
    video_id, timestamp_sec, x1, y1, x2, y2, action_label, is_gt

Frames extracted at 25fps from seconds 900-1800:
    frame_idx = (timestamp_sec - 900) * 25 + 1  (1-indexed)

Each keyframe at second T is the LAST frame of a clip of length clip_length.
Multi-label: each bounding box can have multiple action labels. By default this
loader preserves one multi-hot action target per person box.
"""
import csv
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import VideoAugmentor


# DataLoader processes must not each create an OpenCV thread pool.
cv2.setNumThreads(0)

from yolost.geometry import image_hw as _image_hw

# AVA v2.2 action classes (60 used in standard eval, same as YOWOv3)
# These are the 60 classes with >=25 instances in both train and val
# The 60 classes of the official ActivityNet-2019 AVA label map
# (ava_action_list_v2.2_for_activitynet_2019.pbtxt), in label-map order, so
# model index i matches eval_ava.model_index_to_ava_id()[i].
AVA_VALID_CLASSES = [
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 22, 24, 26, 27,
    28, 29, 30, 34, 36, 37, 38, 41, 43, 45, 46, 47, 48, 49, 51, 52, 54, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 77, 78, 79, 80,
]
AVA_NUM_CLASSES = len(AVA_VALID_CLASSES)  # 60
# Map original 1-indexed AVA label -> 0-indexed model class
AVA_LABEL_MAP = {orig: idx for idx, orig in enumerate(AVA_VALID_CLASSES)}


class AVADataset(Dataset):
    """AVA v2.2 dataset.

    Args:
        frames_root: Path to frames/ directory (contains per-video subdirs).
        annot_file: Path to ava_train_v2.2.csv or ava_val_v2.2.csv.
        clip_length: Number of frames per clip (default 64).
        img_size: Spatial resolution.
        split: 'train' or 'val'.
        fps: Frame rate of extracted frames (default 25).
        augment: Apply augmentations during training.
    """

    FPS = 25
    VALID_SEC_RANGE = range(902, 1799)  # annotated seconds

    def __init__(self, frames_root, annot_file, clip_length=64,
                 img_size=224, split='train', augment=True, multi_label=True,
                 max_samples=None, augment_scale_range=(0.5, 1.5),
                 keyframe_position='end', frame_stride=None,
                 supervised_frames=False, uint8_clips=False):
        self.uint8_clips = bool(uint8_clips)
        if keyframe_position not in ('end', 'center'):
            raise ValueError(f'keyframe_position must be end or center, got {keyframe_position!r}')
        self.keyframe_position = keyframe_position
        self.frame_stride = None if frame_stride is None else int(frame_stride)
        self.emit_supervised_frames = bool(supervised_frames)
        self.frames_root = frames_root
        self.clip_length = clip_length
        self.img_size = img_size
        self.split = split
        self.multi_label = multi_label
        self.max_samples = max_samples
        self.augmentor = (
            VideoAugmentor(img_size, scale_range=tuple(augment_scale_range))
            if augment and split == 'train' else None
        )
        self.num_classes = AVA_NUM_CLASSES

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        # Load annotations: keyframe -> list of (x1,y1,x2,y2, label)
        # key: (video_id, timestamp_sec)
        self.annot = defaultdict(list)  # (vid, sec) -> [(x1,y1,x2,y2,label), ...]
        self._load_annotations(annot_file)

        # Build sample list: only keyframes where video frames exist
        self.samples = []  # list of (video_id, timestamp_sec)
        self._build_sample_list()
        if self.max_samples is not None:
            max_samples = int(self.max_samples)
            if max_samples < len(self.samples):
                step = len(self.samples) / max_samples
                self.samples = [self.samples[int(i * step)] for i in range(max_samples)]

        print(f'AVA {split}: {len(self.samples)} keyframes from {len(set(v for v,_ in self.samples))} videos')

    def _load_annotations(self, annot_file):
        with open(annot_file) as f:
            for row in csv.reader(f):
                if not row or len(row) < 7:
                    continue
                vid = row[0]
                sec = int(row[1])
                x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                label = int(row[6])
                # col 7 is person track ID (not is_gt) — no filtering needed

                if label not in AVA_LABEL_MAP:
                    continue
                if sec not in self.VALID_SEC_RANGE:
                    continue

                mapped_label = AVA_LABEL_MAP[label]
                self.annot[(vid, sec)].append((x1, y1, x2, y2, mapped_label))

    def _build_sample_list(self):
        for (vid, sec) in self.annot.keys():
            vid_dir = os.path.join(self.frames_root, vid)
            if not os.path.isdir(vid_dir):
                continue
            # Check keyframe exists
            key_idx = self._sec_to_frame(sec)
            key_path = os.path.join(vid_dir, f'{vid}_{key_idx:06d}.jpg')
            if os.path.exists(key_path):
                self.samples.append((vid, sec))

    def _sec_to_frame(self, sec):
        """Convert AVA second timestamp to 1-indexed frame number."""
        return (sec - 900) * self.FPS + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, sec = self.samples[idx]
        vid_dir = os.path.join(self.frames_root, vid)

        # Keyframe is the LAST frame of the clip
        key_frame = self._sec_to_frame(sec)
        # Max frame available for this video: 22500 (900s * 25fps)
        max_frame = (1799 - 900) * self.FPS + self.FPS  # ~22500

        # Temporal stride augmentation during training
        if self.frame_stride is not None:
            stride = self.frame_stride
        elif self.augmentor is not None:
            stride = random.choice([1, 2, 3])
        else:
            stride = 1
        key_clip_t = (self.clip_length - 1 if self.keyframe_position == 'end'
                      else self.clip_length // 2)

        # Build frame indices: clip ends at keyframe
        frame_indices = []
        for i in range(self.clip_length):
            fi = key_frame + (i - key_clip_t) * stride
            fi = max(1, min(fi, max_frame))
            frame_indices.append(fi)

        # Load frames
        frames = []
        for fi in frame_indices:
            img_path = os.path.join(vid_dir, f'{vid}_{fi:06d}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                # Use black frame as fallback
                frames.append(np.zeros((240, 320, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Build GT: keyframe is at clip_t = clip_length - 1
        gt_boxes = []
        gt_labels = []
        grouped = {}
        for (x1, y1, x2, y2, label) in self.annot[(vid, sec)]:
            if x2 > x1 + 0.001 and y2 > y1 + 0.001:
                if self.multi_label:
                    key = (round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5))
                    if key not in grouped:
                        grouped[key] = [float(key_clip_t), x1, y1, x2, y2, []]
                    grouped[key][5].append(label)
                else:
                    gt_boxes.append([float(key_clip_t), x1, y1, x2, y2])
                    gt_labels.append(label)

        if self.multi_label:
            for _, (frame_t, x1, y1, x2, y2, labels) in grouped.items():
                gt_boxes.append([frame_t, x1, y1, x2, y2])
                multi_hot = np.zeros(self.num_classes, dtype=np.float32)
                for label in labels:
                    multi_hot[label] = 1.0
                gt_labels.append(multi_hot)

        # Augment
        if self.augmentor is not None and gt_boxes:
            frames, gt_boxes, gt_labels = self.augmentor(frames, gt_boxes, gt_labels)
        elif self.augmentor is not None:
            frames, _, _ = self.augmentor(frames, [], [])

        # Resize + normalize
        _h, _w = _image_hw(self.img_size)
        resized = [cv2.resize(f, (_w, _h)) for f in frames]
        clip = np.stack(resized)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
        if self.uint8_clips:
            # Normalised on the GPU by train.normalize_uint8_clips.
            clip = clip.contiguous()
        else:
            clip = clip.float() / 255.0
            clip = (clip - self.mean) / self.std

        targets = {
            'boxes': (torch.tensor(gt_boxes, dtype=torch.float32)
                      if gt_boxes else torch.zeros(0, 5)),
            'labels': (torch.tensor(np.stack(gt_labels), dtype=torch.float32)
                       if self.multi_label and gt_labels else
                       torch.zeros(0, self.num_classes, dtype=torch.float32)
                       if self.multi_label else
                       torch.tensor(gt_labels, dtype=torch.long)
                       if gt_labels else torch.zeros(0, dtype=torch.long)),
        }
        if self.emit_supervised_frames:
            targets['supervised_frames'] = torch.tensor([key_clip_t], dtype=torch.long)
        return clip, targets
