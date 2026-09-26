"""AVA-Kinetics training clips in the AVA v2.2 label space.

Frames come from tools/stream_ava_kinetics.py:
    <frames_root>/<clip>/00001.jpg ...   (30 fps, short side 256, a window around
                                          every annotated keyframe)
    <manifest_dir>/part_NNN.jsonl        one record per clip: frames, fps and
                                          {"<timestamp:.6f>": 1-indexed frame}
Annotations are kinetics_{train,val}_v1.0.csv rows
    youtube_id, timestamp, x1, y1, x2, y2, action_id[, person_id]
where rows with only two columns mark keyframes without actions.

Samples, clip construction, augmentation, multi-hot labels, keyframe-only
supervision and uint8 transport follow data/ava_dataset.AVADataset exactly, and
labels map through the same official 60-class AVA_LABEL_MAP, so the dataset can
be concatenated with AVA for training.
"""

import csv
import glob
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import VideoAugmentor
from .ava_dataset import AVA_LABEL_MAP, AVA_NUM_CLASSES

cv2.setNumThreads(0)

from yolost.geometry import image_hw as _image_hw  # noqa: E402


def _timestamp_key(value):
    return '%.6f' % float(value)


class AVAKineticsDataset(Dataset):

    def __init__(self, frames_root, manifest_dir, annot_file, clip_length=64, img_size=224,
                 split='train', augment=True, max_samples=None, augment_scale_range=(0.5, 1.5),
                 keyframe_position='center', frame_stride=None, supervised_frames=False,
                 uint8_clips=False, keep_empty_keyframes=False):
        if keyframe_position not in ('end', 'center'):
            raise ValueError('keyframe_position must be end or center')
        self.frames_root = frames_root
        self.clip_length = clip_length
        self.img_size = img_size
        self.split = split
        self.keyframe_position = keyframe_position
        self.frame_stride = None if frame_stride is None else int(frame_stride)
        self.emit_supervised_frames = bool(supervised_frames)
        self.uint8_clips = bool(uint8_clips)
        self.num_classes = AVA_NUM_CLASSES
        self.augmentor = (VideoAugmentor(img_size, scale_range=tuple(augment_scale_range))
                          if augment and split == 'train' else None)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        self.annot = defaultdict(list)
        annotated = set()
        with open(annot_file) as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                key = (row[0], _timestamp_key(row[1]))
                annotated.add(key)
                if len(row) < 7:
                    continue
                label = int(row[6])
                if label not in AVA_LABEL_MAP:
                    continue
                x1, y1, x2, y2 = (float(v) for v in row[2:6])
                self.annot[key].append((x1, y1, x2, y2, AVA_LABEL_MAP[label]))

        self.samples = []
        for path in sorted(glob.glob(os.path.join(manifest_dir, 'part_*.jsonl'))):
            with open(path) as handle:
                for line in handle:
                    record = json.loads(line)
                    if not record.get('ok'):
                        continue
                    for stamp, frame in record['keyframes'].items():
                        key = (record['youtube_id'], _timestamp_key(stamp))
                        if key not in annotated:
                            continue
                        if not self.annot.get(key) and not keep_empty_keyframes:
                            continue
                        if 1 <= int(frame) <= int(record['frames']):
                            self.samples.append((record['clip'], int(record['frames']), int(frame), key))
        if max_samples is not None and int(max_samples) < len(self.samples):
            step = len(self.samples) / int(max_samples)
            self.samples = [self.samples[int(i * step)] for i in range(int(max_samples))]
        print(f'AVA-Kinetics {split}: {len(self.samples)} keyframes from '
              f'{len({s[0] for s in self.samples})} clips')

    def __len__(self):
        return len(self.samples)

    def _key_clip_t(self):
        return self.clip_length - 1 if self.keyframe_position == 'end' else self.clip_length // 2

    def frame_indices(self, key_frame, num_frames, stride):
        key_clip_t = self._key_clip_t()
        return [max(1, min(key_frame + (i - key_clip_t) * stride, num_frames))
                for i in range(self.clip_length)]

    def __getitem__(self, idx):
        clip_name, num_frames, key_frame, key = self.samples[idx]
        if self.frame_stride is not None:
            stride = self.frame_stride
        elif self.augmentor is not None:
            stride = random.choice([1, 2, 3])
        else:
            stride = 1
        key_clip_t = self._key_clip_t()

        frames = []
        for fi in self.frame_indices(key_frame, num_frames, stride):
            img = cv2.imread(os.path.join(self.frames_root, clip_name, f'{fi:05d}.jpg'))
            frames.append(np.zeros((256, 340, 3), dtype=np.uint8) if img is None
                          else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        grouped = {}
        for (x1, y1, x2, y2, label) in self.annot.get(key, []):
            if x2 > x1 + 0.001 and y2 > y1 + 0.001:
                box_key = (round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5))
                if box_key not in grouped:
                    grouped[box_key] = [float(key_clip_t), x1, y1, x2, y2, []]
                grouped[box_key][5].append(label)
        gt_boxes, gt_labels = [], []
        for frame_t, x1, y1, x2, y2, labels in grouped.values():
            gt_boxes.append([frame_t, x1, y1, x2, y2])
            multi_hot = np.zeros(self.num_classes, dtype=np.float32)
            multi_hot[labels] = 1.0
            gt_labels.append(multi_hot)

        if self.augmentor is not None and gt_boxes:
            frames, gt_boxes, gt_labels = self.augmentor(frames, gt_boxes, gt_labels)
        elif self.augmentor is not None:
            frames, _, _ = self.augmentor(frames, [], [])

        height, width = _image_hw(self.img_size)
        clip = np.stack([cv2.resize(f, (width, height)) for f in frames])
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
        if self.uint8_clips:
            clip = clip.contiguous()
        else:
            clip = (clip.float() / 255.0 - self.mean) / self.std

        targets = {
            'boxes': torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros(0, 5),
            'labels': (torch.tensor(np.stack(gt_labels), dtype=torch.float32) if gt_labels
                       else torch.zeros(0, self.num_classes, dtype=torch.float32)),
        }
        if self.emit_supervised_frames:
            targets['supervised_frames'] = torch.tensor([key_clip_t], dtype=torch.long)
        return clip, targets
