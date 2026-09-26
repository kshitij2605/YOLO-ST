"""Weak AVA-Kinetics/Kinetics-style supervision adapter.

This is the first code scaffold for SiA-style weak supervision. It wraps AVA
keyframe boxes and adds a global video action label as a weak action target for
each visible person box. A later AWS pass should replace this naive assignment
with model-based Hungarian assignment to the most relevant actor.
"""

import csv

import torch

from .ava_dataset import AVADataset


class AVAKineticsWeakDataset(AVADataset):
    """AVA boxes with appended weak global Kinetics labels."""

    def __init__(
        self,
        frames_root,
        annot_file,
        weak_label_file,
        clip_length=64,
        img_size=224,
        split="train",
        augment=True,
        multi_label=True,
        num_classes=700,
        augment_scale_range=(0.5, 1.5),
    ):
        self.weak_labels = self._load_weak_labels(weak_label_file)
        self.weak_num_classes = num_classes
        super().__init__(
            frames_root=frames_root,
            annot_file=annot_file,
            clip_length=clip_length,
            img_size=img_size,
            split=split,
            augment=augment,
            multi_label=multi_label,
            augment_scale_range=augment_scale_range,
        )
        self.num_classes = num_classes

    @staticmethod
    def _load_weak_labels(path):
        labels = {}
        with open(path) as f:
            for row in csv.reader(f):
                if not row:
                    continue
                # Expected columns: video_id, class_id
                labels[row[0]] = int(row[1])
        return labels

    def __getitem__(self, idx):
        clip, targets = super().__getitem__(idx)
        vid, _ = self.samples[idx]
        weak_label = self.weak_labels.get(vid)
        if weak_label is None or targets["boxes"].shape[0] == 0:
            targets["labels"] = torch.zeros(targets["boxes"].shape[0], self.weak_num_classes)
            return clip, targets

        labels = torch.zeros(targets["boxes"].shape[0], self.weak_num_classes)
        labels[:, weak_label] = 1.0
        targets["labels"] = labels
        return clip, targets
