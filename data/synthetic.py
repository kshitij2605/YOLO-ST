import random

import torch
import numpy as np
from torch.utils.data import Dataset


class SyntheticActionDataset(Dataset):
    """Synthetic dataset for verifying the YOLO-ST training pipeline.

    Generates random video clips with random bounding box annotations.
    Replace with UCF101_24_Dataset once the real data is available.

    Args:
        num_samples: Number of clips in the dataset.
        clip_length: Temporal frames per clip.
        img_size: Spatial resolution.
        num_classes: Number of action classes.
        max_actors: Maximum actors per clip.
    """

    def __init__(self, num_samples=500, clip_length=64, img_size=320,
                 num_classes=24, max_actors=4):
        super().__init__()
        self.num_samples = num_samples
        self.clip_length = clip_length
        self.img_size = img_size
        self.num_classes = num_classes
        self.max_actors = max_actors

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random clip: (3, T, H, W)
        clip = torch.randn(3, self.clip_length, self.img_size, self.img_size)
        clip = clip * 0.2 + 0.5  # centre around 0.5
        clip = (clip - self.mean) / self.std

        # Generate random actors with smooth trajectories
        n_actors = random.randint(1, self.max_actors)
        gt_boxes = []
        gt_labels = []

        for _ in range(n_actors):
            cls = random.randint(0, self.num_classes - 1)
            # Random start position and size
            cx = random.uniform(0.2, 0.8)
            cy = random.uniform(0.2, 0.8)
            w = random.uniform(0.08, 0.35)
            h = random.uniform(0.08, 0.35)
            # Slow random drift per frame
            vx = random.uniform(-0.003, 0.003)
            vy = random.uniform(-0.003, 0.003)

            # Actor present in a contiguous sub-range of frames
            start_t = random.randint(0, self.clip_length // 4)
            end_t = random.randint(self.clip_length * 3 // 4, self.clip_length - 1)

            for t in range(start_t, end_t + 1):
                cur_cx = cx + vx * (t - start_t)
                cur_cy = cy + vy * (t - start_t)
                x1 = max(0.0, min(1.0, cur_cx - w / 2))
                y1 = max(0.0, min(1.0, cur_cy - h / 2))
                x2 = max(0.0, min(1.0, cur_cx + w / 2))
                y2 = max(0.0, min(1.0, cur_cy + h / 2))
                if x2 > x1 + 0.01 and y2 > y1 + 0.01:
                    gt_boxes.append([float(t), x1, y1, x2, y2])
                    gt_labels.append(cls)

        targets = {
            'boxes': (torch.tensor(gt_boxes, dtype=torch.float32)
                      if gt_boxes else torch.zeros(0, 5)),
            'labels': (torch.tensor(gt_labels, dtype=torch.long)
                       if gt_labels else torch.zeros(0, dtype=torch.long)),
        }

        return clip, targets
