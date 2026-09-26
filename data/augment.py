import random

import cv2
import numpy as np


class VideoAugmentor:
    """Video-aware augmentations applied consistently across all T frames.

    Supports:
        - Random crop + resize (scale 0.5-1.5)
        - Random horizontal flip with box mirroring
        - Color jitter (brightness, contrast, saturation, hue)
        - Random grayscale (forces motion-based features)
        - Random erasing (occlusion robustness)

    All spatial transforms are applied identically to every frame in the clip.
    """

    def __init__(self, img_size=320, scale_range=(0.5, 1.5),
                 flip_prob=0.5, color_jitter=True,
                 grayscale_prob=0.1, erasing_prob=0.3):
        self.img_size = img_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter
        self.grayscale_prob = grayscale_prob
        self.erasing_prob = erasing_prob

    def __call__(self, frames, gt_boxes, gt_labels=None):
        """
        Args:
            frames: list of T numpy images (H, W, 3) uint8.
            gt_boxes: list of [frame_idx, x1, y1, x2, y2] (normalised).
            gt_labels: list of class indices (same length as gt_boxes).

        Returns:
            frames: augmented frames.
            gt_boxes: adjusted boxes.
            gt_labels: filtered labels (if provided).
        """
        # Random horizontal flip
        if random.random() < self.flip_prob:
            frames = [cv2.flip(f, 1) for f in frames]
            gt_boxes = [
                [b[0], 1.0 - b[3], b[2], 1.0 - b[1], b[4]] + b[5:]
                for b in gt_boxes
            ]

        # Random crop + resize
        if self.scale_range != (1.0, 1.0):
            frames, gt_boxes, gt_labels = self._random_crop_resize(
                frames, gt_boxes, gt_labels)

        # Color jitter
        if self.color_jitter:
            frames = self._apply_color_jitter(frames)

        # Random grayscale (forces model to use motion, not color)
        if random.random() < self.grayscale_prob:
            frames = [cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),
                                    cv2.COLOR_GRAY2RGB) for f in frames]

        # Random erasing (occlusion robustness)
        if random.random() < self.erasing_prob:
            frames = self._random_erasing(frames)

        if gt_labels is not None:
            return frames, gt_boxes, gt_labels
        return frames, gt_boxes

    def _random_crop_resize(self, frames, gt_boxes, gt_labels=None):
        """Random spatial crop at random scale, applied to all frames."""
        h, w = frames[0].shape[:2]
        scale = random.uniform(*self.scale_range)

        new_h = int(h * scale)
        new_w = int(w * scale)

        if scale < 1.0:
            # Zoom in: crop a smaller region
            crop_h = int(h * scale)
            crop_w = int(w * scale)
            top = random.randint(0, h - crop_h) if h > crop_h else 0
            left = random.randint(0, w - crop_w) if w > crop_w else 0

            frames = [f[top:top+crop_h, left:left+crop_w] for f in frames]

            # Adjust normalised box coordinates
            new_boxes = []
            new_labels = []
            for i, b in enumerate(gt_boxes):
                x1 = (b[1] * w - left) / crop_w
                y1 = (b[2] * h - top) / crop_h
                x2 = (b[3] * w - left) / crop_w
                y2 = (b[4] * h - top) / crop_h
                # Clip to [0, 1]
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                if x2 > x1 and y2 > y1:
                    new_boxes.append([b[0], x1, y1, x2, y2] + b[5:])
                    if gt_labels is not None:
                        new_labels.append(gt_labels[i])
            gt_boxes = new_boxes
            if gt_labels is not None:
                gt_labels = new_labels
        else:
            # Zoom out: pad and place image at random position
            pad_h = new_h - h
            pad_w = new_w - w
            top = random.randint(0, pad_h) if pad_h > 0 else 0
            left = random.randint(0, pad_w) if pad_w > 0 else 0

            padded = []
            for f in frames:
                canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
                canvas[top:top+h, left:left+w] = f
                padded.append(canvas)
            frames = padded

            # Adjust normalised box coordinates
            new_boxes = []
            for b in gt_boxes:
                x1 = (b[1] * w + left) / new_w
                y1 = (b[2] * h + top) / new_h
                x2 = (b[3] * w + left) / new_w
                y2 = (b[4] * h + top) / new_h
                new_boxes.append([b[0], x1, y1, x2, y2] + b[5:])
            gt_boxes = new_boxes

        return frames, gt_boxes, gt_labels

    def _random_erasing(self, frames):
        """Randomly erase a rectangular region across all frames."""
        h, w = frames[0].shape[:2]
        # Erase 2-20% of the image area
        area_ratio = random.uniform(0.02, 0.20)
        aspect = random.uniform(0.3, 3.3)
        erase_h = int((h * w * area_ratio / aspect) ** 0.5)
        erase_w = int(erase_h * aspect)
        erase_h = min(erase_h, h - 1)
        erase_w = min(erase_w, w - 1)
        top = random.randint(0, h - erase_h)
        left = random.randint(0, w - erase_w)
        # Fill with dataset mean (gray 114)
        for f in frames:
            f[top:top+erase_h, left:left+erase_w] = 114
        return frames

    def _apply_color_jitter(self, frames):
        """Apply consistent color jitter across all frames."""
        brightness = random.uniform(0.6, 1.4)
        contrast = random.uniform(0.6, 1.4)
        saturation = random.uniform(0.6, 1.4)
        hue_shift = random.uniform(-0.1, 0.1)

        augmented = []
        for f in frames:
            img = f.astype(np.float32)

            # Brightness
            img = img * brightness

            # Contrast
            mean = img.mean()
            img = (img - mean) * contrast + mean

            # Saturation
            hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation

            # Hue
            hsv[:, :, 0] = hsv[:, :, 0] + hue_shift * 180
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            augmented.append(img)

        return augmented
