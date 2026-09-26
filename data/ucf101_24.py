import os
import pickle
import random
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import VideoAugmentor


# DataLoader processes must not each create an OpenCV thread pool.
cv2.setNumThreads(0)

from yolost.geometry import image_hw as _image_hw


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy 2 pickles on NumPy 1 without changing their payload."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == 'numpy._core' or module.startswith('numpy._core.'):
                legacy_module = f"numpy.core{module[len('numpy._core'):]}"
                return super().find_class(legacy_module, name)
            raise


def build_clip_starts(num_frames, clip_length, stride=1, overlap=0.5):
    """Return 1-indexed starts that cover a video, including its final frame."""
    span = (clip_length - 1) * stride + 1
    last_start = max(1, num_frames - span + 1)
    step = max(1, int(span * (1.0 - overlap)))
    starts = list(range(1, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


class UCF101_24_Dataset(Dataset):
    """UCF101-24 dataset for spatiotemporal action detection.

    Args:
        root: Path to rgb-images directory.
        annot_file: Path to UCF101v2-GT.pkl.
        clip_length: Number of frames per clip (default 64).
        stride: Temporal stride for frame sampling (default 1).
        split: 'train' or 'test' (split 1).
        img_size: Spatial resolution to resize frames to.
        augment: Whether to apply augmentations.
    """

    CLASSES = [
        'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
        'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
        'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
        'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing',
        'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog',
    ]

    NUM_CLASSES = 24

    # Index into annot['train_videos'] / annot['test_videos'].
    # UCF101-24 and MultiSports ship one split; JHMDB-21 ships three.
    SPLIT_INDEX = 0

    def __init__(self, root, annot_file, clip_length=64, stride=1,
                 split='train', img_size=320, augment=True,
                 boundary_labels=False, boundary_margin=3,
                 filter_empty_clips=False, empty_clip_repeats=1,
                 augment_scale_range=(0.5, 1.5),
                 offline_track_dir=None, offline_min_quality=0.15,
                 offline_min_score=0.03,
                 offline_min_clip_observations=4,
                 offline_max_tracks=16,
                 split_index=None, temporal_jitter=0,
                 boundary_negative_window=0, boundary_negative_weight=1.0):
        super().__init__()
        self.root = root
        self.split_index = (
            self.SPLIT_INDEX if split_index is None else int(split_index)
        )
        self.clip_length = clip_length
        # D-T1: random start-frame offset during training. 0 leaves the parent
        # behaviour bit-identical, RNG stream included, since no draw is taken.
        self.temporal_jitter = max(0, int(temporal_jitter))
        self.stride = stride
        self.img_size = img_size
        self.split = split
        self.boundary_labels = boundary_labels
        self.boundary_margin = boundary_margin
        self.filter_empty_clips = filter_empty_clips
        self.empty_clip_repeats = max(1, int(empty_clip_repeats))
        # Patch 0031: dense-loss weight for unannotated frames within this many
        # frames of an annotated one. 0 disables it and adds nothing to targets.
        self.boundary_negative_window = max(0, int(boundary_negative_window))
        self.boundary_negative_weight = float(boundary_negative_weight)
        self._annotated_frames_cache = {}
        self.offline_track_dir = offline_track_dir
        self.offline_min_quality = float(offline_min_quality)
        self.offline_min_score = float(offline_min_score)
        self.offline_min_clip_observations = max(
            1, int(offline_min_clip_observations)
        )
        self.offline_max_tracks = max(1, int(offline_max_tracks))
        self._offline_track_cache = OrderedDict()
        self.augmentor = (
            VideoAugmentor(img_size, scale_range=tuple(augment_scale_range))
            if augment and split == 'train' else None
        )

        # ImageNet normalisation constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

        # Load annotations
        with open(annot_file, 'rb') as f:
            self.annot = pickle.load(f, encoding='latin1')

        # Build clip index: list of (video_name, start_frame_1indexed, num_frames) tuples
        self.clips = []
        split_key = 'train_videos' if split == 'train' else 'test_videos'
        split_lists = self.annot[split_key]
        if not 0 <= self.split_index < len(split_lists):
            raise ValueError(
                f'{split_key} has {len(split_lists)} split list(s); '
                f'split_index={self.split_index} is out of range'
            )
        video_list = split_lists[self.split_index]

        skipped = 0
        for video_name in video_list:
            video_dir = os.path.join(root, video_name)
            if not os.path.isdir(video_dir):
                skipped += 1
                continue
            num_frames = self.annot['nframes'][video_name]
            # Clips with 50% overlap, frames are 1-indexed in the annotation.
            # Always append a tail-aligned clip so terminal annotations are seen.
            for start in build_clip_starts(num_frames, clip_length, stride):
                has_gt = True
                if filter_empty_clips or self.empty_clip_repeats > 1:
                    has_gt = self._clip_has_gt(video_name, start, num_frames)
                if filter_empty_clips and not has_gt:
                    continue
                repeats = 1 if has_gt else self.empty_clip_repeats
                self.clips.extend([(video_name, start, num_frames)] * repeats)

        if skipped > 0:
            print(f'UCF101-24 {split}: skipped {skipped} videos (missing dirs)')

    def __len__(self):
        return len(self.clips)

    def _frame_weights(self, video_name, frame_indices):
        """Per clip frame: boundary_negative_weight on a frame with no annotation
        within boundary_negative_window frames of an annotated frame, else 1."""
        annotated = self._annotated_frames_cache.get(video_name)
        if annotated is None:
            frames = set()
            for tubes in self.annot.get('gttubes', {}).get(video_name, {}).values():
                for tube in tubes:
                    frames.update(int(row[0]) for row in tube)
            annotated = np.array(sorted(frames), dtype=np.int64)
            self._annotated_frames_cache[video_name] = annotated
        weights = np.ones(len(frame_indices), dtype=np.float32)
        if annotated.size == 0:
            return torch.from_numpy(weights)
        for t, frame in enumerate(frame_indices):
            position = int(np.searchsorted(annotated, frame))
            if position < annotated.size and int(annotated[position]) == int(frame):
                continue
            nearest = min(abs(int(annotated[p]) - int(frame))
                          for p in (position - 1, position)
                          if 0 <= p < annotated.size)
            if nearest <= self.boundary_negative_window:
                weights[t] = self.boundary_negative_weight
        return torch.from_numpy(weights)

    def _clip_has_gt(self, video_name, start_frame, num_frames):
        """Return whether the clip contains at least one valid GT frame."""
        effective_stride = self.stride
        span_frames = {
            min(start_frame + i * effective_stride, num_frames)
            for i in range(self.clip_length)
        }
        video_annot = self.annot.get('gttubes', {}).get(video_name, {})
        for tubes in video_annot.values():
            for tube in tubes:
                for row in tube:
                    if int(row[0]) in span_frames:
                        return True
        return False

    @staticmethod
    def _offline_filename(video_name):
        return f"{video_name.replace('/', '__')}.pkl"

    def _offline_tracks(self, video_name):
        if self.offline_track_dir is None:
            return []
        if video_name in self._offline_track_cache:
            payload = self._offline_track_cache.pop(video_name)
            self._offline_track_cache[video_name] = payload
            return payload
        path = os.path.join(
            self.offline_track_dir, self._offline_filename(video_name)
        )
        if not os.path.isfile(path):
            return []
        with open(path, 'rb') as handle:
            payload = _NumpyCompatUnpickler(handle).load()
        if payload.get('video') != video_name:
            raise ValueError(
                f'Offline trajectory cache mismatch for {video_name}: {path}'
            )
        tracks = payload.get('tracks', [])
        self._offline_track_cache[video_name] = tracks
        while len(self._offline_track_cache) > 32:
            self._offline_track_cache.popitem(last=False)
        return tracks

    def _offline_clip_observations(self, video_name, frame_indices):
        if self.offline_track_dir is None:
            return []
        frame_to_times = {}
        for clip_time, frame in enumerate(frame_indices):
            frame_to_times.setdefault(int(frame), []).append(clip_time)
        observations = []
        tracks = sorted(
            self._offline_tracks(video_name),
            key=lambda track: float(track.get('quality', 0.0)),
            reverse=True,
        )[:self.offline_max_tracks]
        for local_track_id, track in enumerate(tracks):
            quality = float(track.get('quality', 0.0))
            if quality < self.offline_min_quality:
                continue
            track_observations = []
            observed_flags = track.get(
                'observed', np.ones(len(track['frames']), dtype=np.bool_)
            )
            for frame, box, score, observed in zip(
                    track['frames'], track['boxes'], track['scores'],
                    observed_flags):
                score = float(score)
                if score < self.offline_min_score:
                    continue
                for clip_time in frame_to_times.get(int(frame) + 1, []):
                    track_observations.append({
                        'track_id': local_track_id,
                        'label': int(track['label']),
                        'clip_time': clip_time,
                        'box': np.asarray(box, dtype=np.float32),
                        'score': score,
                        'observed': bool(observed),
                        'quality': quality,
                    })
            if len(track_observations) >= self.offline_min_clip_observations:
                observations.extend(track_observations)
        return observations

    def __getitem__(self, idx):
        video_name, start_frame, num_frames = self.clips[idx]

        # Temporal stride augmentation during training
        effective_stride = self.stride
        if self.augmentor is not None:
            effective_stride = random.choice([1, 2, 3])

        # D-T1: jitter the clip start, clamped so the whole clip stays inside
        # the video. Starts otherwise come from the fixed build_clip_starts grid.
        if self.augmentor is not None and self.temporal_jitter > 0:
            span = (self.clip_length - 1) * effective_stride + 1
            last_start = max(1, num_frames - span + 1)
            offset = random.randint(-self.temporal_jitter, self.temporal_jitter)
            start_frame = min(max(1, start_frame + offset), last_start)

        # Build list of 1-indexed frame indices to load
        frame_indices = []
        for i in range(self.clip_length):
            fi = start_frame + i * effective_stride
            fi = min(fi, num_frames)  # clamp to last frame (1-indexed)
            frame_indices.append(fi)

        # Get resolution for normalizing box coordinates
        resolution = self.annot['resolution'].get(video_name, (240, 320))
        orig_h, orig_w = resolution

        # Load frames
        frames = []
        for fi in frame_indices:
            img_path = os.path.join(self.root, video_name, f'{fi:05d}.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(self.root, video_name, f'{fi:05d}.png')
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        # Load ground truth for these frames
        # gttubes format: {class_id: [tube_array, ...]}
        # tube_array shape: (N, 5) with (frame_1indexed, x1, y1, x2, y2) in pixels
        # class_id is already 0-indexed
        gt_boxes = []
        gt_labels = []
        video_annot = self.annot.get('gttubes', {}).get(video_name, {})

        # Pre-compute tube boundaries within clip (for boundary labels)
        tube_boundary_frames = {}  # (class_id, tube_idx) -> (start_t, end_t) in clip
        if self.boundary_labels:
            for class_id, tubes in video_annot.items():
                for ti, tube in enumerate(tubes):
                    clip_ts = []
                    for row in tube:
                        if int(row[0]) in frame_indices:
                            clip_ts.append(frame_indices.index(int(row[0])))
                    if clip_ts:
                        tube_boundary_frames[(class_id, ti)] = (min(clip_ts), max(clip_ts))

        for class_id, tubes in video_annot.items():
            for ti, tube in enumerate(tubes):
                for row in tube:
                    gt_frame = int(row[0])  # 1-indexed frame number
                    if gt_frame in frame_indices:
                        clip_t = frame_indices.index(gt_frame)
                        box = [
                            float(clip_t),
                            row[1] / orig_w,   # x1 normalised
                            row[2] / orig_h,   # y1 normalised
                            row[3] / orig_w,   # x2 normalised
                            row[4] / orig_h,   # y2 normalised
                        ]
                        # Clamp to [0, 1]
                        box[1] = max(0.0, min(1.0, box[1]))
                        box[2] = max(0.0, min(1.0, box[2]))
                        box[3] = max(0.0, min(1.0, box[3]))
                        box[4] = max(0.0, min(1.0, box[4]))
                        if box[3] > box[1] + 0.001 and box[4] > box[2] + 0.001:
                            if self.boundary_labels:
                                # Compute boundary flag and embed as 6th element
                                is_bnd = 0.0
                                if (class_id, ti) in tube_boundary_frames:
                                    start_t, end_t = tube_boundary_frames[(class_id, ti)]
                                    margin = self.boundary_margin
                                    if (abs(clip_t - start_t) <= margin or
                                            abs(clip_t - end_t) <= margin):
                                        is_bnd = 1.0
                                gt_boxes.append(box + [is_bnd])
                            else:
                                gt_boxes.append(box)
                            gt_labels.append(int(class_id))

        offline_observations = self._offline_clip_observations(
            video_name, frame_indices
        )

        # Apply augmentations (jointly to all frames + boxes + labels)
        if self.augmentor is not None:
            combined_boxes = list(gt_boxes)
            combined_labels = [('gt', label) for label in gt_labels]
            for observation in offline_observations:
                combined_boxes.append([
                    float(observation['clip_time']),
                    *observation['box'].tolist(),
                ])
                combined_labels.append((
                    'offline', observation['track_id'], observation['label'],
                    observation['score'], observation['observed'],
                    observation['quality'],
                ))
            frames, combined_boxes, combined_labels = self.augmentor(
                frames, combined_boxes, combined_labels
            )
            gt_boxes, gt_labels, offline_observations = [], [], []
            for box, descriptor in zip(combined_boxes, combined_labels):
                if descriptor[0] == 'gt':
                    gt_boxes.append(box)
                    gt_labels.append(descriptor[1])
                    continue
                offline_observations.append({
                    'track_id': descriptor[1],
                    'label': descriptor[2],
                    'clip_time': int(box[0]),
                    'box': np.asarray(box[1:5], dtype=np.float32),
                    'score': descriptor[3],
                    'observed': descriptor[4],
                    'quality': descriptor[5],
                })

        # Extract boundary flags from 6th box element (if present)
        gt_boundary = []
        if self.boundary_labels and gt_boxes:
            gt_boundary = [b[5] for b in gt_boxes]
            gt_boxes = [b[:5] for b in gt_boxes]
        elif self.boundary_labels:
            gt_boxes = [b[:5] for b in gt_boxes]  # strip 6th element even if empty

        # Resize all frames to target size
        resized = []
        for img in frames:
            _h, _w = _image_hw(self.img_size)
            resized.append(cv2.resize(img, (_w, _h)))
        frames = resized

        # Stack: (T, H, W, 3) -> (3, T, H, W)
        clip = np.stack(frames)
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        clip = (clip - self.mean) / self.std

        targets = {
            'boxes': (torch.tensor(gt_boxes, dtype=torch.float32)
                      if gt_boxes else torch.zeros(0, 5)),
            'labels': (torch.tensor(gt_labels, dtype=torch.long)
                       if gt_labels else torch.zeros(0, dtype=torch.long)),
        }
        if self.boundary_labels:
            targets['boundary'] = (torch.tensor(gt_boundary, dtype=torch.float32)
                                   if gt_boundary else torch.zeros(0))
        if self.boundary_negative_window > 0 and self.split == 'train':
            targets['frame_weights'] = self._frame_weights(video_name, frame_indices)
        if self.offline_track_dir is not None:
            targets.update({
                'offline_boxes': (
                    torch.tensor([
                        [observation['clip_time'], *observation['box'].tolist()]
                        for observation in offline_observations
                    ], dtype=torch.float32)
                    if offline_observations else torch.zeros(0, 5)
                ),
                'offline_labels': (
                    torch.tensor([
                        observation['label']
                        for observation in offline_observations
                    ], dtype=torch.long)
                    if offline_observations else torch.zeros(0, dtype=torch.long)
                ),
                'offline_track_ids': (
                    torch.tensor([
                        observation['track_id']
                        for observation in offline_observations
                    ], dtype=torch.long)
                    if offline_observations else torch.zeros(0, dtype=torch.long)
                ),
                'offline_scores': (
                    torch.tensor([
                        observation['score']
                        for observation in offline_observations
                    ], dtype=torch.float32)
                    if offline_observations else torch.zeros(0)
                ),
                'offline_quality': (
                    torch.tensor([
                        observation['quality']
                        for observation in offline_observations
                    ], dtype=torch.float32)
                    if offline_observations else torch.zeros(0)
                ),
                'offline_observed': (
                    torch.tensor([
                        observation['observed']
                        for observation in offline_observations
                    ], dtype=torch.bool)
                    if offline_observations
                    else torch.zeros(0, dtype=torch.bool)
                ),
            })

        return clip, targets
