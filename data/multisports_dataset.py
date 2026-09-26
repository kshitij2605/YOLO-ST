"""MultiSports dataset adapter for YOLO-ST style training.

The official MultiSports release is distributed separately. This adapter expects
decoded RGB frames plus an annotation pickle in a UCF-style tube format:

{
    "train_videos": ["video_id", ...],
    "test_videos": ["video_id", ...],
    "nframes": {"video_id": 1234, ...},
    "resolution": {"video_id": (height, width), ...},
    "gttubes": {
        "video_id": {
            class_id: [np.ndarray[N, 5], ...]  # frame, x1, y1, x2, y2
        }
    }
}

This mirrors `data.ucf101_24.UCF101_24_Dataset`, but defaults to 66 classes,
matching MultiSports v1.0.
"""

from .ucf101_24 import UCF101_24_Dataset


class MultiSportsDataset(UCF101_24_Dataset):
    """Dense action-tube dataset adapter for MultiSports."""

    NUM_CLASSES = 66

    def __init__(
        self,
        root,
        annot_file,
        clip_length=64,
        stride=1,
        split="train",
        img_size=224,
        augment=True,
        boundary_labels=False,
        boundary_margin=3,
        filter_empty_clips=False,
        augment_scale_range=(0.5, 1.5),
    ):
        super().__init__(
            root=root,
            annot_file=annot_file,
            clip_length=clip_length,
            stride=stride,
            split=split,
            img_size=img_size,
            augment=augment,
            boundary_labels=boundary_labels,
            boundary_margin=boundary_margin,
            filter_empty_clips=filter_empty_clips,
            augment_scale_range=augment_scale_range,
        )
