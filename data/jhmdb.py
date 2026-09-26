"""JHMDB-21 dataset adapter for YOLO-ST style training and evaluation.

JHMDB-21 ships in the same UCF-style tube pickle format used by
`data.ucf101_24.UCF101_24_Dataset`, with three differences that this adapter
handles:

1. Three official splits. `train_videos` and `test_videos` are three-element
   lists, so `split_index` in {0, 1, 2} selects split 1, 2 or 3. Published
   JHMDB numbers are usually the average over all three splits; papers that
   report split 1 only must say so.
2. Twenty-one classes instead of twenty-four.
3. Short clips. Videos are 15 to 40 frames long and every ground-truth tube
   spans the whole video, so the natural protocol is whole-video processing
   with no cross-clip linking, as STAR (CVPR 2024) does. With the default
   ``clip_length=40`` the clip index yields exactly one clip per video, and the
   parent loader clamps frame indices to the last frame, repeating it for
   videos shorter than 40 frames.

Expected layout::

    <root>/<class_name>/<video_name>/00001.png
    <annot_file>  -> JHMDB-GT.pkl

Video identifiers in the pickle already include the class directory, e.g.
``brush_hair/Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_ba_med_3``,
so ``root`` points at the ``Frames`` directory.
"""

from .ucf101_24 import UCF101_24_Dataset


class JHMDBDataset(UCF101_24_Dataset):
    """Action-tube dataset adapter for JHMDB-21."""

    # Official label order, matching the `labels` field of JHMDB-GT.pkl.
    CLASSES = [
        'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump',
        'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball',
        'shoot_bow', 'shoot_gun', 'sit', 'stand', 'swing_baseball', 'throw',
        'walk', 'wave',
    ]

    NUM_CLASSES = 21

    #: Number of official splits in the annotation pickle.
    NUM_SPLITS = 3

    def __init__(
        self,
        root,
        annot_file,
        clip_length=40,
        stride=1,
        split='train',
        img_size=224,
        augment=True,
        boundary_labels=False,
        boundary_margin=3,
        filter_empty_clips=False,
        empty_clip_repeats=1,
        augment_scale_range=(0.5, 1.5),
        split_index=0,
    ):
        if not 0 <= int(split_index) < self.NUM_SPLITS:
            raise ValueError(
                f'JHMDB-21 has {self.NUM_SPLITS} splits; '
                f'got split_index={split_index}'
            )
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
            empty_clip_repeats=empty_clip_repeats,
            augment_scale_range=augment_scale_range,
            split_index=split_index,
        )

    @property
    def whole_video_clips(self):
        """True when the clip index holds exactly one clip per video.

        JHMDB evaluation is normally done without cross-clip tube linking. This
        is the check that the configured ``clip_length`` is long enough for
        that to hold.
        """
        return len(self.clips) == len({clip[0] for clip in self.clips})
