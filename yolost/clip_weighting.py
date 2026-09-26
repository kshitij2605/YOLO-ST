"""Score multipliers for merging detections from overlapping inference clips.

Every evaluator decodes each clip independently and multiplies detection scores
by a window over the clip's local frames before merging clips per frame. The
historical window is ``np.hanning(T + 2)[1:-1]``, which is 0.0023 at both clip
ends. A frame covered by only one clip, such as the first or last frames of a
video or every frame of a video no longer than one clip, therefore keeps
almost none of its score and ranks at the bottom of the global AP list.

Modes, all using the evaluator frame mapping
``global = min(start + local, num_frames) - 1``:

    hann                the historical window; reproduces every earlier number
    hann_coverage_norm  window / total window mass the frame receives over all
                        clips, so each frame's weights sum to 1
    hann_peak_norm      window / largest window value the frame receives, so
                        each frame's most central clip keeps weight 1
    none                every detection keeps its score (JHMDB: one clip)
"""

import numpy as np

MODES = ('hann', 'hann_coverage_norm', 'hann_peak_norm', 'none')


def hann_window(clip_length):
    return np.hanning(clip_length + 2)[1:-1]


def _global_frames(start, clip_length, num_frames):
    local = np.arange(clip_length)
    return np.minimum(start + local, num_frames) - 1


def clip_frame_weights(clip_starts, clip_length, num_frames, mode='hann'):
    """Return ``{start: weights}`` with ``weights[local_frame]`` per clip."""
    if mode not in MODES:
        raise ValueError(f'unknown clip weighting {mode!r}; expected one of {MODES}')
    if mode == 'none':
        return {start: np.ones(clip_length) for start in clip_starts}
    window = hann_window(clip_length)
    if mode == 'hann':
        return {start: window.copy() for start in clip_starts}

    total = np.zeros(num_frames)
    peak = np.zeros(num_frames)
    for start in clip_starts:
        frames = _global_frames(start, clip_length, num_frames)
        np.add.at(total, frames, window)
        np.maximum.at(peak, frames, window)
    reference = total if mode == 'hann_coverage_norm' else peak

    weights = {}
    for start in clip_starts:
        frames = _global_frames(start, clip_length, num_frames)
        weights[start] = window / reference[frames]
    return weights
