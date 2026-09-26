"""Tests for patch 0031 (boundary-negative frame weights). Plain asserts; run directly."""
import os
import sys

import numpy as np
import torch

REPO_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
sys.path.insert(0, REPO_ROOT)

from data.ucf101_24 import UCF101_24_Dataset  # noqa: E402
from yolost.head_boundary import DecoupledHeadBoundary  # noqa: E402
from yolost.loss_boundary import YOLOSTLossBoundary  # noqa: E402

NUM_CLASSES, FRAMES = 6, 8


def stub_dataset(window, weight):
    ds = object.__new__(UCF101_24_Dataset)
    ds.annot = {'gttubes': {'v': {0: [np.array([[5, 1, 1, 9, 9], [6, 1, 1, 9, 9], [7, 1, 1, 9, 9]])]},
                            'empty': {}}}
    ds.boundary_negative_window, ds.boundary_negative_weight = window, weight
    ds._annotated_frames_cache = {}
    return ds


def test_frame_weights():
    ds = stub_dataset(2, 4.0)
    got = ds._frame_weights('v', list(range(1, 12))).tolist()
    expect = [1, 1, 4, 4, 1, 1, 1, 4, 4, 1, 1]  # frames 1..11, annotated 5-7
    assert got == expect, got
    assert ds._frame_weights('empty', [1, 2, 3]).tolist() == [1, 1, 1]
    assert stub_dataset(0, 4.0)._frame_weights('v', [4]).tolist() == [1.0]


def predictions(seed=0):
    torch.manual_seed(seed)
    head = DecoupledHeadBoundary(in_channels_list=[32, 32, 32], num_classes=NUM_CLASSES, reg_max=16)
    feats = [torch.randn(2, 32, FRAMES, 28, 28), torch.randn(2, 32, FRAMES // 2, 14, 14),
             torch.randn(2, 32, FRAMES // 4, 7, 7)]
    return [tuple(scale) for scale in head(feats)]


def targets(frame_weights=None):
    boxes = torch.tensor([[[4.0, 0.30, 0.30, 0.62, 0.70], [5.0, 0.30, 0.30, 0.62, 0.70]],
                          [[4.0, 0.10, 0.20, 0.40, 0.90], [0.0, 0.0, 0.0, 0.0, 0.0]]])
    labels = torch.tensor([[1, 1], [5, 0]])
    out = {'boxes': boxes, 'labels': labels}
    if frame_weights is not None:
        out['frame_weights'] = frame_weights
    return out


def loss(t):
    criterion = YOLOSTLossBoundary(num_classes=NUM_CLASSES, lambda_bnd=0.0)
    return criterion(predictions(), t, temporal_strides=[1, 2, 4])


def test_unit_weights_are_bit_identical():
    plain, plain_parts = loss(targets())
    ones, ones_parts = loss(targets(torch.ones(2, FRAMES)))
    assert torch.equal(plain, ones), (plain, ones)
    for key in ('cls_loss', 'obj_loss', 'box_loss'):
        assert plain_parts[key] == ones_parts[key], key


def test_weights_raise_only_negative_terms():
    plain, plain_parts = loss(targets())
    w = torch.ones(2, FRAMES)
    w[0, 0:3] = 4.0   # clip 0 frames 0-2 carry no box
    w[1, 6:8] = 4.0   # clip 1 frames 6-7 carry no box
    weighted, parts = loss(targets(w))
    assert parts['box_loss'] == plain_parts['box_loss']
    assert parts['cls_loss'] > plain_parts['cls_loss']
    assert parts['obj_loss'] > plain_parts['obj_loss']
    assert torch.isfinite(weighted)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
