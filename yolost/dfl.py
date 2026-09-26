"""Distribution-based box regression for the YOLO-ST dense head.

Why
---
The dense head currently regresses a box as four direct numbers: two
sigmoid-squashed centre offsets and two exponentiated sizes. One number per
side means the head cannot express uncertainty about an edge, and the gradient
it receives is whatever the IoU loss backs into that single value.

Distribution Focal Loss replaces each side with a small categorical
distribution over discrete distances, and decodes by taking its expectation.
An edge the model is unsure about becomes a spread distribution rather than a
confidently wrong number, and the loss supervises the two bins that bracket the
target. This is the parameterisation used by YOLOv8 and YOLO11, and by
YOWOFormer, which reaches 92.48 frame mAP on UCF101-24 with the same
VideoMAE-Large weights our dense owner tops out at 90.54 with. D-FINE (Peng et al.,
2025) reports the refinement of this idea as worth up to +5.3 AP across DETR
variants, with the gains concentrated at high IoU, which is where our video
metric collapses.

Parameterisation
----------------
Boxes become **LTRB distances from the anchor point**, measured in cell units,
rather than centre and size. For anchor ``a`` and stride step ``s``::

    x1 = a_x - l * s      x2 = a_x + r * s
    y1 = a_y - t * s      y2 = a_y + b * s

Each of the four distances is predicted as ``reg_max + 1`` logits. The decoded
distance is the softmax expectation over bin indices ``0..reg_max``, so a
distance is representable up to ``reg_max`` cells. With stride 8 at 224 pixels
and ``reg_max=16``, one side reaches 128 pixels, comfortably more than half the
largest actor box.

``reg_max=0`` means the feature is off and the caller should use the original
four-channel decode. Nothing here is reached in that case.
"""

import torch
import torch.nn.functional as F


def regression_channels(reg_max):
    """Number of regression channels the head must emit."""
    if reg_max <= 0:
        return 4
    return 4 * (reg_max + 1)


def distribution_expectation(reg_flat, reg_max):
    """Softmax expectation of each side's distribution.

    Args:
        reg_flat: ``(N, A, 4 * (reg_max + 1))`` raw logits.
        reg_max: largest representable distance in cell units.

    Returns:
        ``(N, A, 4)`` expected LTRB distances in cell units.
    """
    if reg_max <= 0:
        raise ValueError('distribution_expectation requires reg_max > 0')
    *lead, channels = reg_flat.shape
    expected = regression_channels(reg_max)
    if channels != expected:
        raise ValueError(
            f'expected {expected} regression channels for reg_max={reg_max}, '
            f'got {channels}'
        )
    logits = reg_flat.reshape(*lead, 4, reg_max + 1)
    probability = logits.softmax(dim=-1)
    bins = torch.arange(
        reg_max + 1, device=reg_flat.device, dtype=probability.dtype
    )
    return (probability * bins).sum(dim=-1)


def _split_step(step):
    """Scalar step (square inputs) or (step_x, step_y) for non-square inputs."""
    if isinstance(step, (tuple, list)):
        return step[0], step[1]
    return step, step


def decode_ltrb(distance, anchors, step):
    """Turn LTRB cell distances into normalised ``xyxy`` boxes.

    Args:
        distance: ``(N, A, 4)`` LTRB in cell units.
        anchors: ``(A, 2)`` normalised anchor centres.
        step: stride divided by image size.
    """
    left, top, right, bottom = distance.unbind(-1)
    anchor_x = anchors[:, 0]
    anchor_y = anchors[:, 1]
    step_x, step_y = _split_step(step)
    return torch.stack([
        anchor_x - left * step_x,
        anchor_y - top * step_y,
        anchor_x + right * step_x,
        anchor_y + bottom * step_y,
    ], dim=-1)


def encode_ltrb(boxes, anchors, step, reg_max):
    """Inverse of :func:`decode_ltrb`, clamped to the representable range.

    Targets are clamped just below ``reg_max`` so that the upper bracketing bin
    always exists, which is what YOLOv8 does.
    """
    anchor_x = anchors[:, 0]
    anchor_y = anchors[:, 1]
    step_x, step_y = _split_step(step)
    left = (anchor_x - boxes[..., 0]) / step_x
    top = (anchor_y - boxes[..., 1]) / step_y
    right = (boxes[..., 2] - anchor_x) / step_x
    bottom = (boxes[..., 3] - anchor_y) / step_y
    distance = torch.stack([left, top, right, bottom], dim=-1)
    return distance.clamp_(0, reg_max - 0.01)


def distribution_focal_loss(reg_logits, target_distance, reg_max):
    """Cross-entropy against the two bins bracketing each target distance.

    Args:
        reg_logits: ``(M, 4 * (reg_max + 1))`` logits for M selected cells.
        target_distance: ``(M, 4)`` target LTRB in cell units.
        reg_max: largest representable distance.

    Returns:
        ``(M,)`` loss, averaged over the four sides of each box.
    """
    logits = reg_logits.reshape(-1, reg_max + 1)
    target = target_distance.reshape(-1)
    lower = target.floor().long()
    upper = lower + 1
    weight_upper = target - lower.to(target.dtype)
    weight_lower = 1.0 - weight_upper

    log_probability = F.log_softmax(logits.float(), dim=-1)
    loss_lower = -log_probability.gather(1, lower.clamp(0, reg_max).unsqueeze(1)).squeeze(1)
    loss_upper = -log_probability.gather(1, upper.clamp(0, reg_max).unsqueeze(1)).squeeze(1)
    per_side = loss_lower * weight_lower + loss_upper * weight_upper
    return per_side.reshape(-1, 4).mean(dim=-1)
