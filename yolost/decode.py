"""Multi-label decoding of the dense head for keyframe benchmarks (AVA).

The frame evaluators keep one class per cell (argmax). AVA scores every
action for every person box, so this decoder keeps a cell when its actor score
(objectness) clears a threshold, applies class-agnostic NMS on that score, and
returns the full per-class probability vector for each surviving box.

Box decoding matches eval_tube_queries.decode_dense_frame exactly, for both
the distribution (DFL) head and the legacy sigmoid/exp head, including
non-square inputs (img_size as [height, width]).
"""

import torch
import torchvision

from .dfl import distribution_expectation
from .geometry import image_hw


def _decode_boxes(raw_box, row, column, step_x, step_y):
    if raw_box.shape[-1] != 4:
        reg_max = raw_box.shape[-1] // 4 - 1
        distance = distribution_expectation(raw_box.unsqueeze(0), reg_max)[0]
        anchor_x = (column.float() + 0.5) * step_x
        anchor_y = (row.float() + 0.5) * step_y
        return torch.stack([
            anchor_x - distance[:, 0] * step_x,
            anchor_y - distance[:, 1] * step_y,
            anchor_x + distance[:, 2] * step_x,
            anchor_y + distance[:, 3] * step_y,
        ], dim=-1)
    center_x = (column.float() + torch.sigmoid(raw_box[:, 0])) * step_x
    center_y = (row.float() + torch.sigmoid(raw_box[:, 1])) * step_y
    width = torch.exp(raw_box[:, 2].clamp(max=5.0)) * step_x
    height = torch.exp(raw_box[:, 3].clamp(max=5.0)) * step_y
    return torch.stack([
        center_x - width / 2, center_y - height / 2,
        center_x + width / 2, center_y + height / 2,
    ], dim=-1)


@torch.no_grad()
def decode_dense_multilabel(outputs, temporal_strides, spatial_strides, img_size,
                            batch_index, clip_frame, actor_thresh=0.2,
                            nms_thresh=0.5, max_detections=20):
    """Decode one clip frame of one batch element.

    Args:
        outputs: dense head output, a list of (cls, reg, obj, ...) per scale
            with shapes (B, C, T_scale, H_cells, W_cells).
        temporal_strides, spatial_strides: per-scale strides of the model.
        img_size: model input size, an int (square) or [height, width].
        batch_index: which clip in the batch.
        clip_frame: frame index inside the clip (e.g. the AVA keyframe).

    Returns:
        boxes (N, 4) normalised xyxy clamped to [0, 1], actor scores (N,),
        class probabilities (N, num_classes), sorted by actor score.
    """
    img_h, img_w = image_hw(img_size)
    boxes, actors, classes = [], [], []
    for scale_index, scale_out in enumerate(outputs):
        cls_pred, reg_pred, obj_pred = scale_out[:3]
        detection_frame = clip_frame // temporal_strides[scale_index]
        if detection_frame >= cls_pred.shape[2]:
            continue
        step_x = spatial_strides[scale_index] / img_w
        step_y = spatial_strides[scale_index] / img_h
        actor = torch.sigmoid(obj_pred[batch_index, 0, detection_frame].float())
        mask = actor > actor_thresh
        if not mask.any():
            continue
        row, column = torch.where(mask)
        raw_box = reg_pred[batch_index, :, detection_frame][:, row, column].T.float()
        boxes.append(_decode_boxes(raw_box, row, column, step_x, step_y))
        actors.append(actor[row, column])
        classes.append(torch.sigmoid(
            cls_pred[batch_index, :, detection_frame][:, row, column].T.float()))
    if not boxes:
        num_classes = outputs[0][0].shape[1]
        device = outputs[0][0].device
        return (torch.zeros(0, 4, device=device), torch.zeros(0, device=device),
                torch.zeros(0, num_classes, device=device))
    boxes = torch.cat(boxes).clamp(0.0, 1.0)
    actors = torch.cat(actors)
    classes = torch.cat(classes)
    keep = torchvision.ops.nms(boxes, actors, nms_thresh)[:max_detections]
    return boxes[keep], actors[keep], classes[keep]
