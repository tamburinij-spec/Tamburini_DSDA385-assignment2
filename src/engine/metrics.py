"""Helper functions to compute evaluation metrics."""

import torch


def iou_score(pred, target):
    """Compute simple IoU for binary segmentation predictions."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return torch.tensor(1.0)
    return intersection / union
