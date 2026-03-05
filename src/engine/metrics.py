"""
Helper functions to compute evaluation metrics for segmentation / detection tasks.
"""

import torch


def iou_score(pred, target):
    """
    Compute Intersection over Union (IoU) for binary segmentation.

    Args:
        pred (Tensor): predicted mask
        target (Tensor): ground truth mask

    Returns:
        float: IoU score
    """

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / union).item()


def mean_iou(preds, targets, num_classes):
    """
    Compute mean IoU across multiple classes.

    Args:
        preds (Tensor): predicted labels
        targets (Tensor): ground truth labels
        num_classes (int): number of classes

    Returns:
        float: mean IoU
    """

    ious = []

    for cls in range(num_classes):

        pred_inds = preds == cls
        target_inds = targets == cls

        intersection = (pred_inds & target_inds).sum().float()
        union = pred_inds.sum().float() + target_inds.sum().float() - intersection

        if union == 0:
            ious.append(1.0)
        else:
            ious.append((intersection / union).item())

    return sum(ious) / len(ious)