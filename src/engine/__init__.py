"""Engine package containing training and evaluation logic."""

from .train import Trainer
from .evaluate import evaluate_segmentation
from .metrics import iou_score

__all__ = ["Trainer", "evaluate_segmentation", "iou_score"]
