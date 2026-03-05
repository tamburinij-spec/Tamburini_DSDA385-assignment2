"""Wrapper around torchvision's Faster R-CNN model."""

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights


def create_faster_rcnn(
    num_classes: int,
    pretrained: bool = False,
    device: str = "cpu",
) -> torch.nn.Module:
    """Return a Faster R-CNN model with the specified number of classes."""

    # Select weights properly (new API)
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        weights_backbone = ResNet50_Weights.DEFAULT
    else:
        weights = None
        weights_backbone = None

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )

    return model.to(device)