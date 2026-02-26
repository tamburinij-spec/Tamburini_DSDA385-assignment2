"""Wrapper around torchvision's Faster R-CNN model."""

import torchvision
import torch


def create_faster_rcnn(num_classes: int, pretrained: bool = False, device: str = "cpu") -> torch.nn.Module:
    """Return a Faster R-CNN model with the specified number of classes.

    The callable plays nicely with configuration dictionaries defined in
    ``config/faster_rcnn.yaml``.
    """
    # load pre-trained base model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained,
        num_classes=num_classes,
    )
    return model.to(device)
