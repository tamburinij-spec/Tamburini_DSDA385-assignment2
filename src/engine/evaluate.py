"""Evaluation routines for models."""

import torch


def evaluate_segmentation(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            total_loss += loss.item()
    return total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
