"""Training loop utilities."""

import torch
from tqdm import tqdm
import torchvision
from pathlib import Path
import json
from datetime import datetime


def _masks_to_detection_targets(masks: torch.Tensor):
    """Convert a batch of binary masks to Faster-RCNN target dicts.

    Each mask is expected to be a float tensor in [0,1] shape (H,W) or
    (1,H,W); we produce a single box around the foreground region and a
    label of 1.  This is a quick hack for the PennFudan pedestrian data
    where each image contains exactly one object.
    """
    targets = []
    # masks come in (B,H,W) after dataset returns; ensure correct shape
    if masks.ndim == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)

    for mask in masks:
        mask = (mask > 0.5).float()
        if mask.sum() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            xmin = xs.min().float()
            ymin = ys.min().float()
            xmax = xs.max().float()
            ymax = ys.max().float()
            boxes = torch.stack([xmin, ymin, xmax, ymax]).unsqueeze(0)
            labels = torch.ones((1,), dtype=torch.int64)
        targets.append({"boxes": boxes, "labels": labels})
    return targets


class Trainer:
    def __init__(self, model, optimizer, criterion, device, num_epochs=50, checkpoint_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('outputs/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  [DEBUG] Checkpoint directory: {self.checkpoint_dir.absolute()}")
        print(f"  [DEBUG] Directory exists: {self.checkpoint_dir.exists()}")
        
        self.best_val_loss = float('inf')
        self.metrics_history = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        is_first_batch = True

        progress_bar = tqdm(train_loader, desc="Training")
        for images, masks in progress_bar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Try detection model path (expects targets, returns dict)
            try:
                targets = _masks_to_detection_targets(masks)
                is_first_batch = False
                for t in targets:
                    t["boxes"] = t["boxes"].to(self.device)
                    t["labels"] = t["labels"].to(self.device)
                output = self.model(images, targets)
                # if output is dict, it's a detection model loss dict
                if isinstance(output, dict):
                    loss = sum(v for v in output.values())
                else:
                    # fallback: shouldn't happen but treat as segmentation
                    loss = self.criterion(output, masks.unsqueeze(1))
            except (TypeError, RuntimeError, AssertionError):
                # Model doesn't accept targets (segmentation model) or other errors
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / len(train_loader)
        return epoch_loss

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Try detection path first
                try:
                    targets = _masks_to_detection_targets(masks)
                    for t in targets:
                        t["boxes"] = t["boxes"].to(self.device)
                        t["labels"] = t["labels"].to(self.device)
                    output = self.model(images, targets)
                    # In eval mode, detection models return predictions list, not losses
                    if isinstance(output, dict):
                        loss = sum(v for v in output.values())
                        total_loss += loss.item()
                    # else: predictions list, skip (can't compute scalar loss)
                except (TypeError, RuntimeError, AssertionError):
                    # Fallback to segmentation path
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks.unsqueeze(1))
                    total_loss += loss.item()

                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        return avg_loss

    def _save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint if validation loss improves."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            # Ensure directory exists
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.checkpoint_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f"  ✓ Best checkpoint saved: {best_path} (val_loss: {val_loss:.4f})")
            
            # Save latest model
            latest_path = self.checkpoint_dir / 'latest_model.pt'
            torch.save(checkpoint, latest_path)
        except Exception as e:
            print(f"  ⚠ Error saving checkpoint: {e}")
    
    def _log_metrics(self, epoch, train_loss, val_loss):
        """Log training metrics."""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        self.metrics_history.append(metrics)
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = self.checkpoint_dir / f'metrics_{self.timestamp}.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            print(f"  ✓ Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"  ⚠ Error saving metrics: {e}")

    def train(self, train_loader, val_loader):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            print(f"Epoch [{epoch+1}/{self.num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            # Save checkpoint and log metrics
            self._save_checkpoint(epoch, train_loss, val_loss)
            self._log_metrics(epoch, train_loss, val_loss)
        
        # Save metrics at the end
        self.save_metrics()
        print(f"\n✓ Training complete! Best model saved to {self.checkpoint_dir / 'best_model.pt'}")
