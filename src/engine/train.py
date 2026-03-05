"""Training loop utilities."""

import torch
from tqdm import tqdm
import torchvision
from pathlib import Path
import json
from datetime import datetime


def _masks_to_detection_targets(masks_batch):
    """
    Convert batch of segmentation masks to Faster R-CNN targets.
    Accepts a tuple/list of masks.
    """

    targets = []

    for masks in masks_batch:
        # Ensure masks is 3D (N,H,W) or 2D (H,W)
        if masks.ndim == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)

        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        boxes = []
        labels = []

        for mask in masks:
            pos = mask.nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue

            xmin = pos[:, 1].min()
            xmax = pos[:, 1].max()
            ymin = pos[:, 0].min()
            ymax = pos[:, 0].max()

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # assuming single-class object

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        targets.append({
            "boxes": boxes,
            "labels": labels,
        })

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
        for images, second in progress_bar:
            images = [img.to(self.device) for img in images]
            if isinstance(second, torch.Tensor):
                masks = second.to(self.device)
                targets = _masks_to_detection_targets(masks)

                for t in targets:
                    t["boxes"] = t["boxes"].to(self.device)
                    t["labels"] = t["labels"].to(self.device)

                loss_dict = self.model(images, targets)
                loss = sum(v for v in loss_dict.values())

            else:
                targets = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in second
                ]

                loss_dict = self.model(images, targets)
                loss = sum(v for v in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / len(train_loader)
        return epoch_loss

    def evaluate(self, val_loader):
        self.model.train()   # 🔥 Important for detection loss

        total_loss = 0.0
        batch_count = 0

        for images, targets in val_loader:
            images = [img.to(self.device) for img in images]

            for t in targets:
                t["boxes"] = t["boxes"].to(self.device)
                t["labels"] = t["labels"].to(self.device)

            with torch.no_grad():
                loss_dict = self.model(images, targets)
                loss = sum(v for v in loss_dict.values())

            total_loss += loss.item()
            batch_count += 1

        epoch_loss = total_loss / batch_count

        self.model.eval()  # switch back

        return epoch_loss

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
