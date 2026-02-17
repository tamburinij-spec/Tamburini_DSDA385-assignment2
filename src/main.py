import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, TRAINING_CONFIG, MODEL_CONFIG
from models.model import create_model
from data.dataset import get_data_loaders
from utils.trainer import Trainer

def main():
    print(f"Using device: {DEVICE}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Create model
    print("Creating segmentation model...")
    model = create_model(
        num_classes=1,
        device=DEVICE,
        model_type='segmentation'
    )
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        num_epochs=TRAINING_CONFIG['num_epochs']
    )
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    print("✓ Training complete!")

if __name__ == "__main__":
    main()