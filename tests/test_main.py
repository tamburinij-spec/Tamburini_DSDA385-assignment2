import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model import create_model
from src.config import DEVICE, MODEL_CONFIG

def test_model_creation():
    """Test that model can be created and run inference"""
    print("Testing model creation...")
    model = create_model(num_classes=MODEL_CONFIG['num_classes'], device=DEVICE)
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    
    # Forward pass
    output = model(dummy_input)
    
    assert output.shape == (2, MODEL_CONFIG['num_classes']), f"Expected output shape (2, {MODEL_CONFIG['num_classes']}), got {output.shape}"
    print("✓ Model creation test passed!")

def test_device():
    """Test device configuration"""
    print(f"Testing device: {DEVICE}")
    assert str(DEVICE) in ['cpu', 'cuda', 'cuda:0'], f"Invalid device: {DEVICE}"
    print(f"✓ Device test passed! Using: {DEVICE}")

if __name__ == "__main__":
    test_device()
    test_model_creation()
    print("\n✓ All tests passed!")