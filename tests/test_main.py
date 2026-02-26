import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.faster_rcnn import create_faster_rcnn
from config import DEVICE, MODEL_CONFIG

def test_model_creation():
    """Test that model can be created and run inference"""
    print("Testing model creation...")
    model = create_faster_rcnn(num_classes=MODEL_CONFIG['num_classes'], device=DEVICE)
    model.eval()
    
    # Forward pass on a single-image batch (Faster R-CNN expects a list)
    dummy_input = [torch.randn(3, 224, 224).to(DEVICE)]
    with torch.no_grad():
        output = model(dummy_input)
    
    assert isinstance(output, list), "Faster R-CNN should return a list of detections"
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