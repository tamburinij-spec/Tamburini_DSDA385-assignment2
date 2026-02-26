# DSDA385 Assignment 2: Object Detection

## Overview

This project implements full **pedestrian detection and segmentation** using:
- **Model**: Faster R-CNN (ResNet50 backbone) for object detection
- **Dataset**: PennFudan Pedestrian Detection Dataset
- **Hardware**: GPU-optimized for 8GB VRAM
- **Framework**: PyTorch + Torchvision

## What's Been Implemented

✅ **Complete Training Pipeline**
- Modular architecture with separate config, data, models, and engine packages
- YAML-based configuration system
- Automatic checkpoint saving (best and latest models)
- Metrics tracking and logging

✅ **Data Handling**
- PennFudan dataset loader with train/val/test splits
- Automatic conversion of binary masks to bounding box targets
- Image normalization and augmentation

✅ **Model Training**
- Faster R-CNN with pretrained ResNet50 backbone
- Duck-typing support for both detection and segmentation models
- Batch size optimization (reduced from 32→4 for 8GB GPU)
- 15-epoch training per assignment specs
- Progress bars and real-time loss tracking

✅ **Evaluation & Metrics**
- Precision, Recall, F1-Score computation
- IoU (Intersection over Union) calculation
- Test set evaluation
- Prediction filtering by confidence threshold

✅ **Visualization & Results**
- Bounding box visualization on test images
- Predictions saved with scores and labels
- Comprehensive metrics reporting
- Training logs as CSV files

## Project Structure

```
DSDA385-assignment-2/
├── config/                      # Configuration system
│   ├── __init__.py             # YAML config loader
│   ├── dataset.yaml            # Training hyperparameters
│   ├── faster_rcnn.yaml        # Model config
│   └── yolo.yaml               # YOLO placeholder
├── data/
│   └── pennfudan/              # PennFudan dataset
│       ├── train/              # 130 train images
│       ├── val/                # ~20 val images
│       └── test/               # ~10 test images
├── src/
│   ├── datasets/
│   │   ├── pennfudan.py       # Dataset loader
│   │   └── pets.py            # Placeholder
│   ├── models/
│   │   ├── faster_rcnn.py     # Faster R-CNN wrapper
│   │   └── yolo_wrapper.py    # YOLO placeholder
│   ├── engine/
│   │   ├── train.py           # Trainer class with checkpointing
│   │   ├── inference.py       # Test evaluation
│   │   ├── visualization.py   # Prediction visualization
│   │   └── reporter.py        # Report generation
│   └── main.py                # Main training script
├── outputs/
│   ├── checkpoints/           # Model saves
│   │   ├── best_model.pt
│   │   ├── latest_model.pt
│   │   └── metrics_*.json
│   └── predictions/           # Test results
│       ├── REPORT.txt
│       ├── summary.json
│       ├── training_log.csv
│       ├── test_results.json
│       └── pred_*.jpg          # Visualized predictions
└── README.md
```

## How to Run

### 1. **Install Dependencies**
```bash
pip install torch torchvision
# or if issues with numpy/pandas, use:
pip install --only-binary :all: torch torchvision
```

### 2. **Run Training & Evaluation**
```bash
cd src
python main.py
```

This will:
1. Load the PennFudan dataset (train/val/test splits)
2. Train Faster R-CNN for 15 epochs
3. Save best model checkpoint
4. Evaluate on test set
5. Generate prediction visualizations
6. Save results and metrics

### 3. **Check Results**
Results are saved to `outputs/`:
- **`outputs/checkpoints/`** - Trained models
  - `best_model.pt` - Best model by validation loss
  - `latest_model.pt` - Latest epoch model
  - `metrics_*.json` - Epoch-by-epoch metrics

- **`outputs/predictions/`** - Test evaluation results
  - `REPORT.txt` - Summary report
  - `summary.json` - Complete configuration and metrics
  - `training_log.csv` - Epoch-by-epoch training log
  - `test_results.json` - Test set evaluation metrics
  - `pred_*.jpg` - Visualized predictions with bounding boxes

## Training Configuration

Edit `config/dataset.yaml` to adjust:
```yaml
batch_size: 4              # Optimized for 8GB GPU
num_epochs: 15             # Per assignment spec
learning_rate: 0.001
weight_decay: 1e-5
```

## Model Configuration

The model can be customized in `config/faster_rcnn.yaml`:
```yaml
model_name: faster_rcnn
backbone: resnet50
num_classes: 10            # Pedestrian class
pretrained: false          # Use random initialization
```

## Output Files & Metrics

### Test Metrics (`test_results.json`)
```json
{
  "metrics": {
    "TP": 45,              # True Positives
    "FP": 5,               # False Positives
    "FN": 3,               # False Negatives
    "Precision": 0.9,      # TP / (TP + FP)
    "Recall": 0.9375,      # TP / (TP + FN)
    "F1-Score": 0.9184,    # Harmonic mean
    "Mean_IoU": 0.8234     # Average box overlap
  }
}
```

### Training Log (`training_log.csv`)
```
epoch,train_loss,val_loss
0,2.3456,1.8234
1,1.9876,1.6543
...
```

### Predictions
- `pred_*.jpg` - Test images with bounding boxes drawn
  - Green boxes with confidence scores
  - Saved for visual inspection

## Key Features

✨ **Modular Design**
- Separate packages for config, data, models, engine
- Easy to extend with new models/datasets
- Clean separation of concerns

🚀 **Production-Ready**
- Checkpoint saving and restoration
- Comprehensive metrics computation
- Result visualization and reporting
- Error handling and validation

📊 **Full Tracking**
- Epoch-by-epoch loss tracking
- Test set evaluation metrics
- Prediction visualizations
- JSON schema for all results

## Troubleshooting

**GPU Memory Error?**
→ Reduce batch_size in `config/dataset.yaml` (e.g., 4→2)

**Missing data?**
→ Ensure PennFudan data is in `data/pennfudan/` with subdirectories:
   - `train/images/` and `train/masks/`
   - `val/images/` and `val/masks/`
   - `test/images/` and `test/masks/`

**Import errors?**
→ Make sure you're running from `src/` directory:
   ```bash
   cd src
   python main.py
   ```

## Report Requirements

For the assignment report (4-6 pages), include:

1. **Introduction** - Problem statement and approach
2. **Dataset** - PennFudan description, splits, sample images
3. **Model Architecture** - Faster R-CNN components
4. **Experimental Setup** - Hyperparameters, hardware, training details
5. **Results** - Training curves, test metrics, visualizations
6. **Analysis** - Strengths, weaknesses, failure cases
7. **Conclusion** - Key insights and future work

All content is in `outputs/predictions/` for your report.

## References

- PennFudan Dataset: https://www.cis.upenn.edu/~jshi/ped_html/
- Faster R-CNN Paper: https://arxiv.org/abs/1506.01497
- PyTorch Docs: https://pytorch.org/docs/
