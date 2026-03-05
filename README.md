# DSDA385 – Assignment 2: Object Detection on Pet Dataset

**Author:** Jacopo Tamburini  
**Date:** March 2026  
**GPU:** NVIDIA RTX 4060

---

## 1. Introduction

Object detection is a fundamental computer vision task, aiming to locate and classify objects in images. Modern deep learning models, such as Faster R-CNN and YOLO, have shown high accuracy for detection tasks.  

This assignment explores implementing, training, and evaluating multiple object detection models on a subset of the Oxford-IIIT Pet dataset. Specifically, the project trains and compares:

- **Faster R-CNN** (custom PyTorch implementation)  
- **YOLOv5n** (Ultralytics)  
- **YOLOv8n** (Ultralytics)

The goal is to evaluate model performance in terms of precision, recall, F1-score, mean IoU, and mAP, and to analyze the strengths and weaknesses of each approach.

---

## 2. Dataset Description

### Oxford-IIIT Pet Dataset (Subset)

The dataset contains images of cats and dogs across multiple breeds. For this assignment, a small subset of 8 breeds was used for feasibility on a single GPU.

**Characteristics:**

- Input: RGB images, 3 channels  
- Number of classes: 8 pet breeds + background = 9  
- Dataset split:
  - 70% Training
  - 15% Validation
  - 15% Test

**Preprocessing:**

- Images and masks resized for training  
- Masks used to extract bounding boxes for YOLO training  
- Converted to YOLO format (images + labels) for Ultralytics

---

## 3. Model Architectures

### 3.1 Faster R-CNN

Faster R-CNN is a two-stage detector:

1. **Backbone:** ResNet-50 + FPN extracts image features.  
2. **Region Proposal Network (RPN):** Suggests candidate object regions.  
3. **Detection Head:** Classifies objects and refines bounding boxes.

Transfer learning is used with COCO-pretrained weights.  

### 3.2 YOLOv5n & YOLOv8n

YOLO is a single-stage detector:

- **YOLOv5n:** Nano version optimized for fast training.  
- **YOLOv8n:** Latest Ultralytics version, single-stage detection with higher efficiency.  

Both models are trained using the same dataset split and bounding boxes extracted from masks.

---

## 4. Training Details

### 4.1 Faster R-CNN

| Parameter          | Value |
|------------------|-------|
| Batch size        | 4     |
| Learning rate     | 0.001 |
| Epochs            | 15    |
| Weight decay      | 1e-5  |
| Optimizer         | Adam  |
| Device            | CUDA  |

Training uses BCEWithLogitsLoss for segmentation masks, and Faster R-CNN internal losses for detection.

### 4.2 YOLOv5n / YOLOv8n

| Parameter        | Value |
|-----------------|-------|
| Batch size       | 4     |
| Image size       | 640   |
| Epochs           | 15    |
| Device           | CUDA  |
| Pretrained model | yolov5n.pt / yolov8n.pt |

YOLO automatically handles loss computation, bounding box regression, and training visualization.

---

## 5. Running the Trainings

### 5.1 Faster R-CNN

```bash
python run_training.py
-   Saves checkpoints to: `outputs/checkpoints/best_model.pt`

-   Predictions & metrics in: `outputs/predictions/`

-   Includes `summary.json`, `training_log.csv`, `REPORT.txt`
```

### 5.2 YOLO

1.  Convert masks to YOLO bounding boxes:

python src/datasets/pets_to_yolo.py

2.  Train YOLOv5n:

python src/models/train_yolov5.py

3.  Train YOLOv8n:

python src/models/train_yolov8.py

4.  Compare all models:

python src/engine/compare_models.py

-   Saves comparison table: `outputs/model_comparison.csv`

---

## 6. Results
-----------

### 6.1 Faster R-CNN Test Metrics

| Metric | Value |
| --- | --- |
| Precision | 0.026 |
| Recall | 0.942 |
| F1-score | 0.051 |
| Mean IoU | 0.661 |

### 6.2 YOLO Metrics

> Automatically collected from Ultralytics `results.csv`

| Model | Precision | Recall | F1-score | Mean IoU | mAP@0.5 |
| --- | --- | --- | --- | --- | --- |
| Faster R-CNN | 0.026 | 0.942 | 0.051 | 0.661 | 0.661 |
| YOLOv5n | 0.99944 | 0.995 | 0.99722 | 0.995 | 0.995 |
| YOLOv8n | 0.99957 | 1 | 0.99728 | 0.995 | 0.995 |

> `[auto]` values are automatically saved in `outputs/model_comparison.csv`.

* * * * *

### 6.3 Example Predictions

Prediction visualizations are saved in `outputs/predictions/`. Each image shows:

-   Predicted bounding boxes

-   Class labels

-   Confidence scores

**Observation:** Faster R-CNN tends to generate many overlapping boxes → high recall but low precision. YOLOv5n/YOLOv8n produce fewer false positives.

* * * * *

## 7. Discussion
--------------

-   **Faster R-CNN:** Accurate localization (high IoU), low precision due to multiple predictions per object.

-   **YOLOv5n:** Faster training, better precision/recall balance.

-   **YOLOv8n:** Slightly better accuracy, modern architecture benefits.

**Considerations for small datasets:**

-   Transfer learning is crucial for convergence.

-   Small batch sizes are required to avoid GPU memory issues.

-   Non-Maximum Suppression (NMS) and confidence threshold adjustments improve precision.

-   Visual inspection complements numeric metrics for detection evaluation.

* * * * *

8\. Conclusion
--------------

The project demonstrates training and evaluation of modern object detection models on a small dataset:

-   Faster R-CNN successfully detects most objects but with many false positives.

-   YOLO models provide competitive speed and accuracy.

-   Comparison across models highlights trade-offs between detection accuracy, precision, and computational efficiency.

All results, metrics, and visualizations are included in `outputs/` for submission.

* * * * *

## 9. Folder Structure for Submission
-----------------------------------

DSDA385-assignment-2/\
│\
├── run_training.py\
├── README.md / REPORT.md\
├── src/\
│   ├── main.py\
│   ├── config.py\
│   ├── datasets/\
│   │   ├── pets.py\
│   │   ├── pennfudan.py\
│   │   └── pets_to_yolo.py\
│   ├── models/\
│   │   ├── faster_rcnn.py\
│   │   ├── train_yolov5.py\
│   │   └── train_yolov8.py\
│   └── engine/\
│       ├── train.py\
│       ├── inference.py\
│       ├── visualization.py\
│       ├── reporter.py\
│       └── compare_models.py\
├── outputs/\
│   ├── checkpoints/\
│   │   best_model.pt\
│   ├── predictions/\
│   │   summary.json\
│   │   REPORT.txt\
│   │   training_log.csv\
│   └── model_comparison.csv\
├── data/\
│   ├── pets_subset/\
│   └── yolo/\
└── yolo_dataset.yaml

* * * * *

## 10. References
---------------

-   Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.*

-   Ultralytics YOLO Documentation: <https://docs.ultralytics.com>

-   PyTorch & Torchvision Documentation: <https://pytorch.org>
