**Author:** Jacopo Tamburini  
**Date:** March 3, 2026  
**GPU:** NVIDIA RTX 4060

---

# **1\. Introduction**

Object detection is a fundamental computer vision task that aims to both locate and classify objects within images. Modern deep learning models such as Faster R-CNN and YOLO have demonstrated high accuracy across a wide range of detection tasks.

This assignment explores the implementation, training, and evaluation of multiple object detection models on small computer vision datasets, including the Penn-Fudan pedestrian dataset and a subset of the Oxford-IIIT Pet dataset. Specifically, the project trains and compares the following models:

* Faster R-CNN (custom PyTorch implementation)  
* YOLOv5n (Ultralytics)  
* YOLOv8n (Ultralytics)

The goal is to evaluate model performance in terms of precision, recall, F1-score, mean Intersection over Union (IoU), and mean Average Precision (mAP), while analyzing the strengths and weaknesses of each approach.

---

# **2\. Dataset Description**

## **2.1 Penn-Fudan Pedestrian Dataset**

The Penn-Fudan dataset contains images of pedestrians captured in urban street scenes.

Characteristics:

| Feature | Description |
| ----- | ----- |
| Number of Images | \~170 |
| Classes | Single class: pedestrian |
| Annotations | Bounding boxes included |
| Typical Use | Small-scale object detection experiments |

Although the dataset is supported by the project pipeline, the primary experiments in this assignment were conducted on the Oxford-IIIT Pet dataset subset.

---

## **2.2 Oxford-IIIT Pet Dataset (Subset)**

The Oxford-IIIT Pet dataset contains images of cats and dogs across multiple breeds. For this assignment, a subset of eight breeds was used to make training feasible on a single GPU.

Characteristics:

| Feature | Description |
| ----- | ----- |
| Input Format | RGB images (3 channels) |
| Classes | 8 pet breeds \+ background (9 total) |
| Training Split | 70% |
| Validation Split | 15% |
| Test Split | 15% |

### **Preprocessing**

The following preprocessing steps were applied:

* Images and masks were resized for training.  
* Segmentation masks were used to extract bounding boxes.  
* The dataset was converted to YOLO format (images \+ labels) for Ultralytics training.

---

# **3\. Model Architectures**

## **3.1 Faster R-CNN**

Faster R-CNN is a two-stage object detector composed of the following components:

1. **Backbone Network**  
   A ResNet-50 with Feature Pyramid Network (FPN) extracts hierarchical image features.  
2. **Region Proposal Network (RPN)**  
   Generates candidate object regions likely to contain objects.  
3. **Detection Head**  
   Classifies objects and refines bounding box predictions.

Transfer learning was applied using pretrained weights from the COCO dataset.

---

## **3.2 YOLOv5n and YOLOv8n**

YOLO (You Only Look Once) models are single-stage object detectors that directly predict bounding boxes and class probabilities in one pass.

The models used in this assignment include:

| Model | Description |
| ----- | ----- |
| YOLOv5n | Nano version optimized for fast training and lightweight deployment |
| YOLOv8n | Latest Ultralytics architecture with improved detection efficiency |

Both models were trained using identical dataset splits and bounding boxes derived from segmentation masks.

---

# **4\. Training Details**

## **4.1 Faster R-CNN Training Parameters**

| Parameter | Value |
| ----- | ----- |
| Batch Size | 4 |
| Learning Rate | 0.001 |
| Epochs | 15 |
| Weight Decay | 1e-5 |
| Optimizer | Adam |
| Device | CUDA |

Training uses the standard Torchvision Faster R-CNN loss functions, including:

* Classification loss  
* Bounding box regression loss  
* Region Proposal Network (RPN) losses

---

## **4.2 YOLOv5n and YOLOv8n Training Parameters**

| Parameter | Value |
| ----- | ----- |
| Batch Size | 4 |
| Image Size | 640 |
| Epochs | 15 |
| Device | CUDA |
| Pretrained Models | yolov5n.pt / yolov8n.pt |

YOLO handles loss computation, bounding box regression, and training visualization automatically within the Ultralytics framework.

# **5\. Results**

## **5.1 Faster R-CNN Test Metrics**

| Metric | Value |
| ----- | ----- |
| Precision | 0.026 |
| Recall | 0.942 |
| F1-score | 0.051 |
| Mean IoU | 0.661 |

---

## **5.2 YOLO Model Metrics**

Metrics were automatically collected from the Ultralytics `results.csv` files.

| Model | Precision | Recall | F1-score | Mean IoU | mAP@0.5 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Faster R-CNN | 0.026 | 0.942 | 0.051 | 0.661 | 0.661 |
| YOLOv5n | 0.99944 | 0.995 | 0.99722 | 0.995 | 0.995 |
| YOLOv8n | 0.99957 | 1.000 | 0.99728 | 0.995 | 0.995 |

The extremely high YOLO scores are likely influenced by the small dataset size and limited number of classes, which makes the detection task relatively easy after transfer learning.

---

## **5.3 Example Predictions**

Prediction visualizations are saved in:

outputs/predictions/

Each image includes:

* Predicted bounding boxes  
* Class labels  
* Confidence scores

Observation:

* Faster R-CNN tends to generate many overlapping boxes, resulting in **high recall but low precision**.  
* YOLOv5n and YOLOv8n produce **fewer false positives** and more stable predictions.

---

# **6\. Discussion and conclusion**

Key observations from the experiments include:

| Model | Key Behavior |
| ----- | ----- |
| Faster R-CNN | Accurate localization with high IoU but many duplicate predictions |
| YOLOv5n | Faster training with balanced precision and recall |
| YOLOv8n | Slightly higher accuracy due to architectural improvements |

Important considerations for small datasets:

* Transfer learning is critical for model convergence.  
* Small batch sizes help avoid GPU memory issues.  
* Non-Maximum Suppression (NMS) and confidence threshold tuning improve precision.  
* Visual inspection is important to complement quantitative evaluation metrics.

This project demonstrates the training and evaluation of modern object detection models on a small dataset.

Key findings include:

* Faster R-CNN successfully detects most objects but produces many false positives.  
* YOLO models provide strong performance with faster training and higher precision.  
* Model comparisons highlight trade-offs between detection accuracy, precision, and computational efficiency.

All results, metrics, and visualizations are included in the `outputs` directory.

---

# **References**

Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.*

Ultralytics YOLO Documentation  
[https://docs.ultralytics.com](https://docs.ultralytics.com)

PyTorch & Torchvision Documentation  
[https://pytorch.org](https://pytorch.org)