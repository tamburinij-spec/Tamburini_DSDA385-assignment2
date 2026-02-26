"""Inference and evaluation utilities for detection models."""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.ops import nms
import json


def get_detections(model, images, device, conf_threshold=0.5):
    """Run inference on images and return detections.
    
    Args:
        model: Detection model
        images: Batch of images (B, 3, H, W)
        device: torch device
        conf_threshold: Confidence threshold for filtering detections
    
    Returns:
        List of dicts with 'boxes' and 'scores' for each image
    """
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    
    detections = []
    for pred in predictions:
        # Filter by confidence threshold
        if len(pred['scores']) > 0:
            keep = pred['scores'] >= conf_threshold
            filtered_det = {
                'boxes': pred['boxes'][keep].cpu().numpy(),
                'scores': pred['scores'][keep].cpu().numpy(),
                'labels': pred['labels'][keep].cpu().numpy() if 'labels' in pred else np.ones(keep.sum())
            }
        else:
            filtered_det = {
                'boxes': np.array([]).reshape(0, 4),
                'scores': np.array([]),
                'labels': np.array([])
            }
        detections.append(filtered_det)
    
    return detections


def compute_iou(box1, box2):
    """Compute IoU between two boxes in format [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def evaluate_detections(detections_list, targets_list, iou_threshold=0.5):
    """Evaluate detections against ground truth targets.
    
    Args:
        detections_list: List of detection dicts with 'boxes', 'scores'
        targets_list: List of target dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for TP/FP classification
    
    Returns:
        Dict with evaluation metrics
    """
    tp = 0
    fp = 0
    fn = 0
    
    ious = []
    
    for dets, targets in zip(detections_list, targets_list):
        # Convert targets to numpy if needed
        target_boxes = targets['boxes']
        if isinstance(target_boxes, torch.Tensor):
            target_boxes = target_boxes.cpu().numpy()
        
        det_boxes = dets['boxes']
        det_scores = dets['scores']
        
        # Sort detections by score
        if len(det_boxes) > 0:
            sorted_idx = np.argsort(-det_scores)
            det_boxes = det_boxes[sorted_idx]
        
        matched_targets = set()
        
        for det_box in det_boxes:
            best_iou = 0
            best_target_idx = -1
            
            for target_idx, target_box in enumerate(target_boxes):
                if target_idx in matched_targets:
                    continue
                iou = compute_iou(det_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx
            
            if best_iou >= iou_threshold and best_target_idx >= 0:
                tp += 1
                matched_targets.add(best_target_idx)
                ious.append(best_iou)
            else:
                fp += 1
        
        # Unmatched targets are false negatives
        fn += len(target_boxes) - len(matched_targets)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(ious) if len(ious) > 0 else 0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Mean_IoU': mean_iou
    }


def run_test_evaluation(model, test_loader, device, output_dir='outputs/predictions'):
    """Run evaluation on test set and save results.
    
    Args:
        model: Trained detection model
        test_loader: Test data loader
        device: torch device
        output_dir: Directory to save predictions
    
    Returns:
        Dict with evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_detections = []
    all_targets = []
    
    print("Running test evaluation...")
    for images, masks in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        
        # Get predictions from model
        model.eval()
        with torch.no_grad():
            predictions = model(images)
        
        # Convert predictions to detections
        for pred in predictions:
            det = {
                'boxes': pred['boxes'].cpu().numpy(),
                'scores': pred['scores'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy() if 'labels' in pred else np.ones(len(pred['boxes']))
            }
            all_detections.append(det)
        
        # Convert masks to targets
        from engine.train import _masks_to_detection_targets
        targets = _masks_to_detection_targets(masks)
        for target in targets:
            gt = {
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy()
            }
            all_targets.append(gt)
    
    # Evaluate
    metrics = evaluate_detections(all_detections, all_targets)
    
    # Save results
    results = {
        'metrics': metrics,
        'num_test_images': len(all_detections)
    }
    
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Test Results:")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-Score: {metrics['F1-Score']:.4f}")
    print(f"  Mean IoU: {metrics['Mean_IoU']:.4f}")
    print(f"  Results saved to {results_path}")
    
    return metrics
