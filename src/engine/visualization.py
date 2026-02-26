"""Visualization utilities for prediction results."""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def draw_boxes_on_image(image_np, boxes, scores=None, labels=None, thickness=2, font_scale=0.5):
    """Draw bounding boxes on an image.
    
    Args:
        image_np: Image as numpy array (H, W, 3) in RGB
        boxes: Array of shape (N, 4) in format [x1, y1, x2, y2]
        scores: Array of shape (N,) with confidence scores
        labels: Array of shape (N,) with class labels
        thickness: Box line thickness
        font_scale: Font scale for text
    
    Returns:
        Image with boxes drawn
    """
    image = image_np.copy()
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Color for boxes
    color = (0, 255, 0)  # Green
    text_color = (255, 255, 255)  # White
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(c) for c in box]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw score and label if provided
        label_text = ""
        if scores is not None and i < len(scores):
            label_text += f"Score: {scores[i]:.2f}"
        if labels is not None and i < len(labels):
            label_text += f" Label: {int(labels[i])}"
        
        if label_text:
            # Put text with background
            cv2.putText(image, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    return image


def save_predictions_visualization(model, test_loader, device, output_dir='outputs/predictions'):
    """Generate and save visualizations of predictions on test images.
    
    Args:
        model: Trained detection model
        test_loader: Test data loader
        device: torch device
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    img_count = 0
    
    print("Generating prediction visualizations...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Visualizing"):
            batch_images = images.to(device)
            predictions = model(batch_images)
            
            # Convert images back to numpy
            images_np = images.cpu().numpy()
            
            for img_idx, pred in enumerate(predictions):
                if img_idx >= len(images_np):
                    break
                
                # Get image in HWC format (0-1 range)
                img = images_np[img_idx]  # Shape: (3, H, W)
                
                # Convert from CHW to HWC and scale to 0-255
                img = np.transpose(img, (1, 2, 0))
                img = (img * 255).astype(np.uint8)
                
                # Denormalize (ImageNet stats)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img / 255.0 - mean) / std
                img = np.clip((img * 255).astype(np.uint8), 0, 255)
                
                # Draw predictions
                if len(pred['boxes']) > 0:
                    boxes = pred['boxes'].cpu().numpy()
                    scores = pred['scores'].cpu().numpy()
                    labels = pred['labels'].cpu().numpy() if 'labels' in pred else None
                    
                    img_with_boxes = draw_boxes_on_image(img, boxes, scores, labels)
                else:
                    img_with_boxes = img
                
                # Save visualization
                save_path = output_dir / f'pred_{img_count:05d}.jpg'
                cv2.imwrite(str(save_path), img_with_boxes)
                img_count += 1
    
    print(f"✓ Saved {img_count} prediction visualizations to {output_dir}")


def save_comparison_image(image_np, ground_truth_boxes, predicted_boxes, predicted_scores=None, 
                         output_path=None):
    """Save a side-by-side comparison of ground truth and predictions.
    
    Args:
        image_np: Input image
        ground_truth_boxes: Ground truth boxes
        predicted_boxes: Predicted boxes
        predicted_scores: Prediction scores
        output_path: Path to save the comparison image
    """
    # Draw ground truth in red
    img_gt = draw_boxes_on_image(image_np, ground_truth_boxes, thickness=2)
    if len(img_gt.shape) == 3 and img_gt.shape[2] == 3:
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
    
    # Modify to draw in red
    for box in ground_truth_boxes:
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
        cv2.putText(img_gt, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw predictions in green
    img_pred = draw_boxes_on_image(image_np, predicted_boxes, predicted_scores, thickness=2)
    if len(img_pred.shape) == 3 and img_pred.shape[2] == 3:
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
    
    # Concatenate horizontally
    comparison = np.hstack([img_gt, img_pred])
    
    if output_path:
        cv2.imwrite(str(output_path), comparison)
    
    return comparison
