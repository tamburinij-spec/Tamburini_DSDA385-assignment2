"""
Inference and evaluation utilities for object detection models.
"""

import torch
from tqdm import tqdm


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in format [x1, y1, x2, y2]
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea

    if union <= 0:
        return 0.0

    return float(interArea / union)


def run_test_evaluation(model, test_loader, device, output_dir):
    """
    Run inference on the test dataset and compute detection metrics.
    """

    model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0

    iou_scores = []

    with torch.no_grad():

        for images, targets in tqdm(test_loader, desc="Test"):

            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, tgt in zip(outputs, targets):

                pred_boxes = pred.get("boxes", []).detach().cpu()
                tgt_boxes = tgt.get("boxes", []).detach().cpu()

                matched_targets = set()

                for pbox in pred_boxes:

                    best_iou = 0
                    best_target_idx = -1

                    for idx, tbox in enumerate(tgt_boxes):

                        if idx in matched_targets:
                            continue

                        iou = compute_iou(pbox, tbox)

                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = idx

                    if best_iou >= 0.5:
                        true_positive += 1
                        matched_targets.add(best_target_idx)
                        iou_scores.append(best_iou)
                    else:
                        false_positive += 1

                false_negative += len(tgt_boxes) - len(matched_targets)

    # ---- Compute final metrics ----

    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    mean_iou = sum(iou_scores) / len(iou_scores) if len(iou_scores) > 0 else 0.0

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1-score": float(f1),
        "mean_IoU": float(mean_iou),
    }

    return metrics