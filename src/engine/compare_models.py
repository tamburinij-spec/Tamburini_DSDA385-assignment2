"""
Compare Faster R-CNN, YOLOv5n and YOLOv8n results.

This script reads metrics from:
- Faster R-CNN summary.json
- YOLO results.csv (Ultralytics output)

It creates a comparison table and saves it.
"""

import json
import pandas as pd
from pathlib import Path


def load_faster_rcnn_metrics():
    """Load metrics from Faster R-CNN summary file"""

    summary_path = Path("outputs/predictions/summary.json")

    if not summary_path.exists():
        print("Faster R-CNN summary.json not found")
        return None

    with open(summary_path) as f:
        data = json.load(f)

    m = data["test_metrics"]

    return {
        "Model": "Faster R-CNN",
        "Precision": m["precision"],
        "Recall": m["recall"],
        "F1-Score": m["f1-score"],
        "Mean IoU": m["mean_IoU"],
        "mAP50": None
    }


def load_yolo_metrics(results_path, model_name):
    """Load metrics from YOLO results.csv"""

    results_path = Path(results_path)

    if not results_path.exists():
        print(f"{model_name} results not found")
        return None

    df = pd.read_csv(results_path)

    last = df.iloc[-1]

    return {
        "Model": model_name,
        "Precision": last.get("metrics/precision(B)", None),
        "Recall": last.get("metrics/recall(B)", None),
        "F1-Score": None,
        "Mean IoU": None,
        "mAP50": last.get("metrics/mAP50(B)", None)
    }


def main():

    rows = []

    # Faster R-CNN
    frcnn = load_faster_rcnn_metrics()
    if frcnn:
        rows.append(frcnn)

    # YOLOv5
    yolo5 = load_yolo_metrics(
        "runs/detect/train/results.csv",
        "YOLOv5n"
    )

    if yolo5:
        rows.append(yolo5)

    # YOLOv8
    yolo8 = load_yolo_metrics(
        "runs/detect/train2/results.csv",
        "YOLOv8n"
    )

    if yolo8:
        rows.append(yolo8)

    df = pd.DataFrame(rows)

    output_path = Path("outputs/model_comparison.csv")
    output_path.parent.mkdir(exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nModel Comparison Table\n")
    print(df.to_string(index=False))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()