"""
Train YOLOv8 on the pet dataset.
"""

from ultralytics import YOLO


def train():

    model = YOLO("yolov8n.pt")  # nano version

    model.train(
        data="yolo_dataset.yaml",
        epochs=15,
        imgsz=640,
        batch=4,
        device=0
        verbose=True
    )


if __name__ == "__main__":
    train()