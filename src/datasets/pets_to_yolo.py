# src/datasets/pets_to_yolo.py
import os
import shutil
from pathlib import Path

def convert_split(split):
    in_images = Path(f"data/pets_subset/{split}/images")
    in_masks = Path(f"data/pets_subset/{split}/masks")
    out_images = Path(f"data/yolo/{split}/images")
    out_labels = Path(f"data/yolo/{split}/labels")

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_file in in_images.glob("*.jpg"):
        # copy image
        shutil.copy(img_file, out_images / img_file.name)

        # create dummy label file with class 0 and full-image bbox (as a placeholder)
        label_file = out_labels / img_file.name.replace(".jpg", ".txt")
        # placeholder bbox (entire image, normalized 0–1)
        label_file.write_text(f"0 0.5 0.5 1.0 1.0\n")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)
    print("YOLO dataset folders populated.")