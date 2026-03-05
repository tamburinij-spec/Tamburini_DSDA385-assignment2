import random
import json
from pathlib import Path
import numpy as np
from torchvision.datasets import OxfordIIITPet
from PIL import Image

OUTPUT_ROOT = Path("data/pets_subset")
BREEDS_TO_USE = 8
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

def mask_to_box(mask):
    pos = np.where(mask > 0)
    xmin = int(np.min(pos[1]))
    xmax = int(np.max(pos[1]))
    ymin = int(np.min(pos[0]))
    ymax = int(np.max(pos[0]))
    return [xmin, ymin, xmax, ymax]

def save_sample(split, idx, image, box, label):
    img_path = OUTPUT_ROOT / split / "images" / f"{idx}.jpg"
    ann_path = OUTPUT_ROOT / split / "annotations" / f"{idx}.json"

    image.save(img_path)

    ann = {
        "boxes": [box],
        "labels": [label]
    }

    with open(ann_path, "w") as f:
        json.dump(ann, f)

def main():
    dataset = OxfordIIITPet(
        root="data",
        split="trainval",
        target_types="segmentation",
        download=True
    )

    all_breeds = sorted(set(dataset._labels))
    selected_breeds = all_breeds[:BREEDS_TO_USE]

    indices = [
        i for i, label in enumerate(dataset._labels)
        if label in selected_breeds
    ]

    random.shuffle(indices)

    n = len(indices)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    for split, idx_list in splits.items():
        print(f"Processing {split}...")

        for count, i in enumerate(idx_list):
            image, mask = dataset[i]
            mask = np.array(mask)

            box = mask_to_box(mask)
            label = int(dataset._labels[i]) + 1

            save_sample(split, f"{split}_{count}", image, box, label)

    print("Pet subset dataset ready!")

if __name__ == "__main__":
    main()