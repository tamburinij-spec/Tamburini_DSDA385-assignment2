import json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PetDetectionDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.images = sorted((self.root / "images").glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        ann_path = self.root / "annotations" / f"{img_path.stem}.json"

        image = Image.open(img_path).convert("RGB")
        image = image.resize((512, 512))

        with open(ann_path) as f:
            ann = json.load(f)

        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.tensor(ann["labels"], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        image = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0

        return image, target


def get_pet_data_loaders(batch_size=2):
    train_set = PetDetectionDataset("data/pets_subset/train")
    val_set = PetDetectionDataset("data/pets_subset/val")
    test_set = PetDetectionDataset("data/pets_subset/test")

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader