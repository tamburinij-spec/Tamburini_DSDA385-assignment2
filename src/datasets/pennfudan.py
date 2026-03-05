import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class PennFudanDataset(Dataset):
    """Simple loader for the PennFudanPed segmentation dataset.

    The dataset is assumed to be organized as:

        data/pennfudan/{split}/images/*.png
        data/pennfudan/{split}/masks/*_mask.png

    where ``split`` is one of ``train``, ``val`` or ``test``.
    """

    def __init__(self, images_dir, masks_dir, transform=None, target_size=(512, 512)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.target_size = target_size
        self.image_files = sorted([f.name for f in self.images_dir.glob("*.png")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace(".png", "_mask.png")

        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / mask_name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        return image, mask



def get_data_loaders(batch_size=2, num_workers=0):
    """Return train/val/test DataLoader for PennFudan dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    base = Path("data") / "pennfudan"
    train_dataset = PennFudanDataset(base / "train" / "images", base / "train" / "masks", transform)
    val_dataset = PennFudanDataset(base / "val" / "images", base / "val" / "masks", transform)
    test_dataset = PennFudanDataset(base / "test" / "images", base / "test" / "masks", transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
