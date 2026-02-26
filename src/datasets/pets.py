"""Placeholder dataset module for pet subset experiments."""

from torch.utils.data import Dataset


class PetsDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        # TODO: implement pet dataset loader if required
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError("PetsDataset is not implemented yet")
