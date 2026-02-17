import os
import shutil
from pathlib import Path
import random

def organize_pennfudan_dataset():
    """Organize PennFudan dataset into train/val/test folders"""
    
    source_dir = Path(r"c:\Users\tamburinij\Downloads\PennFudanPed\PennFudanPed")
    data_dir = Path("./data")
    
    # Create directory structure
    (data_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "train" / "masks").mkdir(parents=True, exist_ok=True)
    (data_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "val" / "masks").mkdir(parents=True, exist_ok=True)
    (data_dir / "test" / "images").mkdir(parents=True, exist_ok=True)
    (data_dir / "test" / "masks").mkdir(parents=True, exist_ok=True)
    
    # Find images and masks directories
    images_dir = source_dir / "PNGImages"
    masks_dir = source_dir / "PedMasks"
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        print("Available directories:")
        for item in source_dir.iterdir():
            print(f"  - {item.name}")
        return
    
    image_files = sorted([f.name for f in images_dir.glob('*.png')])
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split (80% train, 10% val, 10% test)
    random.seed(42)
    random.shuffle(image_files)
    train_split = int(0.8 * len(image_files))
    val_split = int(0.9 * len(image_files))
    
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]
    
    # Copy training files
    for img_file in train_files:
        mask_file = img_file.replace('.png', '_mask.png')
        shutil.copy(images_dir / img_file, data_dir / "train" / "images" / img_file)
        shutil.copy(masks_dir / mask_file, data_dir / "train" / "masks" / mask_file)
    
    # Copy validation files
    for img_file in val_files:
        mask_file = img_file.replace('.png', '_mask.png')
        shutil.copy(images_dir / img_file, data_dir / "val" / "images" / img_file)
        shutil.copy(masks_dir / mask_file, data_dir / "val" / "masks" / mask_file)
    
    # Copy test files
    for img_file in test_files:
        mask_file = img_file.replace('.png', '_mask.png')
        shutil.copy(images_dir / img_file, data_dir / "test" / "images" / img_file)
        shutil.copy(masks_dir / mask_file, data_dir / "test" / "masks" / mask_file)
    
    print(f"✓ Train: {len(train_files)} images")
    print(f"✓ Val: {len(val_files)} images")
    print(f"✓ Test: {len(test_files)} images")
    print("✓ Dataset organized successfully!")

if __name__ == "__main__":
    organize_pennfudan_dataset()