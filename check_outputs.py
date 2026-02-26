import os
from pathlib import Path

checkpoints_dir = Path('outputs/checkpoints')
predictions_dir = Path('outputs/predictions')

print(f"Checkpoints directory exists: {checkpoints_dir.exists()}")
print(f"Predictions directory exists: {predictions_dir.exists()}")

if checkpoints_dir.exists():
    files = list(checkpoints_dir.glob('*'))
    print(f"Files in checkpoints: {len(files)}")
    for f in files:
        print(f"  - {f.name}")

if predictions_dir.exists():
    files = list(predictions_dir.glob('*'))
    print(f"Files in predictions: {len(files)}")
    for f in files:
        print(f"  - {f.name}")
