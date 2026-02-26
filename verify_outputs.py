#!/usr/bin/env python
"""Verify training outputs."""

from pathlib import Path
import json

base_path = Path(r'c:\Users\tamburinij\Machine learning\Assignments\DSDA385-assigment-2\outputs')

print("=== CHECKPOINTS ===")
chkpt_dir = base_path / 'checkpoints'
if chkpt_dir.exists():
    for f in sorted(chkpt_dir.glob('*')):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size_mb:.2f} MB")

print("\n=== PREDICTIONS ===")
pred_dir = base_path / 'predictions'
if pred_dir.exists():
    files = list(pred_dir.glob('*'))
    print(f"  Total files: {len(files)}")
    img_files = list(pred_dir.glob('pred_*.jpg'))
    print(f"  Images (pred_*.jpg): {len(img_files)}")
    
    # Show sample files
    for f in sorted(pred_dir.glob('*.json')):
        print(f"  {f.name}")
    for f in sorted(pred_dir.glob('*.txt')):
        print(f"  {f.name}")
    for f in sorted(pred_dir.glob('*.csv')):
        print(f"  {f.name}")
    
    # Check summary.json
    summary_path = pred_dir / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n  Summary JSON keys: {list(summary.keys())}")
else:
    print("  Directory not found!")

print("\n=== LOGS ===")
logs_dir = base_path / 'logs'
if logs_dir.exists():
    for f in logs_dir.glob('*'):
        print(f"  {f.name}")
