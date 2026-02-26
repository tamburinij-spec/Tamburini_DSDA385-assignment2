#!/usr/bin/env python
"""List all output files created."""

from pathlib import Path

base = Path(r'c:\Users\tamburinij\Machine learning\Assignments\DSDA385-assigment-2\outputs')

for filepath in base.rglob('*'):
    if filepath.is_file():
        size = filepath.stat().st_size
        print(f"{filepath.relative_to(base)} ({size} bytes)")
    else:
        print(f"{filepath.relative_to(base)}/ (dir)")
