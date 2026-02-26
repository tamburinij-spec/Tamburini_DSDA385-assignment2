#!/usr/bin/env python
"""Simple script to run training with proper path handling."""

import sys
import os
from pathlib import Path

# Change to project root
project_root = Path(__file__).parent
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

# Add src to path
sys.path.insert(0, str(project_root / 'src'))

# Now run main
from src.main import main

if __name__ == "__main__":
    main()
