#!/usr/bin/env python
"""Run training and capture complete output."""

import sys
import os
from pathlib import Path

# Change to project root
project_root = Path(__file__).parent
os.chdir(project_root)

# Add src to path
sys.path.insert(0, str(project_root / 'src'))

# Now run main
from src.main import main

if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Main completed successfully")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
