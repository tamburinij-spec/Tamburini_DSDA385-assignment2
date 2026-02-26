"""Simple configuration loader for YAML files.

This module exposes constants that other parts of the codebase can import.
The primary file is ``dataset.yaml``, which contains the original values
that were defined in ``src/config.py`` before the reorganization.  Additional
model-specific configuration files (``faster_rcnn.yaml`` and ``yolo.yaml``)
can be loaded on demand by calling :func:`load_config`.
"""

from pathlib import Path
import yaml
import torch

__all__ = [
    "DEVICE",
    "TRAINING_CONFIG",
    "MODEL_CONFIG",
    "DATA_CONFIG",
    "PATHS",
    "load_config",
]

_base = Path(__file__).parent

# load the default dataset/training configuration
with open(_base / "dataset.yaml", "r") as f:
    _cfg = yaml.safe_load(f)

TRAINING_CONFIG = _cfg.get("training", {})
MODEL_CONFIG = _cfg.get("model", {})
DATA_CONFIG = _cfg.get("data", {})
PATHS = _cfg.get("paths", {})

# device is determined at runtime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(name: str) -> dict:
    """Load an arbitrary YAML configuration file located in the same folder.

    Args:
        name: filename without extension, e.g. ``"faster_rcnn"``.
    Returns:
        Parsed configuration dictionary. Raises ``FileNotFoundError`` if the
        file does not exist.
    """
    path = _base / f"{name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)
