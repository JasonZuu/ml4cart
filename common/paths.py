"""Project path helpers.

The compact project keeps code under ``ML4CAR-T`` and data outside the package.
Environment variables can override the defaults without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


def data_root() -> Path:
    return Path(os.environ.get("ML4CART_DATA_ROOT", REPO_ROOT / "data"))


def dynamics_root() -> Path:
    return Path(os.environ.get("ML4CART_DYNAMICS_ROOT", REPO_ROOT / "dynamics_data"))


def tcga_masks_dir() -> Path:
    return data_root() / "TCGA_Data" / "TCGA_WSI_masks"


def onchip_data_dir() -> Path:
    return data_root() / "On-chip_Data"


def dynamics_generated_dir() -> Path:
    return dynamics_root() / "generated"


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
