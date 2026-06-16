"""Shared mask loading and path resolution helpers."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF


MASK_ALIASES = {
    "pd-1": ["pd-1", "pd1", "pd_1", "PD-1", "PD1", "PD_1"],
}


def normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def candidate_mask_names(name: str) -> list[str]:
    base = [name, name.lower(), name.upper()]
    aliases = MASK_ALIASES.get(name.lower(), [])
    out: list[str] = []
    seen: set[str] = set()
    for item in base + aliases:
        for variant in (item, item.lower(), item.upper()):
            if variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


def resolve_mask_path(case_dir: Path, name: str) -> Path | None:
    for candidate in candidate_mask_names(name):
        for path in (case_dir / f"{candidate}.png", case_dir / f"{candidate}_mask.png"):
            if path.exists() and path.is_file():
                return path
    return None


def load_gray_image(path: Path, image_size: int | None = None) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L")
        if image_size is not None:
            img = img.resize((image_size, image_size), resample=Image.NEAREST)
        return TF.pil_to_tensor(img).float() / 255.0
