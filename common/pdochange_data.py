import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


DEFAULT_MASK_NAMES = ["cd8"]
PDO_CHANGE_BIN_EDGES = [-100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
PDO_CHANGE_BIN_LABELS = [
    "x<-100",
    "-100<=x<-80",
    "-80<=x<-60",
    "-60<=x<-40",
    "-40<=x<-20",
    "-20<=x<0",
    "0<=x<20",
    "20<=x<40",
    "40<=x<60",
    "60<=x<80",
    "80<=x<100",
    "x>=100",
]


def load_split_json(split_json_path: Path) -> dict:
    with open(split_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    train_ids = payload.get("train", []) or []
    val_ids = payload.get("val", []) or []
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("Split JSON must contain list fields: train and val.")
    return payload


def load_pdo_change_labels(label_json_path: Path) -> dict[str, float]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: dict[str, float] = {}
    for k, v in payload.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def pdo_change_to_bin_index(value: float) -> int:
    v = float(value)
    for idx, edge in enumerate(PDO_CHANGE_BIN_EDGES):
        if v < edge:
            return idx
    return len(PDO_CHANGE_BIN_EDGES)


def pdo_change_bin_index_to_label(idx: int) -> str:
    i = int(idx)
    if i < 0 or i >= len(PDO_CHANGE_BIN_LABELS):
        raise ValueError(f"Invalid bin index: {idx}")
    return PDO_CHANGE_BIN_LABELS[i]


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _resolve_mask_path(case_dir: Path, name: str) -> Path | None:
    candidates = [
        case_dir / f"{name}.png",
        case_dir / f"{name.lower()}.png",
        case_dir / f"{name.upper()}.png",
        case_dir / f"{name}_mask.png",
        case_dir / f"{name.lower()}_mask.png",
        case_dir / f"{name.upper()}_mask.png",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


class EnsureTensor(torch.nn.Module):
    def forward(self, img):
        if torch.is_tensor(img):
            return img
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[:, :, None]
            return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return TF.to_tensor(img)


class NormalizeByChannels(torch.nn.Module):
    def __init__(self, mean=0.5, std=0.5):
        super().__init__()
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        channels = int(img.shape[0])
        mean = torch.full((channels, 1, 1), self.mean, dtype=img.dtype, device=img.device)
        std = torch.full((channels, 1, 1), self.std, dtype=img.dtype, device=img.device)
        return (img - mean) / std


class RandomMaskAugmentation(torch.nn.Module):
    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = float(p)
        self.augment_methods = ("hflip", "vflip", "rot90", "rot270")

    def forward(self, img):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        if torch.rand(1).item() > self.p:
            return img
        aug_name = self.augment_methods[int(torch.randint(0, len(self.augment_methods), (1,)).item())]
        if aug_name == "hflip":
            return TF.hflip(img)
        if aug_name == "vflip":
            return TF.vflip(img)
        if aug_name == "rot90":
            return torch.rot90(img, k=1, dims=(1, 2))
        if aug_name == "rot270":
            return torch.rot90(img, k=3, dims=(1, 2))
        return img


class MaskImageTransformations:
    def __init__(self, image_size=512, normalize_mean=0.5, normalize_std=0.5):
        self.image_size = int(image_size)
        self.normalize_mean = float(normalize_mean)
        self.normalize_std = float(normalize_std)
        self.train_transformations = transforms.Compose(
            [
                EnsureTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                RandomMaskAugmentation(p=1.0),
                NormalizeByChannels(self.normalize_mean, self.normalize_std),
            ]
        )
        self.validation_transformations = transforms.Compose(
            [
                EnsureTensor(),
                transforms.Resize((self.image_size, self.image_size)),
                NormalizeByChannels(self.normalize_mean, self.normalize_std),
            ]
        )


class OnchipPDOChangeDataset(Dataset):
    def __init__(
        self,
        masks_dir: Path,
        image_ids: list[str],
        pdo_change_labels: dict[str, float],
        mask_names: list[str] | None = None,
        transform=None,
    ):
        self.masks_dir = Path(masks_dir)
        self.image_ids = list(image_ids)
        self.pdo_change_labels = dict(pdo_change_labels)
        self.mask_names = list(mask_names) if mask_names is not None else list(DEFAULT_MASK_NAMES)
        self.transform = transform
        self.case_dir_index = self._build_case_dir_index()
        self.samples = self._build_samples()

    def _build_case_dir_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if not self.masks_dir.exists():
            return index
        for p in sorted(self.masks_dir.rglob("*")):
            if not p.is_dir():
                continue
            key = _normalize_key(p.name)
            if key not in index:
                index[key] = p
        return index

    def _resolve_case_dir(self, image_id: str) -> Path | None:
        direct = self.masks_dir / str(image_id)
        if direct.exists() and direct.is_dir():
            return direct
        return self.case_dir_index.get(_normalize_key(image_id))

    def _build_samples(self) -> list[dict]:
        samples = []
        for image_id in self.image_ids:
            if image_id not in self.pdo_change_labels:
                continue
            case_dir = self._resolve_case_dir(image_id)
            if case_dir is None:
                continue
            input_paths = []
            valid = True
            for name in self.mask_names:
                p = _resolve_mask_path(case_dir, name)
                if p is None:
                    valid = False
                    break
                input_paths.append(p)
            if not valid:
                continue
            samples.append(
                {
                    "image_id": str(image_id),
                    "input_paths": input_paths,
                    "raw_label": float(self.pdo_change_labels[image_id]),
                    "class_label": int(pdo_change_to_bin_index(self.pdo_change_labels[image_id])),
                }
            )
        if not samples:
            raise ValueError("No valid samples found. Check masks_dir, image_ids, labels, and mask names.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mask_arrays = []
        for p in sample["input_paths"]:
            with Image.open(p) as img:
                mask_arrays.append(np.array(img.convert("L"), dtype=np.uint8))
        image = np.stack(mask_arrays, axis=-1)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(sample["class_label"], dtype=torch.long)
        return image, label, sample["image_id"]
