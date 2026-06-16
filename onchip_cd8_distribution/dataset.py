import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


INPUT_MASK_NAMES = ["actin", "ck"]
TARGET_MASK_NAME = "cd8"


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def load_split_json(split_json_path: Path) -> dict:
    with open(split_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    train_ids = payload.get("train", []) or []
    val_ids = payload.get("val", []) or []
    if not isinstance(train_ids, list) or not isinstance(val_ids, list):
        raise ValueError("Split JSON must contain list fields: train and val.")
    return payload


def _load_gray_image(path: Path, image_size: int | None = None) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("L")
        if image_size is not None:
            img = img.resize((image_size, image_size), resample=Image.NEAREST)
        arr = TF.pil_to_tensor(img).float() / 255.0
    return arr


def _resolve_mask_path(case_dir: Path, name: str) -> Path | None:
    candidates = [
        case_dir / f"{name}.png",
        case_dir / f"{name.upper()}.png",
        case_dir / f"{name.lower()}.png",
        case_dir / f"{name}_mask.png",
        case_dir / f"{name.upper()}_mask.png",
        case_dir / f"{name.lower()}_mask.png",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


class OnchipCD8Dataset(Dataset):
    def __init__(
        self,
        masks_dir: Path,
        case_ids: list[str],
        input_mask_names: list[str] | None = None,
        target_mask_name: str = TARGET_MASK_NAME,
        image_size: int = 512,
        augment: bool = True,
        augment_copies: int = 4,
        normalize: bool = True,
    ):
        self.masks_dir = Path(masks_dir)
        self.case_ids = list(case_ids)
        self.input_mask_names = list(input_mask_names) if input_mask_names is not None else list(INPUT_MASK_NAMES)
        self.target_mask_name = str(target_mask_name)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.augment_copies = max(int(augment_copies), 0)
        self.normalize = bool(normalize)
        self.augment_methods = [
            "none",
            "hflip",
            "vflip",
            "rot90",
            "rot180",
            "rot270",
            "brightness",
            "contrast",
            "gaussian_blur",
            "gaussian_noise",
            "random_crop",
        ]
        self.case_dir_index = self._build_case_dir_index()
        self.base_samples = self._build_base_samples()
        self.samples = self._build_samples()

    def _build_case_dir_index(self) -> dict[str, Path]:
        index = {}
        if not self.masks_dir.exists():
            return index
        for p in sorted(self.masks_dir.rglob("*")):
            if not p.is_dir():
                continue
            if _resolve_mask_path(p, "cd8") is None:
                continue
            key = _normalize_key(p.name)
            if key not in index:
                index[key] = p
        return index

    def _resolve_case_dir(self, case_id: str) -> Path | None:
        direct = self.masks_dir / str(case_id)
        if direct.exists() and direct.is_dir():
            return direct
        return self.case_dir_index.get(_normalize_key(case_id))

    def _build_base_samples(self) -> list[dict]:
        samples = []
        for case_id in self.case_ids:
            case_dir = self._resolve_case_dir(case_id)
            if case_dir is None:
                continue
            input_paths = []
            ok = True
            for name in self.input_mask_names:
                p = _resolve_mask_path(case_dir, name)
                if p is None:
                    ok = False
                    break
                input_paths.append(p)
            target_path = _resolve_mask_path(case_dir, self.target_mask_name)
            if target_path is None:
                ok = False
            if ok:
                samples.append(
                    {
                        "case_id": str(case_id),
                        "input_paths": input_paths,
                        "target_path": target_path,
                    }
                )
        if not samples:
            raise ValueError("No valid samples found. Check masks_dir, case_ids, and mask names.")
        return samples

    def _build_samples(self) -> list[dict]:
        samples = []
        if not self.augment:
            for base in self.base_samples:
                samples.append({"base": base, "aug": "none", "case_id": base["case_id"]})
            self.samples = samples
            return samples
        geometric_augs = ["hflip", "vflip", "rot90", "rot180", "rot270"]
        for base in self.base_samples:
            samples.append({"base": base, "aug": "none", "case_id": base["case_id"]})
            for _ in range(self.augment_copies):
                aug = random.choice(geometric_augs)
                samples.append({"base": base, "aug": aug, "case_id": base["case_id"]})
            for _ in range(self.augment_copies):
                samples.append({"base": base, "aug": "brightness", "case_id": base["case_id"]})
            for _ in range(self.augment_copies):
                samples.append({"base": base, "aug": "contrast", "case_id": base["case_id"]})
            for _ in range(self.augment_copies):
                samples.append({"base": base, "aug": "gaussian_blur", "case_id": base["case_id"]})
            for _ in range(self.augment_copies):
                samples.append({"base": base, "aug": "gaussian_noise", "case_id": base["case_id"]})
            for _ in range(self.augment_copies):
                samples.append({"base": base, "aug": "random_crop", "case_id": base["case_id"]})
        self.samples = samples
        return samples

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, x: torch.Tensor, y: torch.Tensor, aug_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if aug_name == "none":
            pass
        elif aug_name == "hflip":
            x = TF.hflip(x)
            y = TF.hflip(y)
        elif aug_name == "vflip":
            x = TF.vflip(x)
            y = TF.vflip(y)
        elif aug_name == "rot90":
            x = torch.rot90(x, k=1, dims=(1, 2))
            y = torch.rot90(y, k=1, dims=(1, 2))
        elif aug_name == "rot180":
            x = torch.rot90(x, k=2, dims=(1, 2))
            y = torch.rot90(y, k=2, dims=(1, 2))
        elif aug_name == "rot270":
            x = torch.rot90(x, k=3, dims=(1, 2))
            y = torch.rot90(y, k=3, dims=(1, 2))
        elif aug_name == "brightness":
            factor = random.uniform(0.7, 1.3)
            x = torch.clamp(x * factor, 0.0, 1.0)
        elif aug_name == "contrast":
            factor = random.uniform(0.7, 1.3)
            mean_val = x.mean(dim=(1, 2), keepdim=True)
            x = torch.clamp((x - mean_val) * factor + mean_val, 0.0, 1.0)
        elif aug_name == "gaussian_blur":
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            x = TF.gaussian_blur(x.unsqueeze(0), kernel_size=kernel_size, sigma=sigma).squeeze(0)
        elif aug_name == "gaussian_noise":
            noise = torch.randn_like(x) * random.uniform(0.01, 0.05)
            x = torch.clamp(x + noise, 0.0, 1.0)
        elif aug_name == "random_crop":
            crop_size = int(self.image_size * random.uniform(0.85, 0.95))
            offset_h = random.randint(0, self.image_size - crop_size)
            offset_w = random.randint(0, self.image_size - crop_size)
            x = x[:, offset_h : offset_h + crop_size, offset_w : offset_w + crop_size]
            y = y[:, offset_h : offset_h + crop_size, offset_w : offset_w + crop_size]
            x = F.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0)
        return x, y

    def __getitem__(self, idx):
        sample = self.samples[idx]
        base = sample["base"]
        input_tensors = [_load_gray_image(p, image_size=self.image_size) for p in base["input_paths"]]
        target_tensor = _load_gray_image(base["target_path"], image_size=self.image_size)
        x_t = torch.cat(input_tensors, dim=0).float()
        y_t = target_tensor.float()
        x_t, y_t = self._apply_augmentation(x_t, y_t, sample["aug"])
        if self.normalize:
            c = int(x_t.shape[0])
            x_t = TF.normalize(x_t, mean=[0.5] * c, std=[0.5] * c)
        return x_t, y_t, base["case_id"]
