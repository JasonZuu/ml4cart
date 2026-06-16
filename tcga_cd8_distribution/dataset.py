import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


INPUT_MASK_NAMES = ["cd4", "cd68", "ck", "actin", "pd-1", "tissue"]
TARGET_MASK_NAME = "cd8"
MASK_ALIASES = {
    "pd-1": ["pd-1", "pd1", "pd_1", "PD-1", "PD1", "PD_1"],
}


def _candidate_names(name: str) -> list[str]:
    base = [name, name.lower(), name.upper()]
    aliases = MASK_ALIASES.get(name.lower(), [])
    merged = []
    seen = set()
    for item in base + aliases:
        for s in [item, item.lower(), item.upper()]:
            if s not in seen:
                seen.add(s)
                merged.append(s)
    return merged


def resolve_mask_path(case_dir: Path, name: str) -> Path | None:
    for n in _candidate_names(name):
        candidates = [
            case_dir / f"{n}.png",
            case_dir / f"{n}_mask.png",
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
    return None


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


class TCGACD8Dataset(Dataset):
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
        self.augment_methods = ["hflip", "vflip", "rot90", "rot270"]
        self.base_samples = self._build_base_samples()
        self.samples = self._build_samples()

    def _build_base_samples(self) -> list[dict]:
        samples = []
        for case_id in self.case_ids:
            case_dir = self.masks_dir / case_id
            if not case_dir.exists() or not case_dir.is_dir():
                continue
            input_paths = []
            ok = True
            for name in self.input_mask_names:
                p = resolve_mask_path(case_dir, name)
                if p is None:
                    ok = False
                    break
                input_paths.append(p)
            target_path = resolve_mask_path(case_dir, self.target_mask_name)
            if target_path is None:
                ok = False
            if ok:
                samples.append(
                    {
                        "case_id": case_id,
                        "input_paths": input_paths,
                        "target_path": target_path,
                    }
                )
        if not samples:
            raise ValueError("No valid samples found. Check masks_dir, case_ids, and mask names.")
        return samples

    def _build_samples(self) -> list[dict]:
        samples = []
        for base in self.base_samples:
            samples.append({"base": base, "aug": "none", "case_id": base["case_id"]})
            if self.augment:
                for aug_name in self.augment_methods[: self.augment_copies]:
                    samples.append({"base": base, "aug": aug_name, "case_id": base["case_id"]})
        return samples

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, x: torch.Tensor, y: torch.Tensor, aug_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if aug_name == "hflip":
            x = TF.hflip(x)
            y = TF.hflip(y)
        elif aug_name == "vflip":
            x = TF.vflip(x)
            y = TF.vflip(y)
        elif aug_name == "rot90":
            x = torch.rot90(x, k=1, dims=(1, 2))
            y = torch.rot90(y, k=1, dims=(1, 2))
        elif aug_name == "rot270":
            x = torch.rot90(x, k=3, dims=(1, 2))
            y = torch.rot90(y, k=3, dims=(1, 2))
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
