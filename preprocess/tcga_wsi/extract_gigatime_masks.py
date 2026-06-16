import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image


CHANNEL_NAMES = [
    "DAPI",
    "TRITC",
    "Cy5",
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default="data_f/TCGA_gigatime")
    parser.add_argument("--output-dir", type=Path, default="data_f/TCGA_WSI_masks")
    parser.add_argument("--pattern", type=str, default="*.npy")
    parser.add_argument("--mask-shape", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _extract_channel_mask(packed: np.ndarray, channel_index: int) -> np.ndarray:
    if packed.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, packed_channels), got shape={packed.shape}")

    byte_idx = int(channel_index // 2)
    shift = 4 if (channel_index % 2) == 1 else 0
    if byte_idx < 0 or byte_idx >= packed.shape[2]:
        raise IndexError(f"channel_index={channel_index} out of range for packed shape={packed.shape}")

    vals = (packed[:, :, byte_idx] >> shift) & 0x0F
    return (vals > 0).astype(np.uint8, copy=False)


def main() -> int:
    args = parse_args()
    mask_shape = int(args.mask_shape)
    if mask_shape <= 0:
        raise ValueError("--mask-shape must be > 0")

    name_to_idx = {name: i for i, name in enumerate(CHANNEL_NAMES)}
    required = {
        "CD8": "cd8",
        "CD4": "cd4",
        "CD68": "cd68",
        "Actin-D": "actin",
        "CK": "ck",
        "PD-1": "pd1",
    }

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    inputs = sorted([p for p in input_dir.glob(args.pattern) if p.is_file()], key=lambda p: p.name)
    for in_path in tqdm(inputs, desc="Extracting masks"):
        packed = np.load(in_path, mmap_mode="r")
        image_name = in_path.stem
        out_dir = out_root / image_name
        expected_paths = {key: out_dir / f"{key}.png" for key in required.values()}

        masks = {}
        for name, key in required.items():
            idx = name_to_idx.get(name)
            if idx is None:
                raise KeyError(f"Missing channel name: {name}")
            masks[key] = _extract_channel_mask(packed, idx)

        out_dir.mkdir(parents=True, exist_ok=True)
        for key, mask in masks.items():
            if not args.overwrite and expected_paths[key].exists():
                continue
            img = Image.fromarray((mask * 255).astype(np.uint8, copy=False), mode="L")
            if img.size != (mask_shape, mask_shape):
                img = img.resize((mask_shape, mask_shape), resample=Image.NEAREST)
            img.save(expected_paths[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
