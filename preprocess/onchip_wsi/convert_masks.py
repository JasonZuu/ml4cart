import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


CHANNEL_MAP = {
    "ck": "640",
    "cd8": "561",
    "actin": "BF",
    "dapi": "405",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert On-chip WSI data to TCGA-format masks and generate metadata JSONs."
    )
    parser.add_argument(
        "--base-dir", "--base_dir",
        dest="base_dir",
        default="data/On-chip_Data",
        help="Base directory containing Chip-Rx_WSI and Chip-Rx_mask subdirs",
    )
    parser.add_argument(
        "--rounds",
        default="R1,R2,R3",
        help="Comma-separated rounds to include, e.g. R1,R2,R3",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing mask PNG files",
    )
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        default=False,
        help="Skip renaming existing Chip-R1_mask subdirs to new format",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        default=False,
        help="Skip mask generation (only rename and build JSONs)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _parse_patient_name(dir_name: str) -> str:
    """Strip leading index from patient dir name: '1. NYU 318' -> 'NYU 318'."""
    match = re.match(r"^\d+\.\s+(.+)$", dir_name)
    return match.group(1) if match else dir_name


def _patient_compact(patient_name: str) -> str:
    """Remove spaces and dashes: 'NYU 318' -> 'NYU318', 'NCI 2' -> 'NCI2'."""
    return re.sub(r"[\s\-]+", "", patient_name)


def _parse_device(name: str) -> str:
    """Extract device/day identifier from image dir name."""
    match = re.search(r"(?:^|_|-|\b)day[_\s-]*([0-9]+)", name, re.IGNORECASE)
    if match:
        return f"d{match.group(1)}"
    match = re.search(r"(?:^|_|-|\b)d([0-9]+)", name, re.IGNORECASE)
    if match:
        return f"d{match.group(1)}"
    return name.strip().lower().replace(" ", "_")


def _make_image_id(case_id: str, patient_name: str, device: str) -> str:
    """Build lowercase image ID: 'Chip-R1', 'NYU 318', 'd1' -> 'chip-r1_nyu318-d1'."""
    compact = _patient_compact(patient_name).lower()
    return f"{case_id.lower()}_{compact}-{device}"


def _make_mask_dir_name(case_id: str, patient_name: str, device: str) -> str:
    """Build mask dir name: 'Chip-R1', 'NYU 318', 'd1' -> 'Chip-R1_NYU318-d1'."""
    compact = _patient_compact(patient_name)
    return f"{case_id}_{compact}-{device}"


# ---------------------------------------------------------------------------
# Channel / image processing
# ---------------------------------------------------------------------------

def _is_rgb_name(name: str) -> bool:
    return "RGB" in name.upper()


def _resolve_channel_path(folder: Path, tag: str) -> Path | None:
    matches = []
    for p in folder.glob("*.tif"):
        name = p.name
        if name.lower().endswith(f" {tag.lower()}.tif") and _is_rgb_name(name):
            matches.append(p)
    matches.sort(key=lambda p: p.name)
    return matches[0] if matches else None


def _otsu_threshold(arr: np.ndarray) -> int:
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    if total == 0:
        return 0
    prob = hist / float(total)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.nanargmax(sigma_b))


def _save_png(src_path: Path, dst_path: Path, overwrite: bool) -> bool:
    if dst_path.exists() and not overwrite:
        return False
    with Image.open(src_path) as img:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        thresh = _otsu_threshold(arr)
        mask = (arr > thresh).astype(np.uint8) * 255
        Image.fromarray(mask).save(dst_path)
    return True


# ---------------------------------------------------------------------------
# R1 mask renaming
# ---------------------------------------------------------------------------

def rename_r1_masks(mask_dir: Path) -> dict[str, str]:
    """Rename Chip-R1_mask subdirs from old to new convention.

    Old: Chip-round1_NCI-2_d1   New: Chip-R1_NCI2-d1
    Old: Chip-round1_NYU-318_d1  New: Chip-R1_NYU318-d1
    Returns mapping {old_name: new_name} for all renamed dirs.
    """
    pattern = re.compile(r"^Chip-round1_([A-Za-z]+-\d+(?:\.\d+)?)_([Dd]\d+.*)$")
    renames: dict[str, str] = {}
    for subdir in sorted(mask_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = pattern.match(subdir.name)
        if not m:
            continue
        patient_with_dash = m.group(1)   # e.g. "NCI-2" or "NYU-318"
        device = m.group(2)              # e.g. "d1"
        patient_compact = patient_with_dash.replace("-", "")  # "NCI2" / "NYU318"
        new_name = f"Chip-R1_{patient_compact}-{device}"
        new_path = mask_dir / new_name
        if subdir.name != new_name:
            subdir.rename(new_path)
            renames[subdir.name] = new_name
    return renames


# ---------------------------------------------------------------------------
# Image-ID mapping
# ---------------------------------------------------------------------------

def build_image_id_mapping(base_dir: Path, rounds: list[str]) -> dict[str, str]:
    """Scan all WSI dirs and return {image_id: relative_dir_path}."""
    mapping: dict[str, str] = {}
    for r in rounds:
        wsi_dir = base_dir / f"Chip-{r}_WSI"
        case_id = f"Chip-{r}"
        if not wsi_dir.is_dir():
            print(f"  [warning] WSI dir not found: {wsi_dir}")
            continue
        for patient_dir in sorted(wsi_dir.iterdir(), key=lambda p: p.name):
            if not patient_dir.is_dir():
                continue
            patient_name = _parse_patient_name(patient_dir.name)
            for day_dir in sorted(patient_dir.iterdir(), key=lambda p: p.name):
                if not day_dir.is_dir():
                    continue
                device = _parse_device(day_dir.name)
                image_id = _make_image_id(case_id, patient_name, device)
                mapping[image_id] = str(day_dir)
    return mapping


# ---------------------------------------------------------------------------
# Mask generation for a single round
# ---------------------------------------------------------------------------

def process_round(
    wsi_dir: Path,
    mask_dir: Path,
    case_id: str,
    overwrite: bool,
) -> tuple[int, int, int]:
    """Generate binary mask PNGs for all images in wsi_dir.

    Returns (created, skipped_missing, skipped_exists).
    """
    if not wsi_dir.is_dir():
        print(f"  [warning] WSI dir not found, skipping: {wsi_dir}")
        return 0, 0, 0

    created = skipped_missing = skipped_exists = 0

    for patient_dir in sorted(wsi_dir.iterdir(), key=lambda p: p.name):
        if not patient_dir.is_dir():
            continue
        patient_name = _parse_patient_name(patient_dir.name)
        for day_dir in sorted(patient_dir.iterdir(), key=lambda p: p.name):
            if not day_dir.is_dir():
                continue
            device = _parse_device(day_dir.name)
            out_folder = mask_dir / _make_mask_dir_name(case_id, patient_name, device)
            out_folder.mkdir(parents=True, exist_ok=True)

            channel_paths: dict[str, Path] = {}
            missing = False
            for ch_key, tag in CHANNEL_MAP.items():
                src = _resolve_channel_path(day_dir, tag)
                if src is None:
                    missing = True
                    break
                channel_paths[ch_key] = src

            if missing:
                skipped_missing += 1
                continue

            wrote_any = False
            for ch_key, src in channel_paths.items():
                dst = out_folder / f"{ch_key}.png"
                if _save_png(src, dst, overwrite):
                    wrote_any = True
            if wrote_any:
                created += 1
            else:
                skipped_exists += 1

    return created, skipped_missing, skipped_exists


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)
    rounds = [r.strip() for r in args.rounds.split(",")]

    # Step 1: Rename existing Chip-R1_mask subdirs
    if not args.skip_rename and "R1" in rounds:
        r1_mask_dir = base_dir / "Chip-R1_mask"
        if r1_mask_dir.is_dir():
            renames = rename_r1_masks(r1_mask_dir)
            print(f"[Chip-R1] renamed {len(renames)} mask dirs")
            for old, new in renames.items():
                print(f"  {old} -> {new}")
        else:
            print(f"[Chip-R1] mask dir not found, skipping rename: {r1_mask_dir}")

    # Step 2: Process R2 and R3 (R1 masks already exist)
    if not args.skip_process:
        for r in rounds:
            if r == "R1":
                continue  # R1 already processed; only renamed above
            case_id = f"Chip-{r}"
            wsi_dir = base_dir / f"Chip-{r}_WSI"
            mask_dir = base_dir / f"Chip-{r}_mask"
            created, skipped_missing, skipped_exists = process_round(
                wsi_dir, mask_dir, case_id, args.overwrite
            )
            print(
                f"[{case_id}] created={created} "
                f"skipped_missing={skipped_missing} skipped_exists={skipped_exists}"
            )

    # Step 3: Build and save image_id_mapping.json
    mapping = build_image_id_mapping(base_dir, rounds)
    mapping_path = base_dir / "image_id_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"Saved {len(mapping)} image IDs -> {mapping_path}")

    # Step 4: Build and save data_split.json
    train_ids = sorted(k for k in mapping if k.startswith(("chip-r1_", "chip-r2_")))
    val_ids = sorted(k for k in mapping if k.startswith("chip-r3_"))
    split = {"train": train_ids, "val": val_ids, "test": []}
    split_path = base_dir / "data_split.json"
    split_path.write_text(json.dumps(split, indent=2))
    print(f"Saved data_split.json: {len(train_ids)} train, {len(val_ids)} val -> {split_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
