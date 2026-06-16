import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess.onchip_wsi.convert_masks import (
    CHANNEL_MAP,
    _resolve_channel_path,
    _save_png,
)

R4_WSI_PREFIX = "488-aSMA_PE-CD8_647-CK19_DAPI_"
DRUG_SPLIT_KEY = {
    "fap": "test_FAP",
    "igg": "test_IgG",
    "iareg": "test_iAREG",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Chip-R4 binary masks and update data_split.json additively."
    )
    parser.add_argument(
        "--base-dir", "--base_dir",
        dest="base_dir",
        default="data/On-chip_Data",
        help="Base directory containing Chip-R4_WSI and where Chip-R4_mask will be created",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing mask PNG files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print actions without writing any files",
    )
    return parser.parse_args(argv)


def _parse_r4_dir_name(dir_name: str) -> tuple[str, str, str] | None:
    """Parse '488-aSMA_PE-CD8_647-CK19_DAPI_285_FAP_d1' -> ('285', 'FAP', 'd1').

    Returns None for directories that do not match the expected prefix.
    The day token must start with 'd' followed by a digit.
    """
    if not dir_name.startswith(R4_WSI_PREFIX):
        return None
    rest = dir_name[len(R4_WSI_PREFIX):]  # e.g. "285_FAP_d1"
    parts = rest.split("_", 2)
    if len(parts) != 3:
        return None
    patient, drug, day = parts
    if not (day.startswith("d") and day[1:].isdigit()):
        return None
    return patient, drug, day


def _make_r4_case_id(patient: str, drug: str, day: str) -> str:
    """Returns e.g. 'chip-r4_nyu285_fap-d1'."""
    return f"chip-r4_nyu{patient}_{drug.lower()}-{day}"


def _make_r4_mask_dir_name(patient: str, drug: str, day: str) -> str:
    """Returns e.g. 'Chip-R4_NYU285_FAP-d1'."""
    return f"Chip-R4_NYU{patient}_{drug}-{day}"


def process_r4(
    wsi_dir: Path,
    mask_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, str]:
    """Generate binary mask PNGs for all R4 samples.

    Returns {case_id: mask_dir_path_str} for all successfully processed samples.
    """
    if not wsi_dir.is_dir():
        print(f"  [error] WSI dir not found: {wsi_dir}")
        return {}

    mapping: dict[str, str] = {}
    created = skipped_missing = skipped_exists = 0

    for sample_dir in sorted(wsi_dir.iterdir(), key=lambda p: p.name):
        if not sample_dir.is_dir():
            continue
        parsed = _parse_r4_dir_name(sample_dir.name)
        if parsed is None:
            print(f"  [skip] unrecognized dir: {sample_dir.name}")
            continue
        patient, drug, day = parsed
        case_id = _make_r4_case_id(patient, drug, day)
        out_folder = mask_dir / _make_r4_mask_dir_name(patient, drug, day)

        if dry_run:
            print(f"  [dry-run] {case_id} -> {out_folder}")
            # Still resolve channels to report missing ones
            missing = [
                ch_key for ch_key, tag in CHANNEL_MAP.items()
                if _resolve_channel_path(sample_dir, tag) is None
            ]
            if missing:
                print(f"    [warn] missing channels: {missing}")
            else:
                mapping[case_id] = str(out_folder)
            continue

        out_folder.mkdir(parents=True, exist_ok=True)

        channel_paths: dict[str, Path] = {}
        missing = False
        for ch_key, tag in CHANNEL_MAP.items():
            src = _resolve_channel_path(sample_dir, tag)
            if src is None:
                print(f"  [warn] {case_id}: missing channel '{ch_key}' (tag={tag}), skipping sample")
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
            print(f"  [created] {case_id}")
        else:
            skipped_exists += 1
            print(f"  [exists]  {case_id}")

        mapping[case_id] = str(out_folder)

    if not dry_run:
        print(
            f"\n[Chip-R4] created={created} "
            f"skipped_missing={skipped_missing} skipped_exists={skipped_exists}"
        )
    return mapping


def update_data_split_json(split_path: Path, r4_case_ids: list[str]) -> None:
    """Additively update data_split.json with R4 test splits.

    Existing train/val entries are never modified.
    """
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    split.setdefault("test", [])
    split.setdefault("test_FAP", [])
    split.setdefault("test_IgG", [])
    split.setdefault("test_iAREG", [])

    existing_test = set(split["test"])
    for cid in sorted(r4_case_ids):
        if cid in existing_test:
            continue
        split["test"].append(cid)
        drug_lower = cid.split("_")[2].split("-")[0]  # e.g. "fap", "igg", "iareg"
        split_key = DRUG_SPLIT_KEY.get(drug_lower)
        if split_key:
            if cid not in split[split_key]:
                split[split_key].append(cid)

    split["test"] = sorted(set(split["test"]))
    split["test_FAP"] = sorted(set(split["test_FAP"]))
    split["test_IgG"] = sorted(set(split["test_IgG"]))
    split["test_iAREG"] = sorted(set(split["test_iAREG"]))

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print(
        f"Updated {split_path}: "
        f"test={len(split['test'])} "
        f"test_FAP={len(split['test_FAP'])} "
        f"test_IgG={len(split['test_IgG'])} "
        f"test_iAREG={len(split['test_iAREG'])}"
    )


def update_image_id_mapping(mapping_path: Path, r4_mapping: dict[str, str]) -> None:
    """Additively merge R4 entries into image_id_mapping.json."""
    existing: dict[str, str] = {}
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(r4_mapping)
    merged = dict(sorted(existing.items()))
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    print(f"Updated {mapping_path}: {len(merged)} total entries")


def main(argv=None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)
    wsi_dir = base_dir / "Chip-R4_WSI"
    mask_dir = base_dir / "Chip-R4_mask"
    split_path = base_dir / "data_split.json"
    mapping_path = base_dir / "image_id_mapping.json"

    print(f"WSI dir : {wsi_dir}")
    print(f"Mask dir: {mask_dir}")
    print(f"Dry run : {args.dry_run}")
    print()

    r4_mapping = process_r4(wsi_dir, mask_dir, args.overwrite, args.dry_run)

    if not r4_mapping:
        print("[warn] No R4 samples processed.")
        return 1

    if args.dry_run:
        print(f"\n[dry-run] Would add {len(r4_mapping)} case IDs to data_split.json")
        for cid in sorted(r4_mapping):
            drug_lower = cid.split("_")[2].split("-")[0]
            split_key = DRUG_SPLIT_KEY.get(drug_lower, "unknown")
            print(f"  {cid}  -> {split_key}")
        return 0

    update_data_split_json(split_path, list(r4_mapping.keys()))
    update_image_id_mapping(mapping_path, r4_mapping)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
