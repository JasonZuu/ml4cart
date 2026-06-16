import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


DEFAULT_WSI_ROOT_NAME = (
    "Chip WSI_20260416_8 Patients CART, stroma, immune, and PDO for AI_Round 1"
)
CASE_ID = "Chip-R1"
R2_CHANNEL_MAP = {
    "cd68": "405",
    "cd8": "561",
    "ck": "640",
    "actin": "BF",
}
VAL_DAY_PRIORITY = ("d1", "d2", "d3")


@dataclass(frozen=True)
class R2Sample:
    image_id: str
    mask_dir_name: str
    patient_id: str
    day: str
    source_dir: Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate standalone masks and split files for data/On-chip_Data_R2."
    )
    parser.add_argument(
        "--base-dir",
        "--base_dir",
        dest="base_dir",
        default="data/On-chip_Data_R2",
        help="Base directory containing the new On-chip_Data_R2 WSI export.",
    )
    parser.add_argument(
        "--wsi-root-name",
        default=DEFAULT_WSI_ROOT_NAME,
        help="Name of the WSI root directory inside base-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing mask PNG files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print actions without writing masks or JSON files.",
    )
    return parser.parse_args(argv)


def _patient_compact(patient: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", patient).upper()


def _parse_sample_dir(sample_dir: Path) -> R2Sample | None:
    pattern = re.compile(
        r"^BV421-CD68_488-a-SMA_PE-CD8_647-CK19_"
        r"(?P<index>\d+)_(?P<patient>[A-Za-z]+\d+)_"
        r"(?P<day>d\d+)$",
        re.IGNORECASE,
    )
    match = pattern.match(sample_dir.name)
    if match is None:
        return None

    patient_id = _patient_compact(match.group("patient"))
    day = match.group("day").lower()
    image_id = f"{CASE_ID.lower()}_{patient_id.lower()}-{day}"
    mask_dir_name = f"{CASE_ID}_{patient_id}-{day}"
    return R2Sample(
        image_id=image_id,
        mask_dir_name=mask_dir_name,
        patient_id=patient_id.lower(),
        day=day,
        source_dir=sample_dir,
    )


def _resolve_channel_path(folder: Path, tag: str) -> Path | None:
    matches = []
    for p in folder.glob("*.tif"):
        name = p.name
        if name.lower().endswith(f" {tag.lower()}.tif") and "RGB" in name.upper():
            matches.append(p)
    matches.sort(key=lambda p: p.name)
    return matches[0] if matches else None


def _save_mask_png(src_path: Path, dst_path: Path, overwrite: bool) -> bool:
    from preprocess.onchip_wsi.convert_masks import _save_png

    return _save_png(src_path, dst_path, overwrite)


def _iter_r2_samples(wsi_root: Path) -> list[R2Sample]:
    samples: list[R2Sample] = []
    if not wsi_root.is_dir():
        return samples
    for patient_dir in sorted(wsi_root.iterdir(), key=lambda p: p.name):
        if not patient_dir.is_dir():
            continue
        for sample_dir in sorted(patient_dir.iterdir(), key=lambda p: p.name):
            if not sample_dir.is_dir():
                continue
            sample = _parse_sample_dir(sample_dir)
            if sample is None:
                print(f"  [skip] unrecognized sample dir: {sample_dir}")
                continue
            samples.append(sample)
    return samples


def _resolve_sample_channels(sample: R2Sample) -> dict[str, Path] | None:
    channel_paths: dict[str, Path] = {}
    missing = []
    for mask_name, tag in R2_CHANNEL_MAP.items():
        src = _resolve_channel_path(sample.source_dir, tag)
        if src is None:
            missing.append(f"{mask_name}({tag})")
            continue
        channel_paths[mask_name] = src
    if missing:
        print(f"  [warn] {sample.image_id}: missing channels {', '.join(missing)}")
        return None
    return channel_paths


def process_samples(
    samples: list[R2Sample],
    mask_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, R2Sample]:
    processed: dict[str, R2Sample] = {}
    created = skipped_missing = skipped_exists = 0

    for sample in samples:
        channel_paths = _resolve_sample_channels(sample)
        if channel_paths is None:
            skipped_missing += 1
            continue

        out_folder = mask_dir / sample.mask_dir_name
        if dry_run:
            print(f"  [dry-run] {sample.image_id} -> {out_folder}")
            processed[sample.image_id] = sample
            continue

        out_folder.mkdir(parents=True, exist_ok=True)
        wrote_any = False
        for mask_name, src in channel_paths.items():
            dst = out_folder / f"{mask_name}.png"
            if _save_mask_png(src, dst, overwrite):
                wrote_any = True

        if wrote_any:
            created += 1
            print(f"  [created] {sample.image_id}")
        else:
            skipped_exists += 1
            print(f"  [exists]  {sample.image_id}")
        processed[sample.image_id] = sample

    if not dry_run:
        print(
            f"\n[{CASE_ID}] created={created} "
            f"skipped_missing={skipped_missing} skipped_exists={skipped_exists}"
        )
    elif skipped_missing:
        print(f"\n[dry-run] skipped_missing={skipped_missing}")
    return dict(sorted(processed.items()))


def build_split(processed: dict[str, R2Sample]) -> dict[str, list[str]]:
    by_patient: dict[str, list[R2Sample]] = {}
    for sample in processed.values():
        by_patient.setdefault(sample.patient_id, []).append(sample)

    val_ids: list[str] = []
    for patient_id in sorted(by_patient):
        patient_samples = by_patient[patient_id]
        day_to_sample = {sample.day: sample for sample in patient_samples}
        selected = None
        for day in VAL_DAY_PRIORITY:
            if day in day_to_sample:
                selected = day_to_sample[day]
                break
        if selected is None:
            print(
                f"  [warn] {patient_id}: no d1/d2/d3 sample available; "
                "all samples remain in train"
            )
            continue
        val_ids.append(selected.image_id)

    val_set = set(val_ids)
    train_ids = sorted(image_id for image_id in processed if image_id not in val_set)
    return {"train": train_ids, "val": sorted(val_ids), "test": []}


def _validate_split(split: dict[str, list[str]]) -> None:
    patient_seen: set[str] = set()
    duplicates: list[str] = []
    for image_id in split["val"]:
        match = re.match(r"^chip-r1_([a-z0-9]+)-d\d+$", image_id)
        if match is None:
            continue
        patient_id = match.group(1)
        if patient_id in patient_seen:
            duplicates.append(patient_id)
        patient_seen.add(patient_id)
    if duplicates:
        raise ValueError(f"Validation split has duplicate patient IDs: {duplicates}")


def write_outputs(base_dir: Path, processed: dict[str, R2Sample], split: dict[str, list[str]]) -> None:
    mapping = {
        image_id: str(sample.source_dir)
        for image_id, sample in sorted(processed.items())
    }
    mapping_path = base_dir / "image_id_mapping.json"
    split_path = base_dir / "data_split.json"

    mapping_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Saved {len(mapping)} image IDs -> {mapping_path}")
    print(
        f"Saved data_split.json: {len(split['train'])} train, "
        f"{len(split['val'])} val, {len(split['test'])} test -> {split_path}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)
    wsi_root = base_dir / args.wsi_root_name
    mask_dir = base_dir / f"{CASE_ID}_mask"

    print(f"WSI root: {wsi_root}")
    print(f"Mask dir: {mask_dir}")
    print(f"Dry run : {args.dry_run}")
    print()

    samples = _iter_r2_samples(wsi_root)
    if not samples:
        print(f"[error] No R2 samples found under: {wsi_root}")
        return 1

    processed = process_samples(samples, mask_dir, args.overwrite, args.dry_run)
    if not processed:
        print("[error] No R2 samples had all required channels.")
        return 1

    split = build_split(processed)
    _validate_split(split)

    print()
    print(f"Split preview: train={len(split['train'])} val={len(split['val'])} test=0")
    print("Validation IDs:")
    for image_id in split["val"]:
        print(f"  {image_id}")

    if args.dry_run:
        print("\n[dry-run] Would write image_id_mapping.json and data_split.json")
        return 0

    write_outputs(base_dir, processed, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
