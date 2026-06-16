import argparse
import csv
from pathlib import Path
import numpy as np


def _parse_int(value):
    if value is None:
        return None
    v = str(value).strip()
    if v in {"", "--", "'--"}:
        return None
    try:
        return int(float(v))
    except Exception:
        return None


def _load_clinical(data_filepath):
    info = {}
    with open(data_filepath, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_id = (row.get("case_submitter_id") or row.get("case_id") or "").strip()
            if not case_id:
                continue
            death_occurred = 1 if (row.get("vital_status") or "").strip().lower() == "dead" else 0
            days_to_death = _parse_int(row.get("days_to_death"))
            days_to_last = _parse_int(row.get("days_to_last_follow_up") or row.get("days_to_last_known_disease_status"))
            survival_time = days_to_death if days_to_death is not None else days_to_last
            if survival_time is None:
                continue
            if case_id not in info:
                info[case_id] = (survival_time, death_occurred)
    return info


def _case_id_from_mask_dir(name):
    parts = name.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return name


def generate_annotations(image_dir, data_filepath, out_path, *, val_ratio=0.1, seed=1):
    image_dir = Path(image_dir)
    clinical_map = _load_clinical(data_filepath)
    rows = []
    for p in sorted([d for d in image_dir.iterdir() if d.is_dir()]):
        case_id = _case_id_from_mask_dir(p.name)
        hit = clinical_map.get(case_id)
        if hit is None:
            continue
        survival_time, death_occurred = hit
        rows.append([str(p), int(survival_time), int(death_occurred)])

    count = len(rows)
    if count:
        val_ratio = float(val_ratio)
        if val_ratio < 0.0:
            val_ratio = 0.0
        if val_ratio > 1.0:
            val_ratio = 1.0
        indices = np.arange(count, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
        split_at = int(np.floor(count * (1.0 - val_ratio)))
        val_indices = set(indices[split_at:].tolist())
    else:
        val_indices = set()

    out_path = Path(out_path)
    with open(out_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_dir", "survival_time", "death_occurred", "set"])
        for i, row in enumerate(rows):
            split = "val" if i in val_indices else "train"
            writer.writerow([row[0], row[1], row[2], split])
    print(f"Finished generating annotations: {out_path} ({len(rows)} rows)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/TCGA_WSI_masks")
    parser.add_argument("--clinical-csv", default="data/clinical.csv")
    parser.add_argument("--output", default="annotations.csv")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    generate_annotations(
        args.image_dir,
        args.clinical_csv,
        args.output,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
