import argparse
import json
import random
from pathlib import Path


PD1_ALIASES = ["pd-1", "pd1", "pd_1", "PD-1", "PD1", "PD_1"]
REQUIRED_MASKS = ["cd4", "cd68", "ck", "actin", "cd8", "tissue"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", type=Path, default=Path("data_f/TCGA_WSI_masks"))
    parser.add_argument("--output-json", type=Path, default=Path("wsi_process/data_split.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def _has_file(case_dir: Path, name: str) -> bool:
    candidates = [
        case_dir / f"{name}.png",
        case_dir / f"{name.lower()}.png",
        case_dir / f"{name.upper()}.png",
        case_dir / f"{name}_mask.png",
        case_dir / f"{name.lower()}_mask.png",
        case_dir / f"{name.upper()}_mask.png",
    ]
    return any(p.exists() and p.is_file() for p in candidates)


def _has_pd1(case_dir: Path) -> bool:
    return any(_has_file(case_dir, name) for name in PD1_ALIASES)


def _is_valid_case_dir(case_dir: Path) -> bool:
    if not case_dir.is_dir():
        return False
    if not _has_pd1(case_dir):
        return False
    for name in REQUIRED_MASKS:
        if not _has_file(case_dir, name):
            return False
    return True


def collect_case_ids(masks_dir: Path) -> list[str]:
    case_ids = []
    for p in sorted(masks_dir.iterdir(), key=lambda x: x.name):
        if _is_valid_case_dir(p):
            case_ids.append(p.name)
    return case_ids


def _group_key(case_id: str) -> str:
    parts = case_id.split("-")
    if len(parts) >= 5:
        return "-".join(parts[:5])
    return case_id


def _build_grouped_case_ids(case_ids: list[str]) -> dict[str, list[str]]:
    grouped = {}
    for case_id in case_ids:
        key = _group_key(case_id)
        grouped.setdefault(key, []).append(case_id)
    for key in grouped:
        grouped[key] = sorted(grouped[key])
    return grouped


def split_case_ids(case_ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not case_ids:
        raise ValueError("No valid case_id directories found for splitting.")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0, 1).")
    grouped = _build_grouped_case_ids(case_ids)
    group_keys = list(grouped.keys())
    random.Random(seed).shuffle(group_keys)

    if len(group_keys) == 1 or val_ratio == 0:
        return sorted(case_ids), []

    total_cases = len(case_ids)
    target_val_count = int(round(total_cases * float(val_ratio)))
    if target_val_count <= 0:
        target_val_count = 1
    if target_val_count >= total_cases:
        target_val_count = total_cases - 1

    cumulative = []
    running = 0
    for key in group_keys:
        running += len(grouped[key])
        cumulative.append(running)

    best_k = 1
    best_diff = abs(cumulative[0] - target_val_count)
    for i in range(1, len(group_keys) - 1):
        diff = abs(cumulative[i] - target_val_count)
        if diff < best_diff:
            best_diff = diff
            best_k = i + 1

    val_group_keys = set(group_keys[:best_k])
    val_ids = []
    train_ids = []
    for key, members in grouped.items():
        if key in val_group_keys:
            val_ids.extend(members)
        else:
            train_ids.extend(members)

    return sorted(train_ids), sorted(val_ids)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.masks_dir.exists() or not args.masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir not found or not a directory: {args.masks_dir}")

    case_ids = collect_case_ids(args.masks_dir)
    train_ids, val_ids = split_case_ids(case_ids, val_ratio=float(args.val_ratio), seed=int(args.seed))
    payload = {
        "train": train_ids,
        "val": val_ids,
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "num_cases": len(case_ids),
        "grouping": "first_five_dash_tokens",
        "num_groups": len(_build_grouped_case_ids(case_ids)),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved split to: {args.output_json}")
    print(f"train={len(train_ids)} val={len(val_ids)} total={len(case_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
