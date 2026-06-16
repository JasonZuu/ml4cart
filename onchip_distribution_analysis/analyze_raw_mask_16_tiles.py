"""Fast native-resolution raw mask analysis using 16x16 tiles.

This script computes only the sample-level quantities needed for the
low-actin versus high-actin figures/report, avoiding the very large tile-level
CSV that would result from storing every 16x16 native-resolution tile.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from common.pdochange_data import load_split_json
from onchip_distribution_analysis.analyze_caf_cd8_local import (
    _build_case_dir_index,
    _load_binary_mask,
    _resolve_case_dir,
    _resolve_mask_path,
)
from onchip_distribution_analysis.analyze_caf_cd8_local_combined import (
    _ck_stratified_delta,
    _partial_spearman_given_ck,
    _residualize_on_covariate,
)


@dataclass(frozen=True)
class CohortSpec:
    name: str
    masks_dir: Path
    split_json: Path


COHORTS = {
    "r1": CohortSpec(
        name="r1",
        masks_dir=Path("data/On-chip_Data"),
        split_json=Path("data/On-chip_Data/data_split.json"),
    ),
    "r2": CohortSpec(
        name="r2",
        masks_dir=Path("data/On-chip_Data_R2"),
        split_json=Path("data/On-chip_Data_R2/data_split.json"),
    ),
}

SAMPLE_FIELDS = [
    "round",
    "source_image_id",
    "split",
    "tile_size",
    "n_tiles",
    "n_low_actin_tiles",
    "n_high_actin_tiles",
    "mean_actin_frac",
    "mean_cd8_frac",
    "mean_ck_frac",
    "spearman_actin_cd8",
    "partial_spearman_actin_cd8_given_ck",
    "cd8_low_actin_q1_mean",
    "cd8_high_actin_q4_mean",
    "delta_high_minus_low",
    "residual_cd8_low_actin_q1_mean",
    "residual_cd8_high_actin_q4_mean",
    "residual_delta_high_minus_low",
    "ck_stratified_delta_high_minus_low",
    "n_ck_strata",
    "n_ck_stratified_tiles",
]

COHORT_FIELDS = [
    "round",
    "metric",
    "alternative",
    "n_samples",
    "mean",
    "median",
    "sd",
    "sem",
    "wilcoxon_statistic",
    "wilcoxon_p",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze raw masks with native-resolution 16x16 tiles.")
    parser.add_argument("--cohort", choices=sorted(COHORTS), required=True)
    parser.add_argument("--masks-dir", type=Path, default=None, help="Override the selected cohort mask directory.")
    parser.add_argument("--split-json", type=Path, default=None, help="Override the selected cohort split JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["all"], choices=["train", "val", "test", "all"])
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--caf-channel", default="actin")
    parser.add_argument("--cart-channel", default="cd8")
    parser.add_argument("--tumor-channel", default="ck")
    return parser.parse_args(argv)


def selected_splits(names: list[str]) -> list[str]:
    if "all" in names:
        return ["train", "val", "test"]
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def block_fraction(mask: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = mask.shape
    h_crop = (h // tile_size) * tile_size
    w_crop = (w // tile_size) * tile_size
    if h_crop == 0 or w_crop == 0:
        raise ValueError(f"Mask too small for tile_size={tile_size}: shape={mask.shape}")
    cropped = mask[:h_crop, :w_crop]
    blocks = cropped.reshape(h_crop // tile_size, tile_size, w_crop // tile_size, tile_size)
    return blocks.mean(axis=(1, 3)).reshape(-1).astype(np.float64)


def safe_spearman(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def safe_mean(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def summarize_sample(round_name: str, image_id: str, split_name: str, actin: np.ndarray, cd8: np.ndarray, ck: np.ndarray, tile_size: int) -> dict:
    q25, q75 = np.quantile(actin, [0.25, 0.75])
    low_sel = actin <= q25
    high_sel = actin >= q75
    cd8_low = safe_mean(cd8[low_sel])
    cd8_high = safe_mean(cd8[high_sel])
    residual = _residualize_on_covariate(cd8, ck)
    residual_low = safe_mean(residual[low_sel])
    residual_high = safe_mean(residual[high_sel])
    stratified_delta, n_strata, n_stratified_tiles = _ck_stratified_delta(actin, cd8, ck)
    return {
        "round": round_name,
        "source_image_id": image_id,
        "split": split_name,
        "tile_size": int(tile_size),
        "n_tiles": int(actin.size),
        "n_low_actin_tiles": int(low_sel.sum()),
        "n_high_actin_tiles": int(high_sel.sum()),
        "mean_actin_frac": float(actin.mean()),
        "mean_cd8_frac": float(cd8.mean()),
        "mean_ck_frac": float(ck.mean()),
        "spearman_actin_cd8": safe_spearman(actin, cd8),
        "partial_spearman_actin_cd8_given_ck": _partial_spearman_given_ck(actin, cd8, ck),
        "cd8_low_actin_q1_mean": cd8_low,
        "cd8_high_actin_q4_mean": cd8_high,
        "delta_high_minus_low": float(cd8_high - cd8_low),
        "residual_cd8_low_actin_q1_mean": residual_low,
        "residual_cd8_high_actin_q4_mean": residual_high,
        "residual_delta_high_minus_low": float(residual_high - residual_low),
        "ck_stratified_delta_high_minus_low": stratified_delta,
        "n_ck_strata": int(n_strata),
        "n_ck_stratified_tiles": int(n_stratified_tiles),
    }


def wilcoxon_row(round_name: str, metric: str, values: list[float], alternative: str = "less") -> dict:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        stat, p_val = float("nan"), float("nan")
    elif np.allclose(vals, 0.0):
        stat, p_val = 0.0, 1.0
    else:
        stat, p_val = wilcoxon(vals, alternative=alternative)
        stat, p_val = float(stat), float(p_val)
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    sem = float(sd / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return {
        "round": round_name,
        "metric": metric,
        "alternative": alternative,
        "n_samples": int(vals.size),
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "median": float(np.median(vals)) if vals.size else float("nan"),
        "sd": sd,
        "sem": sem,
        "wilcoxon_statistic": stat,
        "wilcoxon_p": p_val,
    }


def build_cohort_stats(round_name: str, sample_rows: list[dict]) -> list[dict]:
    metrics = [
        "spearman_actin_cd8",
        "partial_spearman_actin_cd8_given_ck",
        "delta_high_minus_low",
        "residual_delta_high_minus_low",
        "ck_stratified_delta_high_minus_low",
    ]
    return [
        wilcoxon_row(round_name, metric, [row[metric] for row in sample_rows], "less")
        for metric in metrics
    ]


def format_value(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isnan(v):
            return ""
        return f"{v:.10g}"
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fieldnames})


def main(argv=None) -> int:
    args = parse_args(argv)
    cohort = COHORTS[args.cohort]
    if args.masks_dir is not None or args.split_json is not None:
        cohort = CohortSpec(
            name=cohort.name,
            masks_dir=args.masks_dir or cohort.masks_dir,
            split_json=args.split_json or cohort.split_json,
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_payload = load_split_json(cohort.split_json)
    image_split_pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for split_name in selected_splits(args.splits):
        for image_id in split_payload.get(split_name, []) or []:
            image_id = str(image_id)
            if image_id in seen:
                continue
            seen.add(image_id)
            image_split_pairs.append((image_id, split_name))
    if not image_split_pairs:
        raise ValueError(f"No samples selected for {args.cohort}")

    case_index = _build_case_dir_index(cohort.masks_dir)
    sample_rows: list[dict] = []
    for idx, (image_id, split_name) in enumerate(image_split_pairs, start=1):
        case_dir = _resolve_case_dir(cohort.masks_dir, case_index, image_id)
        paths = {
            "actin": _resolve_mask_path(case_dir, args.caf_channel),
            "cd8": _resolve_mask_path(case_dir, args.cart_channel),
            "ck": _resolve_mask_path(case_dir, args.tumor_channel),
        }
        missing = [name for name, path in paths.items() if path is None]
        if missing:
            raise FileNotFoundError(f"{args.cohort}:{image_id} missing masks: {missing} in {case_dir}")
        masks = {
            name: _load_binary_mask(path, float(args.mask_threshold))
            for name, path in paths.items()
        }
        shapes = {mask.shape for mask in masks.values()}
        if len(shapes) != 1:
            raise ValueError(f"{args.cohort}:{image_id} mask shape mismatch: {sorted(shapes)}")
        actin = block_fraction(masks["actin"], int(args.tile_size))
        cd8 = block_fraction(masks["cd8"], int(args.tile_size))
        ck = block_fraction(masks["ck"], int(args.tile_size))
        sample_rows.append(summarize_sample(args.cohort, image_id, split_name, actin, cd8, ck, int(args.tile_size)))
        print(f"Processed {args.cohort} {idx}/{len(image_split_pairs)}: {image_id}", flush=True)

    cohort_rows = build_cohort_stats(args.cohort, sample_rows)
    write_csv(out_dir / "raw16_sample_stats.csv", sample_rows, SAMPLE_FIELDS)
    write_csv(out_dir / "raw16_cohort_stats.csv", cohort_rows, COHORT_FIELDS)
    with (out_dir / "raw16_cohort_stats.json").open("w", encoding="utf-8") as f:
        json.dump([{k: format_value(v) for k, v in row.items()} for row in cohort_rows], f, indent=2)

    delta = next(row for row in cohort_rows if row["metric"] == "delta_high_minus_low")
    partial = next(row for row in cohort_rows if row["metric"] == "partial_spearman_actin_cd8_given_ck")
    print()
    print(f"Analyzed {len(sample_rows)} {args.cohort} samples with {args.tile_size}x{args.tile_size} native tiles.")
    print(
        "Raw 16px endpoint: "
        f"delta median={format_value(delta['median'])}, p<0={format_value(delta['wilcoxon_p'])}; "
        f"partial rho median={format_value(partial['median'])}, p<0={format_value(partial['wilcoxon_p'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
