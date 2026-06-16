"""Local CAF-CD8 spatial analysis for R2 on-chip masks.

The primary analysis tests whether actin-high local regions have lower CD8
signal. Actin is treated as the CAF/stroma proxy and CD8 as the CAR-T proxy.

Outputs are written under:
  onchip_pdochange_prediction/results/r2/optimal/caf_cd8_local_analysis

Usage:
    python onchip_pdochange_prediction/analyze_caf_cd8_local.py
    python onchip_pdochange_prediction/analyze_caf_cd8_local.py --limit 2 --tile-sizes 512 --no-plots
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr, wilcoxon

from common.pdochange_data import (
    load_pdo_change_labels,
    load_split_json,
    pdo_change_bin_index_to_label,
    pdo_change_to_bin_index,
)


PDO_BIN_MIDPOINTS = {
    "x<-100": -110.0,
    "-100<=x<-80": -90.0,
    "-80<=x<-60": -70.0,
    "-60<=x<-40": -50.0,
    "-40<=x<-20": -30.0,
    "-20<=x<0": -10.0,
    "0<=x<20": 10.0,
    "20<=x<40": 30.0,
    "40<=x<60": 50.0,
    "60<=x<80": 70.0,
    "80<=x<100": 90.0,
    "x>=100": 110.0,
}

TILE_FIELDS = [
    "image_id",
    "split",
    "pdo_change",
    "pdo_bin_idx",
    "pdo_bin_label",
    "pdo_midpoint",
    "tile_size",
    "tile_row",
    "tile_col",
    "y0",
    "x0",
    "actin_frac",
    "cd8_frac",
    "ck_frac",
]

SAMPLE_FIELDS = [
    "image_id",
    "split",
    "pdo_change",
    "pdo_bin_idx",
    "pdo_bin_label",
    "pdo_midpoint",
    "tile_size",
    "endpoint",
    "n_tiles",
    "mean_actin_frac",
    "mean_cd8_frac",
    "mean_ck_frac",
    "spearman_rho",
    "spearman_p",
    "actin_q25",
    "actin_q75",
    "n_low_actin_tiles",
    "n_high_actin_tiles",
    "cd8_low_actin_q1_mean",
    "cd8_high_actin_q4_mean",
    "delta_high_minus_low",
]

DISTANCE_FIELDS = [
    "image_id",
    "split",
    "pdo_change",
    "pdo_bin_idx",
    "pdo_bin_label",
    "pdo_midpoint",
    "band_index",
    "band_label",
    "distance_lo",
    "distance_hi",
    "pixel_count",
    "cd8_frac",
]

DISTANCE_SAMPLE_FIELDS = [
    "image_id",
    "split",
    "pdo_change",
    "pdo_bin_idx",
    "pdo_bin_label",
    "pdo_midpoint",
    "endpoint",
    "n_bands",
    "distance_spearman_rho",
    "distance_spearman_p",
    "cd8_nearest_band",
    "cd8_farthest_band",
    "delta_far_minus_near",
]

COHORT_FIELDS = [
    "analysis",
    "endpoint",
    "tile_size",
    "metric",
    "alternative",
    "n_samples",
    "mean",
    "median",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "wilcoxon_statistic",
    "wilcoxon_p",
    "supports_hypothesis",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local CAF-CD8 analysis for R2 masks")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data_R2"))
    parser.add_argument("--split-json", type=Path, default=Path("data/On-chip_Data_R2/data_split.json"))
    parser.add_argument("--label-json", type=Path, default=Path("data/On-chip_Data_R2/pdo_change_label.json"))
    parser.add_argument(
        "--gradcam-summary",
        type=Path,
        default=Path("onchip_pdochange_prediction/results/r2/optimal/val_analysis/gradcam_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onchip_pdochange_prediction/results/r2/optimal/caf_cd8_local_analysis"),
    )
    parser.add_argument("--caf-channel", type=str, default="actin")
    parser.add_argument("--cart-channel", type=str, default="cd8")
    parser.add_argument("--tumor-channel", type=str, default="ck")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--primary-tile-size", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--tumor-min-frac", type=float, default=0.0)
    parser.add_argument("--distance-bands", nargs="+", type=int, default=[0, 128, 256, 512, 1024])
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


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


def _build_case_dir_index(masks_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks_dir not found: {masks_dir}")
    for p in sorted(masks_dir.rglob("*")):
        if not p.is_dir():
            continue
        key = _normalize_key(p.name)
        if key not in index:
            index[key] = p
    return index


def _resolve_case_dir(masks_dir: Path, index: dict[str, Path], image_id: str) -> Path:
    direct = masks_dir / str(image_id)
    if direct.exists() and direct.is_dir():
        return direct
    resolved = index.get(_normalize_key(image_id))
    if resolved is None:
        raise FileNotFoundError(f"Could not resolve case dir for {image_id}")
    return resolved


def _threshold_to_pixel_value(threshold: float) -> float:
    return float(threshold) * 255.0 if float(threshold) <= 1.0 else float(threshold)


def _load_binary_mask(path: Path, threshold: float) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0, 255}):
        preview = sorted(unique)[:12]
        raise ValueError(f"Mask is not binary 0/255: {path} values={preview}")
    return arr > _threshold_to_pixel_value(threshold)


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _sample_meta(image_id: str, split_name: str, labels: dict[str, float]) -> dict:
    pdo_change = _float_or_nan(labels.get(image_id, float("nan")))
    if math.isfinite(pdo_change):
        bin_idx = int(pdo_change_to_bin_index(pdo_change))
        bin_label = pdo_change_bin_index_to_label(bin_idx)
    else:
        bin_idx = -1
        bin_label = ""
    return {
        "image_id": image_id,
        "split": split_name,
        "pdo_change": pdo_change,
        "pdo_bin_idx": bin_idx,
        "pdo_bin_label": bin_label,
        "pdo_midpoint": PDO_BIN_MIDPOINTS.get(bin_label, float("nan")),
    }


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_spearman(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    rho, p_val = spearmanr(x, y)
    return float(rho), float(p_val)


def _safe_wilcoxon(values: np.ndarray, alternative: str) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if np.allclose(vals, 0.0):
        return 0.0, 1.0
    try:
        stat, p_val = wilcoxon(vals, alternative=alternative)
    except ValueError:
        return float("nan"), float("nan")
    return float(stat), float(p_val)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_iter: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    idx = rng.integers(0, vals.size, size=(int(n_iter), vals.size))
    medians = np.median(vals[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def _format_csv_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if math.isnan(v):
            return ""
        return f"{v:.10g}"
    return value


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_csv_value(row.get(field, "")) for field in fieldnames})


def _compute_tile_rows(
    meta: dict,
    caf_mask: np.ndarray,
    cart_mask: np.ndarray,
    tumor_mask: np.ndarray,
    tile_sizes: list[int],
) -> list[dict]:
    rows: list[dict] = []
    height, width = caf_mask.shape
    for tile_size in tile_sizes:
        n_rows = height // int(tile_size)
        n_cols = width // int(tile_size)
        for tile_row in range(n_rows):
            y0 = tile_row * int(tile_size)
            y1 = y0 + int(tile_size)
            for tile_col in range(n_cols):
                x0 = tile_col * int(tile_size)
                x1 = x0 + int(tile_size)
                caf_tile = caf_mask[y0:y1, x0:x1]
                cart_tile = cart_mask[y0:y1, x0:x1]
                tumor_tile = tumor_mask[y0:y1, x0:x1]
                rows.append({
                    **meta,
                    "tile_size": int(tile_size),
                    "tile_row": int(tile_row),
                    "tile_col": int(tile_col),
                    "y0": int(y0),
                    "x0": int(x0),
                    "actin_frac": float(caf_tile.mean()),
                    "cd8_frac": float(cart_tile.mean()),
                    "ck_frac": float(tumor_tile.mean()),
                })
    return rows


def _summarize_tiles(meta: dict, tile_rows: list[dict], endpoint: str) -> dict:
    if not tile_rows:
        return {
            **meta,
            "tile_size": "",
            "endpoint": endpoint,
            "n_tiles": 0,
            "mean_actin_frac": float("nan"),
            "mean_cd8_frac": float("nan"),
            "mean_ck_frac": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "actin_q25": float("nan"),
            "actin_q75": float("nan"),
            "n_low_actin_tiles": 0,
            "n_high_actin_tiles": 0,
            "cd8_low_actin_q1_mean": float("nan"),
            "cd8_high_actin_q4_mean": float("nan"),
            "delta_high_minus_low": float("nan"),
        }

    tile_size = int(tile_rows[0]["tile_size"])
    actin = np.array([r["actin_frac"] for r in tile_rows], dtype=np.float64)
    cd8 = np.array([r["cd8_frac"] for r in tile_rows], dtype=np.float64)
    ck = np.array([r["ck_frac"] for r in tile_rows], dtype=np.float64)
    rho, p_val = _safe_spearman(actin, cd8)
    q25, q75 = np.quantile(actin, [0.25, 0.75])
    low = cd8[actin <= q25]
    high = cd8[actin >= q75]
    low_mean = _safe_mean(low)
    high_mean = _safe_mean(high)
    return {
        **meta,
        "tile_size": tile_size,
        "endpoint": endpoint,
        "n_tiles": int(len(tile_rows)),
        "mean_actin_frac": float(np.mean(actin)),
        "mean_cd8_frac": float(np.mean(cd8)),
        "mean_ck_frac": float(np.mean(ck)),
        "spearman_rho": rho,
        "spearman_p": p_val,
        "actin_q25": float(q25),
        "actin_q75": float(q75),
        "n_low_actin_tiles": int(low.size),
        "n_high_actin_tiles": int(high.size),
        "cd8_low_actin_q1_mean": low_mean,
        "cd8_high_actin_q4_mean": high_mean,
        "delta_high_minus_low": float(high_mean - low_mean) if math.isfinite(high_mean) and math.isfinite(low_mean) else float("nan"),
    }


def _distance_band_rows(meta: dict, caf_mask: np.ndarray, cart_mask: np.ndarray, bands: list[int]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    if not caf_mask.any():
        return rows, {
            **meta,
            "endpoint": "distance_from_actin",
            "n_bands": 0,
            "distance_spearman_rho": float("nan"),
            "distance_spearman_p": float("nan"),
            "cd8_nearest_band": float("nan"),
            "cd8_farthest_band": float("nan"),
            "delta_far_minus_near": float("nan"),
        }

    dist = distance_transform_edt(~caf_mask)
    band_edges = sorted({int(x) for x in bands})
    if not band_edges or band_edges[0] != 0:
        band_edges = [0, *band_edges]

    for idx, lo in enumerate(band_edges):
        hi = band_edges[idx + 1] if idx + 1 < len(band_edges) else math.inf
        if math.isinf(hi):
            selector = dist >= lo
            label = f">={lo}"
            hi_value = ""
        else:
            selector = (dist >= lo) & (dist < hi)
            label = f"{lo}-{hi}"
            hi_value = int(hi)
        count = int(selector.sum())
        cd8_frac = float(cart_mask[selector].mean()) if count > 0 else float("nan")
        rows.append({
            **meta,
            "band_index": int(idx),
            "band_label": label,
            "distance_lo": int(lo),
            "distance_hi": hi_value,
            "pixel_count": count,
            "cd8_frac": cd8_frac,
        })

    finite_rows = [r for r in rows if math.isfinite(float(r["cd8_frac"]))]
    band_idx = np.array([r["band_index"] for r in finite_rows], dtype=np.float64)
    cd8 = np.array([r["cd8_frac"] for r in finite_rows], dtype=np.float64)
    rho, p_val = _safe_spearman(band_idx, cd8)
    near = float(cd8[0]) if cd8.size else float("nan")
    far = float(cd8[-1]) if cd8.size else float("nan")
    stat_row = {
        **meta,
        "endpoint": "distance_from_actin",
        "n_bands": int(cd8.size),
        "distance_spearman_rho": rho,
        "distance_spearman_p": p_val,
        "cd8_nearest_band": near,
        "cd8_farthest_band": far,
        "delta_far_minus_near": float(far - near) if math.isfinite(far) and math.isfinite(near) else float("nan"),
    }
    return rows, stat_row


def _cohort_row(
    analysis: str,
    endpoint: str,
    tile_size,
    metric: str,
    values: list[float],
    alternative: str,
    rng: np.random.Generator,
    bootstrap_iterations: int,
) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    stat, p_val = _safe_wilcoxon(vals, alternative=alternative)
    ci_low, ci_high = _bootstrap_ci(vals, rng, bootstrap_iterations)
    if alternative == "less":
        supported = bool(vals.size and math.isfinite(ci_high) and ci_high < 0 and math.isfinite(p_val) and p_val < 0.05)
    else:
        supported = bool(vals.size and math.isfinite(ci_low) and ci_low > 0 and math.isfinite(p_val) and p_val < 0.05)
    return {
        "analysis": analysis,
        "endpoint": endpoint,
        "tile_size": tile_size,
        "metric": metric,
        "alternative": alternative,
        "n_samples": int(vals.size),
        "mean": float(np.mean(vals)) if vals.size else float("nan"),
        "median": float(np.median(vals)) if vals.size else float("nan"),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "wilcoxon_statistic": stat,
        "wilcoxon_p": p_val,
        "supports_hypothesis": supported,
    }


def _build_cohort_stats(
    sample_rows: list[dict],
    distance_sample_rows: list[dict],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict]:
    rng = np.random.default_rng(int(bootstrap_seed))
    out: list[dict] = []
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in sample_rows:
        grouped.setdefault((str(row["endpoint"]), int(row["tile_size"])), []).append(row)
    for (endpoint, tile_size), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append(_cohort_row(
            "tile",
            endpoint,
            tile_size,
            "spearman_rho",
            [r["spearman_rho"] for r in rows],
            "less",
            rng,
            bootstrap_iterations,
        ))
        out.append(_cohort_row(
            "tile",
            endpoint,
            tile_size,
            "delta_high_minus_low",
            [r["delta_high_minus_low"] for r in rows],
            "less",
            rng,
            bootstrap_iterations,
        ))

    out.append(_cohort_row(
        "distance",
        "distance_from_actin",
        "",
        "distance_spearman_rho",
        [r["distance_spearman_rho"] for r in distance_sample_rows],
        "greater",
        rng,
        bootstrap_iterations,
    ))
    out.append(_cohort_row(
        "distance",
        "distance_from_actin",
        "",
        "delta_far_minus_near",
        [r["delta_far_minus_near"] for r in distance_sample_rows],
        "greater",
        rng,
        bootstrap_iterations,
    ))
    return out


def _read_gradcam_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_gradcam_secondary(gradcam_path: Path, primary_sample_rows: list[dict]) -> list[dict]:
    gradcam_rows = _read_gradcam_summary(gradcam_path)
    primary_by_id = {r["image_id"]: r for r in primary_sample_rows}
    out: list[dict] = []
    for g_row in gradcam_rows:
        image_id = g_row.get("image_id", "")
        sample_row = primary_by_id.get(image_id)
        if sample_row is None:
            continue
        merged = {
            "image_id": image_id,
            "split": sample_row["split"],
            "pdo_change": sample_row["pdo_change"],
            "pdo_bin_idx": sample_row["pdo_bin_idx"],
            "pdo_bin_label": sample_row["pdo_bin_label"],
            "tile_size": sample_row["tile_size"],
            "sample_spearman_rho": sample_row["spearman_rho"],
            "sample_delta_high_minus_low": sample_row["delta_high_minus_low"],
            "sample_mean_actin_frac": sample_row["mean_actin_frac"],
            "sample_mean_cd8_frac": sample_row["mean_cd8_frac"],
        }
        for key in ("actin_cam_mean", "actin_cam_hi_frac", "cd8_cam_mean", "cd8_cam_hi_frac"):
            merged[key] = _float_or_nan(g_row.get(key, float("nan")))
        out.append(merged)
    return out


def _validate_primary_tile_counts(tile_rows: list[dict], meta_by_id: dict[str, dict], primary_tile_size: int) -> None:
    counts: dict[str, int] = {}
    for row in tile_rows:
        if int(row["tile_size"]) != int(primary_tile_size):
            continue
        counts[row["image_id"]] = counts.get(row["image_id"], 0) + 1
    failures = []
    for image_id, meta in meta_by_id.items():
        expected = (int(meta["height"]) // int(primary_tile_size)) * (int(meta["width"]) // int(primary_tile_size))
        actual = counts.get(image_id, 0)
        if actual != expected:
            failures.append(f"{image_id}: expected {expected}, got {actual}")
    if failures:
        raise ValueError("Primary tile count validation failed: " + "; ".join(failures))


def _plot_tile_hexbin(tile_rows: list[dict], primary_tile_size: int, out_dir: Path) -> None:
    rows = [r for r in tile_rows if int(r["tile_size"]) == int(primary_tile_size)]
    x = np.array([r["actin_frac"] for r in rows], dtype=np.float64)
    y = np.array([r["cd8_frac"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    hb = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis")
    ax.set_xlabel(f"Actin fraction per {primary_tile_size}px tile")
    ax.set_ylabel(f"CD8 fraction per {primary_tile_size}px tile")
    ax.set_title("Local actin vs CD8 density")
    ax.grid(True, alpha=0.2)
    fig.colorbar(hb, ax=ax, label="Tile count")
    fig.tight_layout()
    fig.savefig(out_dir / f"tile_scatter_hexbin_{primary_tile_size}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_sample_rho(sample_rows: list[dict], primary_tile_size: int, out_dir: Path) -> None:
    rows = [
        r for r in sample_rows
        if int(r["tile_size"]) == int(primary_tile_size) and r["endpoint"] == "raw_all_tiles"
    ]
    rows = sorted(rows, key=lambda r: (float("inf") if math.isnan(float(r["spearman_rho"])) else float(r["spearman_rho"])))
    y = np.arange(len(rows))
    x = np.array([r["spearman_rho"] for r in rows], dtype=np.float64)
    labels = [str(r["image_id"]).replace("chip-r1_", "") for r in rows]
    fig, ax = plt.subplots(figsize=(6, max(4, len(rows) * 0.18)), dpi=140)
    ax.scatter(x, y, s=24)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Per-sample Spearman rho (actin fraction vs CD8 fraction)")
    ax.set_title(f"Sample-level local association ({primary_tile_size}px tiles)")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"sample_rho_forest_{primary_tile_size}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_high_low(sample_rows: list[dict], primary_tile_size: int, out_dir: Path) -> None:
    rows = [
        r for r in sample_rows
        if int(r["tile_size"]) == int(primary_tile_size) and r["endpoint"] == "raw_all_tiles"
    ]
    low = np.array([r["cd8_low_actin_q1_mean"] for r in rows], dtype=np.float64)
    high = np.array([r["cd8_high_actin_q4_mean"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.boxplot([low, high], tick_labels=["Low actin Q1", "High actin Q4"], showfliers=False)
    for lo, hi in zip(low, high):
        if math.isfinite(float(lo)) and math.isfinite(float(hi)):
            ax.plot([1, 2], [lo, hi], color="0.65", linewidth=0.8, alpha=0.7)
    ax.set_ylabel("Mean CD8 fraction")
    ax.set_title(f"CD8 in low vs high actin tiles ({primary_tile_size}px)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"high_low_cd8_boxplot_{primary_tile_size}.png", bbox_inches="tight")
    plt.close(fig)


def _plot_distance_gradient(distance_rows: list[dict], out_dir: Path) -> None:
    grouped: dict[int, list[dict]] = {}
    for row in distance_rows:
        grouped.setdefault(int(row["band_index"]), []).append(row)
    labels = []
    means = []
    sems = []
    for band_idx in sorted(grouped):
        rows = grouped[band_idx]
        vals = np.array([r["cd8_frac"] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        labels.append(str(rows[0]["band_label"]))
        means.append(float(np.mean(vals)) if vals.size else float("nan"))
        sems.append(float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.errorbar(np.arange(len(labels)), means, yerr=sems, marker="o", capsize=3)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Distance from actin-positive pixels (px)")
    ax.set_ylabel("Mean CD8 fraction")
    ax.set_title("CD8 density by distance from actin")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "distance_gradient.png", bbox_inches="tight")
    plt.close(fig)


def _plot_gradcam_secondary(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    labels = [str(r["image_id"]).replace("chip-r1_", "") for r in rows]
    x = np.arange(len(rows))
    actin = np.array([r.get("actin_cam_mean", float("nan")) for r in rows], dtype=np.float64)
    cd8 = np.array([r.get("cd8_cam_mean", float("nan")) for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.75), 4), dpi=140)
    width = 0.38
    ax.bar(x - width / 2, actin, width=width, label="Actin CAM mean")
    ax.bar(x + width / 2, cd8, width=width, label="CD8 CAM mean")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("CAM mean inside mask")
    ax.set_title("Secondary SegX-GradCAM channel attention")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_secondary_summary.png", bbox_inches="tight")
    plt.close(fig)


def _make_plots(
    tile_rows: list[dict],
    sample_rows: list[dict],
    distance_rows: list[dict],
    gradcam_rows: list[dict],
    primary_tile_size: int,
    out_dir: Path,
) -> None:
    _plot_tile_hexbin(tile_rows, primary_tile_size, out_dir)
    _plot_sample_rho(sample_rows, primary_tile_size, out_dir)
    _plot_high_low(sample_rows, primary_tile_size, out_dir)
    _plot_distance_gradient(distance_rows, out_dir)
    _plot_gradcam_secondary(gradcam_rows, out_dir)


def _cohort_lookup(rows: list[dict], endpoint: str, tile_size, metric: str) -> dict | None:
    for row in rows:
        if row["endpoint"] == endpoint and str(row["tile_size"]) == str(tile_size) and row["metric"] == metric:
            return row
    return None


def _fmt_num(value, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}g}"


def _gradcam_corr_lines(gradcam_rows: list[dict]) -> list[str]:
    if not gradcam_rows:
        return ["- GradCAM summary was not available or had no matching samples."]
    lines = []
    spatial_metrics = ["sample_spearman_rho", "sample_delta_high_minus_low"]
    cam_metrics = ["actin_cam_mean", "actin_cam_hi_frac", "cd8_cam_mean", "cd8_cam_hi_frac"]
    for spatial in spatial_metrics:
        for cam in cam_metrics:
            x = np.array([r.get(spatial, float("nan")) for r in gradcam_rows], dtype=np.float64)
            y = np.array([r.get(cam, float("nan")) for r in gradcam_rows], dtype=np.float64)
            rho, p_val = _safe_spearman(x, y)
            lines.append(f"- Spearman({spatial}, {cam}) = {_fmt_num(rho)}, p = {_fmt_num(p_val)}")
    return lines


def _write_report(
    out_path: Path,
    args,
    n_samples: int,
    cohort_rows: list[dict],
    gradcam_rows: list[dict],
) -> None:
    primary_rho = _cohort_lookup(cohort_rows, "raw_all_tiles", int(args.primary_tile_size), "spearman_rho")
    primary_delta = _cohort_lookup(cohort_rows, "raw_all_tiles", int(args.primary_tile_size), "delta_high_minus_low")
    tumor_rho = _cohort_lookup(cohort_rows, "tumor_context_tiles", int(args.primary_tile_size), "spearman_rho")
    tumor_delta = _cohort_lookup(cohort_rows, "tumor_context_tiles", int(args.primary_tile_size), "delta_high_minus_low")
    distance_rho = _cohort_lookup(cohort_rows, "distance_from_actin", "", "distance_spearman_rho")
    distance_delta = _cohort_lookup(cohort_rows, "distance_from_actin", "", "delta_far_minus_near")

    primary_supported = bool(
        primary_rho
        and primary_delta
        and primary_rho["supports_hypothesis"]
        and primary_delta["supports_hypothesis"]
    )
    if primary_supported:
        interpretation = "The primary raw local CD8 endpoint supports CAF-high/CD8-low."
    else:
        interpretation = "The primary raw local CD8 endpoint does not support CAF-high/CD8-low."

    lines = [
        "# Local CAF-CD8 Quantitative Analysis",
        "",
        "## Inputs",
        f"- Samples analyzed: {n_samples}",
        f"- CAF proxy: `{args.caf_channel}`",
        f"- CAR-T proxy: `{args.cart_channel}`",
        f"- Tumor-context channel: `{args.tumor_channel}`",
        f"- Tile sizes: {', '.join(str(x) for x in args.tile_sizes)} px",
        f"- Primary tile size: {args.primary_tile_size} px",
        f"- Distance bands: {', '.join(str(x) for x in args.distance_bands)} px",
        "",
        "## Primary Endpoint",
        f"- Spearman rho median: {_fmt_num(primary_rho.get('median') if primary_rho else float('nan'))}, "
        f"95% bootstrap CI [{_fmt_num(primary_rho.get('bootstrap_ci_low') if primary_rho else float('nan'))}, "
        f"{_fmt_num(primary_rho.get('bootstrap_ci_high') if primary_rho else float('nan'))}], "
        f"one-sided Wilcoxon p(rho < 0) = {_fmt_num(primary_rho.get('wilcoxon_p') if primary_rho else float('nan'))}",
        f"- High-minus-low CD8 delta median: {_fmt_num(primary_delta.get('median') if primary_delta else float('nan'))}, "
        f"95% bootstrap CI [{_fmt_num(primary_delta.get('bootstrap_ci_low') if primary_delta else float('nan'))}, "
        f"{_fmt_num(primary_delta.get('bootstrap_ci_high') if primary_delta else float('nan'))}], "
        f"one-sided Wilcoxon p(delta < 0) = {_fmt_num(primary_delta.get('wilcoxon_p') if primary_delta else float('nan'))}",
        f"- Interpretation: {interpretation}",
        "",
        "## Backup Endpoints",
        f"- Tumor-context rho median: {_fmt_num(tumor_rho.get('median') if tumor_rho else float('nan'))}, "
        f"p(rho < 0) = {_fmt_num(tumor_rho.get('wilcoxon_p') if tumor_rho else float('nan'))}",
        f"- Tumor-context delta median: {_fmt_num(tumor_delta.get('median') if tumor_delta else float('nan'))}, "
        f"p(delta < 0) = {_fmt_num(tumor_delta.get('wilcoxon_p') if tumor_delta else float('nan'))}",
        f"- Distance-gradient rho median: {_fmt_num(distance_rho.get('median') if distance_rho else float('nan'))}, "
        f"p(rho > 0) = {_fmt_num(distance_rho.get('wilcoxon_p') if distance_rho else float('nan'))}",
        f"- Far-minus-near CD8 delta median: {_fmt_num(distance_delta.get('median') if distance_delta else float('nan'))}, "
        f"p(delta > 0) = {_fmt_num(distance_delta.get('wilcoxon_p') if distance_delta else float('nan'))}",
        "",
        "## Secondary SegX-GradCAM",
        *(_gradcam_corr_lines(gradcam_rows)),
        "",
        "## Output Files",
        "- `tile_metrics.csv`",
        "- `sample_stats.csv`",
        "- `cohort_stats.csv`",
        "- `cohort_stats.json`",
        "- `distance_gradient.csv`",
        "- `distance_sample_stats.csv`",
        "- `gradcam_secondary.csv`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_payload = load_split_json(args.split_json)
    labels = load_pdo_change_labels(args.label_json)
    split_by_id = {}
    image_ids = []
    for split_name in ("train", "val"):
        for image_id in split_payload.get(split_name, []) or []:
            image_id = str(image_id)
            split_by_id[image_id] = split_name
            image_ids.append(image_id)
    if args.limit is not None:
        image_ids = image_ids[: int(args.limit)]
    if not image_ids:
        raise ValueError("No R2 image IDs found to analyze.")

    tile_sizes = [int(x) for x in args.tile_sizes]
    if int(args.primary_tile_size) not in tile_sizes:
        tile_sizes.append(int(args.primary_tile_size))
        tile_sizes = sorted(tile_sizes)

    case_index = _build_case_dir_index(args.masks_dir)
    tile_rows: list[dict] = []
    sample_rows: list[dict] = []
    distance_rows: list[dict] = []
    distance_sample_rows: list[dict] = []
    meta_by_id: dict[str, dict] = {}

    for idx, image_id in enumerate(image_ids, start=1):
        case_dir = _resolve_case_dir(args.masks_dir, case_index, image_id)
        paths = {
            args.caf_channel: _resolve_mask_path(case_dir, args.caf_channel),
            args.cart_channel: _resolve_mask_path(case_dir, args.cart_channel),
            args.tumor_channel: _resolve_mask_path(case_dir, args.tumor_channel),
        }
        missing = [name for name, path in paths.items() if path is None]
        if missing:
            raise FileNotFoundError(f"{image_id} missing masks: {missing} in {case_dir}")

        caf_mask = _load_binary_mask(paths[args.caf_channel], args.mask_threshold)
        cart_mask = _load_binary_mask(paths[args.cart_channel], args.mask_threshold)
        tumor_mask = _load_binary_mask(paths[args.tumor_channel], args.mask_threshold)
        shapes = {caf_mask.shape, cart_mask.shape, tumor_mask.shape}
        if len(shapes) != 1:
            raise ValueError(f"{image_id} mask shape mismatch: {sorted(shapes)}")

        meta = _sample_meta(image_id, split_by_id.get(image_id, ""), labels)
        meta_by_id[image_id] = {**meta, "height": caf_mask.shape[0], "width": caf_mask.shape[1]}
        sample_tile_rows = _compute_tile_rows(meta, caf_mask, cart_mask, tumor_mask, tile_sizes)
        tile_rows.extend(sample_tile_rows)

        for tile_size in tile_sizes:
            rows_for_size = [r for r in sample_tile_rows if int(r["tile_size"]) == int(tile_size)]
            sample_rows.append(_summarize_tiles(meta, rows_for_size, "raw_all_tiles"))
            tumor_rows = [r for r in rows_for_size if float(r["ck_frac"]) > float(args.tumor_min_frac)]
            sample_rows.append(_summarize_tiles(meta, tumor_rows, "tumor_context_tiles"))

        dist_rows, dist_stat = _distance_band_rows(meta, caf_mask, cart_mask, args.distance_bands)
        distance_rows.extend(dist_rows)
        distance_sample_rows.append(dist_stat)
        print(f"Processed {idx}/{len(image_ids)}: {image_id}")

    _validate_primary_tile_counts(tile_rows, meta_by_id, int(args.primary_tile_size))

    primary_rows = [
        r for r in sample_rows
        if int(r["tile_size"]) == int(args.primary_tile_size) and r["endpoint"] == "raw_all_tiles"
    ]
    gradcam_rows = _build_gradcam_secondary(args.gradcam_summary, primary_rows)
    cohort_rows = _build_cohort_stats(
        sample_rows=sample_rows,
        distance_sample_rows=distance_sample_rows,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    _write_csv(out_dir / "tile_metrics.csv", tile_rows, TILE_FIELDS)
    _write_csv(out_dir / "sample_stats.csv", sample_rows, SAMPLE_FIELDS)
    _write_csv(out_dir / "distance_gradient.csv", distance_rows, DISTANCE_FIELDS)
    _write_csv(out_dir / "distance_sample_stats.csv", distance_sample_rows, DISTANCE_SAMPLE_FIELDS)
    _write_csv(out_dir / "cohort_stats.csv", cohort_rows, COHORT_FIELDS)
    with (out_dir / "cohort_stats.json").open("w", encoding="utf-8") as f:
        json.dump([{k: _format_csv_value(v) for k, v in row.items()} for row in cohort_rows], f, indent=2)

    gradcam_fields = [
        "image_id",
        "split",
        "pdo_change",
        "pdo_bin_idx",
        "pdo_bin_label",
        "tile_size",
        "sample_spearman_rho",
        "sample_delta_high_minus_low",
        "sample_mean_actin_frac",
        "sample_mean_cd8_frac",
        "actin_cam_mean",
        "actin_cam_hi_frac",
        "cd8_cam_mean",
        "cd8_cam_hi_frac",
    ]
    _write_csv(out_dir / "gradcam_secondary.csv", gradcam_rows, gradcam_fields)

    if not args.no_plots:
        _make_plots(tile_rows, sample_rows, distance_rows, gradcam_rows, int(args.primary_tile_size), out_dir)

    _write_report(out_dir / "caf_cd8_local_report.md", args, len(image_ids), cohort_rows, gradcam_rows)

    primary_rho = _cohort_lookup(cohort_rows, "raw_all_tiles", int(args.primary_tile_size), "spearman_rho")
    primary_delta = _cohort_lookup(cohort_rows, "raw_all_tiles", int(args.primary_tile_size), "delta_high_minus_low")
    print()
    print(f"Analyzed {len(image_ids)} samples. Outputs in: {out_dir}")
    if primary_rho and primary_delta:
        print(
            "Primary 512px raw endpoint: "
            f"rho median={_fmt_num(primary_rho['median'])}, p<0={_fmt_num(primary_rho['wilcoxon_p'])}; "
            f"delta median={_fmt_num(primary_delta['median'])}, p<0={_fmt_num(primary_delta['wilcoxon_p'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
