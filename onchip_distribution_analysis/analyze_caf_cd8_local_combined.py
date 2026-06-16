"""Combined R1+R2 local CAF-CD8 spatial analysis.

This script pools train+val samples from the original on-chip dataset
(`data/On-chip_Data`) and the processed R2 dataset (`data/On-chip_Data_R2`).
It reuses the native-resolution mask analysis from `analyze_caf_cd8_local.py`
while keeping cohort/source IDs explicit so overlapping image IDs cannot collide.
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from common.pdochange_data import load_pdo_change_labels, load_split_json
from onchip_distribution_analysis.analyze_caf_cd8_local import (
    COHORT_FIELDS,
    DISTANCE_FIELDS,
    DISTANCE_SAMPLE_FIELDS,
    SAMPLE_FIELDS,
    TILE_FIELDS,
    _build_case_dir_index,
    _build_cohort_stats,
    _cohort_row,
    _cohort_lookup,
    _compute_tile_rows,
    _distance_band_rows,
    _fmt_num,
    _load_binary_mask,
    _make_plots,
    _resolve_case_dir,
    _resolve_mask_path,
    _sample_meta,
    _safe_spearman,
    _summarize_tiles,
    _validate_primary_tile_counts,
    _write_csv,
)


@dataclass(frozen=True)
class CohortSpec:
    name: str
    masks_dir: Path
    split_json: Path
    label_json: Path


DEFAULT_COHORTS = [
    CohortSpec(
        name="r1",
        masks_dir=Path("data/On-chip_Data"),
        split_json=Path("data/On-chip_Data/data_split.json"),
        label_json=Path("data/On-chip_Data/pdo_change_label.json"),
    ),
    CohortSpec(
        name="r2",
        masks_dir=Path("data/On-chip_Data_R2"),
        split_json=Path("data/On-chip_Data_R2/data_split.json"),
        label_json=Path("data/On-chip_Data_R2/pdo_change_label.json"),
    ),
]

TUMOR_ADJUSTED_SAMPLE_FIELDS = [
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
    "spearman_actin_cd8",
    "spearman_actin_ck",
    "spearman_ck_cd8",
    "partial_spearman_actin_cd8_given_ck",
    "residual_cd8_low_actin_q1_mean",
    "residual_cd8_high_actin_q4_mean",
    "residual_delta_high_minus_low",
    "ck_stratified_delta_high_minus_low",
    "n_ck_strata",
    "n_ck_stratified_tiles",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Combined R1+R2 local CAF-CD8 analysis.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onchip_pdochange_prediction/results/combined_r1_r2/caf_cd8_local_analysis"),
    )
    parser.add_argument("--cohorts", nargs="+", default=["r1", "r2"], choices=["r1", "r2"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test", "all"])
    parser.add_argument("--caf-channel", type=str, default="actin")
    parser.add_argument("--cart-channel", type=str, default="cd8")
    parser.add_argument("--tumor-channel", type=str, default="ck")
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--primary-tile-size", type=int, default=512)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--tumor-min-frac", type=float, default=0.0)
    parser.add_argument("--distance-bands", nargs="+", type=int, default=[0, 128, 256, 512, 1024])
    parser.add_argument("--skip-distance", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--limit-per-cohort", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def _unique_image_id(cohort_name: str, image_id: str) -> str:
    return f"{cohort_name}:{image_id}"


def _combined_fields(fields: list[str]) -> list[str]:
    return ["cohort", "source_image_id", *fields]


def _selected_cohorts(names: list[str]) -> list[CohortSpec]:
    selected = set(names)
    return [spec for spec in DEFAULT_COHORTS if spec.name in selected]


def _selected_splits(names: list[str]) -> list[str]:
    if "all" in names:
        return ["train", "val", "test"]
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def _group_cohort_stats(
    sample_rows: list[dict],
    distance_sample_rows: list[dict],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict]:
    grouped_rows: list[dict] = []
    group_names = ["all", *sorted({str(r["cohort"]) for r in sample_rows})]
    for group_name in group_names:
        if group_name == "all":
            s_rows = sample_rows
            d_rows = distance_sample_rows
        else:
            s_rows = [r for r in sample_rows if str(r["cohort"]) == group_name]
            d_rows = [r for r in distance_sample_rows if str(r["cohort"]) == group_name]
        stats = _build_cohort_stats(
            sample_rows=s_rows,
            distance_sample_rows=d_rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        for row in stats:
            grouped_rows.append({"group": group_name, **row})
    return grouped_rows


def _stat_lookup(rows: list[dict], group: str, endpoint: str, tile_size, metric: str) -> dict | None:
    for row in rows:
        if (
            row.get("group") == group
            and row.get("endpoint") == endpoint
            and str(row.get("tile_size")) == str(tile_size)
            and row.get("metric") == metric
        ):
            return row
    return None


def _finite_pair(*arrays: np.ndarray) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=np.float64) for a in arrays]
    valid = np.ones(arrs[0].shape, dtype=bool)
    for arr in arrs:
        valid &= np.isfinite(arr)
    return [arr[valid] for arr in arrs]


def _pearson_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x, y = _finite_pair(x_values, y_values)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _residualize_on_covariate(y_values: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    y = np.asarray(y_values, dtype=np.float64)
    x = np.asarray(x_values, dtype=np.float64)
    residuals = np.full(y.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < 2:
        return residuals
    design = np.column_stack([np.ones(int(valid.sum()), dtype=np.float64), x[valid]])
    beta, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    residuals[valid] = y[valid] - design @ beta
    return residuals


def _partial_spearman_given_ck(actin: np.ndarray, cd8: np.ndarray, ck: np.ndarray) -> float:
    a, c, k = _finite_pair(actin, cd8, ck)
    if a.size < 3:
        return float("nan")
    rank_a = rankdata(a, method="average")
    rank_c = rankdata(c, method="average")
    rank_k = rankdata(k, method="average")
    a_resid = _residualize_on_covariate(rank_a, rank_k)
    c_resid = _residualize_on_covariate(rank_c, rank_k)
    return _pearson_corr(a_resid, c_resid)


def _ck_stratified_delta(
    actin: np.ndarray,
    cd8: np.ndarray,
    ck: np.ndarray,
    n_strata: int = 4,
    min_tiles_per_stratum: int = 4,
) -> tuple[float, int, int]:
    a, c, k = _finite_pair(actin, cd8, ck)
    if a.size < int(min_tiles_per_stratum):
        return float("nan"), 0, 0
    order = np.argsort(k, kind="mergesort")
    strata = np.array_split(order, min(int(n_strata), int(order.size)))
    deltas: list[float] = []
    weights: list[int] = []
    compared_tiles = 0
    for stratum in strata:
        if stratum.size < int(min_tiles_per_stratum):
            continue
        stratum_actin = a[stratum]
        stratum_cd8 = c[stratum]
        if np.allclose(stratum_actin, stratum_actin[0]):
            continue
        q25, q75 = np.quantile(stratum_actin, [0.25, 0.75])
        low = stratum_cd8[stratum_actin <= q25]
        high = stratum_cd8[stratum_actin >= q75]
        if low.size == 0 or high.size == 0:
            continue
        deltas.append(float(high.mean() - low.mean()))
        weight = int(low.size + high.size)
        weights.append(weight)
        compared_tiles += weight
    if not deltas:
        return float("nan"), 0, 0
    return float(np.average(np.asarray(deltas), weights=np.asarray(weights))), int(len(deltas)), int(compared_tiles)


def _summarize_tumor_adjusted(meta: dict, tile_rows: list[dict], endpoint: str) -> dict:
    if not tile_rows:
        return {
            **meta,
            "tile_size": "",
            "endpoint": endpoint,
            "n_tiles": 0,
            "mean_actin_frac": float("nan"),
            "mean_cd8_frac": float("nan"),
            "mean_ck_frac": float("nan"),
            "spearman_actin_cd8": float("nan"),
            "spearman_actin_ck": float("nan"),
            "spearman_ck_cd8": float("nan"),
            "partial_spearman_actin_cd8_given_ck": float("nan"),
            "residual_cd8_low_actin_q1_mean": float("nan"),
            "residual_cd8_high_actin_q4_mean": float("nan"),
            "residual_delta_high_minus_low": float("nan"),
            "ck_stratified_delta_high_minus_low": float("nan"),
            "n_ck_strata": 0,
            "n_ck_stratified_tiles": 0,
        }

    tile_size = int(tile_rows[0]["tile_size"])
    actin = np.asarray([r["actin_frac"] for r in tile_rows], dtype=np.float64)
    cd8 = np.asarray([r["cd8_frac"] for r in tile_rows], dtype=np.float64)
    ck = np.asarray([r["ck_frac"] for r in tile_rows], dtype=np.float64)
    rho_ac, _ = _safe_spearman(actin, cd8)
    rho_ak, _ = _safe_spearman(actin, ck)
    rho_kc, _ = _safe_spearman(ck, cd8)
    partial_rho = _partial_spearman_given_ck(actin, cd8, ck)

    cd8_resid = _residualize_on_covariate(cd8, ck)
    q25, q75 = np.quantile(actin, [0.25, 0.75])
    low = cd8_resid[actin <= q25]
    high = cd8_resid[actin >= q75]
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    low_mean = float(low.mean()) if low.size else float("nan")
    high_mean = float(high.mean()) if high.size else float("nan")
    residual_delta = float(high_mean - low_mean) if math.isfinite(high_mean) and math.isfinite(low_mean) else float("nan")
    stratified_delta, n_strata, n_stratified_tiles = _ck_stratified_delta(actin, cd8, ck)

    return {
        **meta,
        "tile_size": tile_size,
        "endpoint": endpoint,
        "n_tiles": int(len(tile_rows)),
        "mean_actin_frac": float(np.mean(actin)),
        "mean_cd8_frac": float(np.mean(cd8)),
        "mean_ck_frac": float(np.mean(ck)),
        "spearman_actin_cd8": rho_ac,
        "spearman_actin_ck": rho_ak,
        "spearman_ck_cd8": rho_kc,
        "partial_spearman_actin_cd8_given_ck": partial_rho,
        "residual_cd8_low_actin_q1_mean": low_mean,
        "residual_cd8_high_actin_q4_mean": high_mean,
        "residual_delta_high_minus_low": residual_delta,
        "ck_stratified_delta_high_minus_low": stratified_delta,
        "n_ck_strata": n_strata,
        "n_ck_stratified_tiles": n_stratified_tiles,
    }


def _build_tumor_adjusted_group_stats(
    tumor_adjusted_rows: list[dict],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict]:
    rng = np.random.default_rng(int(bootstrap_seed))
    out: list[dict] = []
    group_names = ["all", *sorted({str(r["cohort"]) for r in tumor_adjusted_rows})]
    metrics = [
        "partial_spearman_actin_cd8_given_ck",
        "residual_delta_high_minus_low",
        "ck_stratified_delta_high_minus_low",
    ]
    for group_name in group_names:
        rows_for_group = tumor_adjusted_rows if group_name == "all" else [
            r for r in tumor_adjusted_rows if str(r["cohort"]) == group_name
        ]
        grouped: dict[tuple[str, int], list[dict]] = {}
        for row in rows_for_group:
            if row.get("tile_size") == "":
                continue
            grouped.setdefault((str(row["endpoint"]), int(row["tile_size"])), []).append(row)
        for (endpoint, tile_size), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            for metric in metrics:
                stat_row = _cohort_row(
                    "tumor_adjusted_tile",
                    endpoint,
                    tile_size,
                    metric,
                    [r[metric] for r in rows],
                    "less",
                    rng,
                    int(bootstrap_iterations),
                )
                out.append({"group": group_name, **stat_row})
    return out


def _plot_tumor_adjusted(tumor_adjusted_rows: list[dict], primary_tile_size: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r for r in tumor_adjusted_rows
        if int(r["tile_size"]) == int(primary_tile_size) and r["endpoint"] == "tumor_adjusted_all_tiles"
    ]
    if not rows:
        return
    metrics = [
        ("partial_spearman_actin_cd8_given_ck", "Partial rho | CK"),
        ("residual_delta_high_minus_low", "CD8 residual delta"),
        ("ck_stratified_delta_high_minus_low", "CK-stratified delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), dpi=140)
    for ax, (metric, label) in zip(axes, metrics):
        vals = np.asarray([r[metric] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        ax.boxplot(vals, showfliers=False)
        jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size else []
        if vals.size:
            ax.scatter(np.ones(vals.size) + jitter, vals, s=12, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks([1])
        ax.set_xticklabels([label], rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Tumor-adjusted CAF-CD8 endpoints ({primary_tile_size}px tiles)")
    fig.tight_layout()
    fig.savefig(out_dir / f"tumor_adjusted_endpoints_{primary_tile_size}.png", bbox_inches="tight")
    plt.close(fig)


def _write_combined_report(
    out_path: Path,
    args,
    group_counts: dict[str, int],
    grouped_stats: list[dict],
    tumor_adjusted_grouped_stats: list[dict],
) -> None:
    primary_size = int(args.primary_tile_size)
    primary_rho = _stat_lookup(grouped_stats, "all", "raw_all_tiles", primary_size, "spearman_rho")
    primary_delta = _stat_lookup(grouped_stats, "all", "raw_all_tiles", primary_size, "delta_high_minus_low")
    tumor_rho = _stat_lookup(grouped_stats, "all", "tumor_context_tiles", primary_size, "spearman_rho")
    tumor_delta = _stat_lookup(grouped_stats, "all", "tumor_context_tiles", primary_size, "delta_high_minus_low")
    distance_rho = _stat_lookup(grouped_stats, "all", "distance_from_actin", "", "distance_spearman_rho")
    distance_delta = _stat_lookup(grouped_stats, "all", "distance_from_actin", "", "delta_far_minus_near")
    partial_rho = _stat_lookup(
        tumor_adjusted_grouped_stats,
        "all",
        "tumor_adjusted_all_tiles",
        primary_size,
        "partial_spearman_actin_cd8_given_ck",
    )
    residual_delta = _stat_lookup(
        tumor_adjusted_grouped_stats,
        "all",
        "tumor_adjusted_all_tiles",
        primary_size,
        "residual_delta_high_minus_low",
    )
    stratified_delta = _stat_lookup(
        tumor_adjusted_grouped_stats,
        "all",
        "tumor_adjusted_all_tiles",
        primary_size,
        "ck_stratified_delta_high_minus_low",
    )

    if primary_rho and primary_delta and primary_rho["supports_hypothesis"] and primary_delta["supports_hypothesis"]:
        interpretation = "The pooled raw local CD8 endpoint supports CAF-high/CD8-low."
    else:
        interpretation = "The pooled raw local CD8 endpoint does not support CAF-high/CD8-low."

    lines = [
        "# Local CAF-CD8 Quantitative Analysis",
        "",
        "## Inputs",
        f"- Samples analyzed: {sum(group_counts.values())}",
        *[f"- {name}: {count} samples" for name, count in sorted(group_counts.items())],
        f"- Cohorts: {', '.join(args.cohorts)}",
        f"- Splits: {', '.join(_selected_splits(args.splits))}",
        f"- CAF proxy: `{args.caf_channel}`",
        f"- CAR-T proxy: `{args.cart_channel}`",
        f"- Tumor-context channel: `{args.tumor_channel}`",
        f"- Tile sizes: {', '.join(str(x) for x in args.tile_sizes)} px",
        f"- Primary tile size: {args.primary_tile_size} px",
        f"- Distance bands: {'skipped' if args.skip_distance else ', '.join(str(x) for x in args.distance_bands) + ' px'}",
        "",
        "## Pooled Primary Endpoint",
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
        "## Tumor-Adjusted Endpoint",
        f"- Partial Spearman rho(actin, CD8 | CK) median: "
        f"{_fmt_num(partial_rho.get('median') if partial_rho else float('nan'))}, "
        f"95% bootstrap CI [{_fmt_num(partial_rho.get('bootstrap_ci_low') if partial_rho else float('nan'))}, "
        f"{_fmt_num(partial_rho.get('bootstrap_ci_high') if partial_rho else float('nan'))}], "
        f"one-sided Wilcoxon p(rho < 0) = {_fmt_num(partial_rho.get('wilcoxon_p') if partial_rho else float('nan'))}",
        f"- CK-adjusted CD8 residual high-minus-low actin delta median: "
        f"{_fmt_num(residual_delta.get('median') if residual_delta else float('nan'))}, "
        f"95% bootstrap CI [{_fmt_num(residual_delta.get('bootstrap_ci_low') if residual_delta else float('nan'))}, "
        f"{_fmt_num(residual_delta.get('bootstrap_ci_high') if residual_delta else float('nan'))}], "
        f"one-sided Wilcoxon p(delta < 0) = {_fmt_num(residual_delta.get('wilcoxon_p') if residual_delta else float('nan'))}",
        f"- CK-stratified high-minus-low actin CD8 delta median: "
        f"{_fmt_num(stratified_delta.get('median') if stratified_delta else float('nan'))}, "
        f"95% bootstrap CI [{_fmt_num(stratified_delta.get('bootstrap_ci_low') if stratified_delta else float('nan'))}, "
        f"{_fmt_num(stratified_delta.get('bootstrap_ci_high') if stratified_delta else float('nan'))}], "
        f"one-sided Wilcoxon p(delta < 0) = {_fmt_num(stratified_delta.get('wilcoxon_p') if stratified_delta else float('nan'))}",
        "",
        "## Pooled Backup Endpoints",
        f"- Tumor-context rho median: {_fmt_num(tumor_rho.get('median') if tumor_rho else float('nan'))}, "
        f"p(rho < 0) = {_fmt_num(tumor_rho.get('wilcoxon_p') if tumor_rho else float('nan'))}",
        f"- Tumor-context delta median: {_fmt_num(tumor_delta.get('median') if tumor_delta else float('nan'))}, "
        f"p(delta < 0) = {_fmt_num(tumor_delta.get('wilcoxon_p') if tumor_delta else float('nan'))}",
        f"- Distance-gradient rho median: {_fmt_num(distance_rho.get('median') if distance_rho else float('nan'))}, "
        f"p(rho > 0) = {_fmt_num(distance_rho.get('wilcoxon_p') if distance_rho else float('nan'))}",
        f"- Far-minus-near CD8 delta median: {_fmt_num(distance_delta.get('median') if distance_delta else float('nan'))}, "
        f"p(delta > 0) = {_fmt_num(distance_delta.get('wilcoxon_p') if distance_delta else float('nan'))}",
        "",
        "## By Cohort, 512px Raw Endpoint",
    ]
    for group_name in sorted(group_counts):
        rho = _stat_lookup(grouped_stats, group_name, "raw_all_tiles", primary_size, "spearman_rho")
        delta = _stat_lookup(grouped_stats, group_name, "raw_all_tiles", primary_size, "delta_high_minus_low")
        lines.append(
            f"- {group_name}: rho median {_fmt_num(rho.get('median') if rho else float('nan'))}, "
            f"delta median {_fmt_num(delta.get('median') if delta else float('nan'))}, "
            f"p(delta < 0) {_fmt_num(delta.get('wilcoxon_p') if delta else float('nan'))}"
        )
    lines.extend([
        "",
        "## Output Files",
        "- `tile_metrics.csv`",
        "- `sample_stats.csv`",
        "- `distance_gradient.csv`",
        "- `distance_sample_stats.csv`",
        "- `cohort_stats.csv`",
        "- `cohort_stats_by_group.csv` / `cohort_stats_by_group.json`",
        "- `tumor_adjusted_sample_stats.csv`",
        "- `tumor_adjusted_cohort_stats_by_group.csv` / `tumor_adjusted_cohort_stats_by_group.json`",
        "- `tile_scatter_hexbin_512.png`",
        "- `sample_rho_forest_512.png`",
        "- `high_low_cd8_boxplot_512.png`",
        "- `tumor_adjusted_endpoints_512.png`",
        "- `distance_gradient.png`",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_sizes = [int(x) for x in args.tile_sizes]
    if int(args.primary_tile_size) not in tile_sizes:
        tile_sizes = sorted([*tile_sizes, int(args.primary_tile_size)])

    tile_rows: list[dict] = []
    sample_rows: list[dict] = []
    tumor_adjusted_rows: list[dict] = []
    distance_rows: list[dict] = []
    distance_sample_rows: list[dict] = []
    meta_by_id: dict[str, dict] = {}
    group_counts: dict[str, int] = {}

    selected_splits = _selected_splits(args.splits)
    for cohort in _selected_cohorts(args.cohorts):
        split_payload = load_split_json(cohort.split_json)
        labels = load_pdo_change_labels(cohort.label_json)
        image_split_pairs: list[tuple[str, str]] = []
        seen_ids: set[str] = set()
        for split_name in selected_splits:
            for image_id in split_payload.get(split_name, []) or []:
                image_id = str(image_id)
                if image_id in seen_ids:
                    continue
                seen_ids.add(image_id)
                image_split_pairs.append((image_id, split_name))
        if args.limit_per_cohort is not None:
            image_split_pairs = image_split_pairs[: int(args.limit_per_cohort)]
        if not image_split_pairs:
            raise ValueError(f"No image IDs found for cohort {cohort.name}.")

        case_index = _build_case_dir_index(cohort.masks_dir)
        group_counts[cohort.name] = len(image_split_pairs)
        for idx, (source_image_id, split_name) in enumerate(image_split_pairs, start=1):
            case_dir = _resolve_case_dir(cohort.masks_dir, case_index, source_image_id)
            paths = {
                args.caf_channel: _resolve_mask_path(case_dir, args.caf_channel),
                args.cart_channel: _resolve_mask_path(case_dir, args.cart_channel),
                args.tumor_channel: _resolve_mask_path(case_dir, args.tumor_channel),
            }
            missing = [name for name, path in paths.items() if path is None]
            if missing:
                raise FileNotFoundError(f"{cohort.name}:{source_image_id} missing masks: {missing} in {case_dir}")

            caf_mask = _load_binary_mask(paths[args.caf_channel], args.mask_threshold)
            cart_mask = _load_binary_mask(paths[args.cart_channel], args.mask_threshold)
            tumor_mask = _load_binary_mask(paths[args.tumor_channel], args.mask_threshold)
            shapes = {caf_mask.shape, cart_mask.shape, tumor_mask.shape}
            if len(shapes) != 1:
                raise ValueError(f"{cohort.name}:{source_image_id} mask shape mismatch: {sorted(shapes)}")

            meta = _sample_meta(source_image_id, split_name, labels)
            meta = {
                **meta,
                "cohort": cohort.name,
                "source_image_id": source_image_id,
                "image_id": _unique_image_id(cohort.name, source_image_id),
            }
            meta_by_id[meta["image_id"]] = {**meta, "height": caf_mask.shape[0], "width": caf_mask.shape[1]}

            sample_tile_rows = _compute_tile_rows(meta, caf_mask, cart_mask, tumor_mask, tile_sizes)
            tile_rows.extend(sample_tile_rows)

            for tile_size in tile_sizes:
                rows_for_size = [r for r in sample_tile_rows if int(r["tile_size"]) == int(tile_size)]
                sample_rows.append(_summarize_tiles(meta, rows_for_size, "raw_all_tiles"))
                tumor_rows = [r for r in rows_for_size if float(r["ck_frac"]) > float(args.tumor_min_frac)]
                sample_rows.append(_summarize_tiles(meta, tumor_rows, "tumor_context_tiles"))
                tumor_adjusted_rows.append(
                    _summarize_tumor_adjusted(meta, rows_for_size, "tumor_adjusted_all_tiles")
                )
                tumor_adjusted_rows.append(
                    _summarize_tumor_adjusted(meta, tumor_rows, "tumor_adjusted_tumor_context_tiles")
                )

            if not args.skip_distance:
                dist_rows, dist_stat = _distance_band_rows(meta, caf_mask, cart_mask, args.distance_bands)
                distance_rows.extend(dist_rows)
                distance_sample_rows.append(dist_stat)
            print(f"Processed {cohort.name} {idx}/{len(image_split_pairs)}: {source_image_id}", flush=True)

    _validate_primary_tile_counts(tile_rows, meta_by_id, int(args.primary_tile_size))

    cohort_rows = _build_cohort_stats(
        sample_rows=sample_rows,
        distance_sample_rows=distance_sample_rows,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    grouped_stats = _group_cohort_stats(
        sample_rows=sample_rows,
        distance_sample_rows=distance_sample_rows,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    tumor_adjusted_grouped_stats = _build_tumor_adjusted_group_stats(
        tumor_adjusted_rows=tumor_adjusted_rows,
        bootstrap_iterations=int(args.bootstrap_iterations),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    _write_csv(out_dir / "tile_metrics.csv", tile_rows, _combined_fields(TILE_FIELDS))
    _write_csv(out_dir / "sample_stats.csv", sample_rows, _combined_fields(SAMPLE_FIELDS))
    _write_csv(
        out_dir / "tumor_adjusted_sample_stats.csv",
        tumor_adjusted_rows,
        _combined_fields(TUMOR_ADJUSTED_SAMPLE_FIELDS),
    )
    _write_csv(out_dir / "distance_gradient.csv", distance_rows, _combined_fields(DISTANCE_FIELDS))
    _write_csv(out_dir / "distance_sample_stats.csv", distance_sample_rows, _combined_fields(DISTANCE_SAMPLE_FIELDS))
    _write_csv(out_dir / "cohort_stats.csv", cohort_rows, COHORT_FIELDS)
    _write_csv(out_dir / "cohort_stats_by_group.csv", grouped_stats, ["group", *COHORT_FIELDS])
    _write_csv(
        out_dir / "tumor_adjusted_cohort_stats_by_group.csv",
        tumor_adjusted_grouped_stats,
        ["group", *COHORT_FIELDS],
    )
    with (out_dir / "cohort_stats.json").open("w", encoding="utf-8") as f:
        json.dump(cohort_rows, f, indent=2)
    with (out_dir / "cohort_stats_by_group.json").open("w", encoding="utf-8") as f:
        json.dump(grouped_stats, f, indent=2)
    with (out_dir / "tumor_adjusted_cohort_stats_by_group.json").open("w", encoding="utf-8") as f:
        json.dump(tumor_adjusted_grouped_stats, f, indent=2)

    if not args.no_plots:
        _make_plots(tile_rows, sample_rows, distance_rows, [], int(args.primary_tile_size), out_dir)
        _plot_tumor_adjusted(tumor_adjusted_rows, int(args.primary_tile_size), out_dir)

    _write_combined_report(
        out_dir / "caf_cd8_local_combined_report.md",
        args,
        group_counts,
        grouped_stats,
        tumor_adjusted_grouped_stats,
    )

    primary_rho = _cohort_lookup(cohort_rows, "raw_all_tiles", int(args.primary_tile_size), "spearman_rho")
    primary_delta = _cohort_lookup(cohort_rows, "raw_all_tiles", int(args.primary_tile_size), "delta_high_minus_low")
    print()
    print(f"Analyzed {sum(group_counts.values())} samples. Outputs in: {out_dir}")
    if primary_rho and primary_delta:
        print(
            "Pooled 512px raw endpoint: "
            f"rho median={_fmt_num(primary_rho['median'])}, p<0={_fmt_num(primary_rho['wilcoxon_p'])}; "
            f"delta median={_fmt_num(primary_delta['median'])}, p<0={_fmt_num(primary_delta['wilcoxon_p'])}"
        )
    adjusted_rho = _stat_lookup(
        tumor_adjusted_grouped_stats,
        "all",
        "tumor_adjusted_all_tiles",
        int(args.primary_tile_size),
        "partial_spearman_actin_cd8_given_ck",
    )
    adjusted_delta = _stat_lookup(
        tumor_adjusted_grouped_stats,
        "all",
        "tumor_adjusted_all_tiles",
        int(args.primary_tile_size),
        "ck_stratified_delta_high_minus_low",
    )
    if adjusted_rho and adjusted_delta:
        print(
            "Tumor-adjusted 512px endpoint: "
            f"partial rho median={_fmt_num(adjusted_rho['median'])}, "
            f"p<0={_fmt_num(adjusted_rho['wilcoxon_p'])}; "
            f"CK-stratified delta median={_fmt_num(adjusted_delta['median'])}, "
            f"p<0={_fmt_num(adjusted_delta['wilcoxon_p'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
