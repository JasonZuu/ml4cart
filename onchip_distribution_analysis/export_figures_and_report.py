"""Export standalone figures, plotting data, code snapshots, and report.

Run from the repository root after generating:
  on-chip_distribution-analysis/results/r1/{raw_mask,segxgradcam}
  on-chip_distribution-analysis/results/r2/{raw_mask,segxgradcam}
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("on-chip_distribution-analysis")
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "plotting_data"
CODE_DIR = ROOT / "code"
RESULTS_DIR = ROOT / "results"

COHORT_LABELS = {"r1": "R1", "r2": "R2"}
COLORS = {"r1": "#3267A8", "r2": "#C04B37"}


def ensure_dirs() -> None:
    for path in (FIG_DIR, DATA_DIR, CODE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def copy_code_snapshots() -> None:
    sources = [
        Path("onchip_pdochange_prediction/analyze_caf_cd8_local_combined.py"),
        Path("onchip_pdochange_prediction/analyze_caf_cd8_segxgradcam.py"),
    ]
    for src in sources:
        shutil.copy2(src, CODE_DIR / src.name)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_raw_tiles(cohort: str) -> pd.DataFrame:
    df = read_csv(RESULTS_DIR / cohort / "raw_mask" / "tile_metrics.csv")
    df = df[df["tile_size"].astype(int) == 512].copy()
    df.insert(0, "round", cohort)
    keep = ["round", "source_image_id", "split", "tile_size", "actin_frac", "cd8_frac", "ck_frac"]
    return df[keep]


def load_raw_sample(cohort: str) -> pd.DataFrame:
    df = read_csv(RESULTS_DIR / cohort / "raw_mask" / "sample_stats.csv")
    df = df[(df["tile_size"].astype(int) == 512) & (df["endpoint"] == "raw_all_tiles")].copy()
    df.insert(0, "round", cohort)
    keep = [
        "round",
        "source_image_id",
        "split",
        "mean_actin_frac",
        "mean_cd8_frac",
        "mean_ck_frac",
        "spearman_rho",
        "delta_high_minus_low",
        "cd8_low_actin_q1_mean",
        "cd8_high_actin_q4_mean",
    ]
    return df[keep]


def load_tumor_adjusted_sample(cohort: str) -> pd.DataFrame:
    df = read_csv(RESULTS_DIR / cohort / "raw_mask" / "tumor_adjusted_sample_stats.csv")
    df = df[
        (df["tile_size"].astype(int) == 512)
        & (df["endpoint"] == "tumor_adjusted_all_tiles")
    ].copy()
    df.insert(0, "round", cohort)
    keep = [
        "round",
        "source_image_id",
        "split",
        "spearman_actin_cd8",
        "spearman_actin_ck",
        "spearman_ck_cd8",
        "partial_spearman_actin_cd8_given_ck",
        "residual_delta_high_minus_low",
        "ck_stratified_delta_high_minus_low",
    ]
    return df[keep]


def load_segx_sample(cohort: str) -> pd.DataFrame:
    df = read_csv(RESULTS_DIR / cohort / "segxgradcam" / "segx_sample_stats.csv")
    df.insert(0, "round", cohort)
    keep = [
        "round",
        "image_id",
        "split",
        "correct",
        "delta_high_minus_low_cd8_cam",
        "attention_weighted_minus_unweighted_actin",
        "spearman_actin_vs_cd8_cam",
        "partial_spearman_actin_vs_cd8_cam_given_ck",
        "residual_delta_high_minus_low_cd8_cam_given_ck",
        "ck_stratified_delta_high_minus_low_cd8_cam",
        "attention_weighted_minus_unweighted_distance",
    ]
    return df[keep]


def get_stat(cohort: str, source: str, metric: str, endpoint: str | None = None) -> dict:
    if source == "raw":
        df = read_csv(RESULTS_DIR / cohort / "raw_mask" / "cohort_stats_by_group.csv")
        sel = (
            (df["group"] == "all")
            & (pd.to_numeric(df["tile_size"], errors="coerce") == 512)
            & (df["metric"] == metric)
        )
        if endpoint is not None:
            sel &= df["endpoint"].eq(endpoint)
    elif source == "tumor":
        df = read_csv(RESULTS_DIR / cohort / "raw_mask" / "tumor_adjusted_cohort_stats_by_group.csv")
        sel = (
            (df["group"] == "all")
            & (pd.to_numeric(df["tile_size"], errors="coerce") == 512)
            & (df["endpoint"] == "tumor_adjusted_all_tiles")
            & (df["metric"] == metric)
        )
    elif source == "segx":
        df = read_csv(RESULTS_DIR / cohort / "segxgradcam" / "segx_cohort_stats.csv")
        sel = df["metric"].eq(metric)
    else:
        raise ValueError(source)
    rows = df[sel]
    if rows.empty:
        raise KeyError((cohort, source, metric, endpoint))
    row = rows.iloc[0].to_dict()
    return row


def format_num(value: float, digits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}g}"


def export_plotting_data() -> dict[str, pd.DataFrame]:
    data = {
        "raw_tile_metrics_512": pd.concat([load_raw_tiles("r1"), load_raw_tiles("r2")], ignore_index=True),
        "raw_sample_primary_512": pd.concat([load_raw_sample("r1"), load_raw_sample("r2")], ignore_index=True),
        "tumor_adjusted_sample_512": pd.concat(
            [load_tumor_adjusted_sample("r1"), load_tumor_adjusted_sample("r2")],
            ignore_index=True,
        ),
        "segx_sample_summary": pd.concat([load_segx_sample("r1"), load_segx_sample("r2")], ignore_index=True),
    }

    summary_rows = []
    for cohort in ("r1", "r2"):
        for source, metric, endpoint, label in [
            ("raw", "spearman_rho", "raw_all_tiles", "Raw Spearman rho"),
            ("raw", "delta_high_minus_low", "raw_all_tiles", "Raw high-minus-low CD8 delta"),
            ("tumor", "partial_spearman_actin_cd8_given_ck", None, "Partial rho | CK"),
            ("tumor", "residual_delta_high_minus_low", None, "CK-adjusted CD8 residual delta"),
            ("tumor", "ck_stratified_delta_high_minus_low", None, "CK-stratified CD8 delta"),
            ("segx", "delta_high_minus_low_cd8_cam", None, "SegX CD8-CAM high-minus-low delta"),
            ("segx", "partial_spearman_actin_vs_cd8_cam_given_ck", None, "SegX partial rho | CK"),
            ("segx", "ck_stratified_delta_high_minus_low_cd8_cam", None, "SegX CK-stratified delta"),
        ]:
            row = get_stat(cohort, source, metric, endpoint)
            summary_rows.append({
                "round": cohort,
                "source": source,
                "metric": metric,
                "label": label,
                "n_samples": row.get("n_samples", ""),
                "mean": row.get("mean", np.nan),
                "median": row.get("median", np.nan),
                "bootstrap_ci_low": row.get("bootstrap_ci_low", np.nan),
                "bootstrap_ci_high": row.get("bootstrap_ci_high", np.nan),
                "wilcoxon_p": row.get("wilcoxon_p", np.nan),
            })
    data["summary_statistics"] = pd.DataFrame(summary_rows)

    for name, df in data.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
    return data


def strip_boxplot(ax, df: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    cohorts = ["r1", "r2"]
    values = [
        pd.to_numeric(df[df["round"] == cohort][metric], errors="coerce").dropna().to_numpy()
        for cohort in cohorts
    ]
    bp = ax.boxplot(values, positions=[1, 2], widths=0.45, patch_artist=True, showfliers=False)
    for patch, cohort in zip(bp["boxes"], cohorts):
        patch.set_facecolor(COLORS[cohort])
        patch.set_alpha(0.24)
        patch.set_edgecolor(COLORS[cohort])
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color("0.25")
            artist.set_linewidth(1.0)
    rng = np.random.default_rng(1)
    for idx, (cohort, vals) in enumerate(zip(cohorts, values), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(np.full(vals.size, idx) + jitter, vals, s=12, alpha=0.45, color=COLORS[cohort], edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([COHORT_LABELS[c] for c in cohorts])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.22)


def fig1_raw_tile_density(raw_tiles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=160, sharex=True, sharey=True)
    for ax, cohort in zip(axes, ["r1", "r2"]):
        sub = raw_tiles[raw_tiles["round"] == cohort]
        hb = ax.hexbin(
            sub["actin_frac"],
            sub["cd8_frac"],
            gridsize=45,
            mincnt=1,
            cmap="viridis",
            linewidths=0,
        )
        ax.set_title(f"{COHORT_LABELS[cohort]} 512px tiles")
        ax.set_xlabel("Actin fraction")
        ax.grid(True, alpha=0.18)
        fig.colorbar(hb, ax=ax, label="Tile count")
    axes[0].set_ylabel("CD8 fraction")
    fig.suptitle("Local actin and CD8 density remain positively associated")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_raw_tile_density.svg", bbox_inches="tight")
    plt.close(fig)


def fig2_raw_sample_effects(raw_sample: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=160)
    strip_boxplot(axes[0], raw_sample, "spearman_rho", "Per-sample Spearman rho", "Actin vs CD8")
    strip_boxplot(axes[1], raw_sample, "delta_high_minus_low", "High-minus-low CD8 fraction", "CD8 in high vs low actin")
    fig.suptitle("Raw local CD8 does not decrease in actin-rich tiles")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_raw_sample_effects.svg", bbox_inches="tight")
    plt.close(fig)


def fig3_tumor_adjusted(tumor_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=160)
    strip_boxplot(
        axes[0],
        tumor_df,
        "partial_spearman_actin_cd8_given_ck",
        "Partial Spearman rho",
        "Actin vs CD8 | CK",
    )
    strip_boxplot(
        axes[1],
        tumor_df,
        "residual_delta_high_minus_low",
        "Residual CD8 delta",
        "CD8 residual after CK",
    )
    strip_boxplot(
        axes[2],
        tumor_df,
        "ck_stratified_delta_high_minus_low",
        "CK-stratified CD8 delta",
        "Matched CK strata",
    )
    fig.suptitle("Tumor-adjusted endpoints remain non-negative")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_tumor_adjusted_effects.svg", bbox_inches="tight")
    plt.close(fig)


def fig4_segx_attention(segx_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), dpi=160)
    strip_boxplot(
        axes[0],
        segx_df,
        "delta_high_minus_low_cd8_cam",
        "CD8-CAM delta",
        "High vs low actin",
    )
    strip_boxplot(
        axes[1],
        segx_df,
        "attention_weighted_minus_unweighted_actin",
        "Actin context shift",
        "CAM-weighted actin",
    )
    strip_boxplot(
        axes[2],
        segx_df,
        "partial_spearman_actin_vs_cd8_cam_given_ck",
        "Partial rho",
        "CD8-CAM vs actin | CK",
    )
    strip_boxplot(
        axes[3],
        segx_df,
        "ck_stratified_delta_high_minus_low_cd8_cam",
        "CK-stratified CD8-CAM delta",
        "Matched CK strata",
    )
    fig.suptitle("PDO-change model attention does not prefer low-actin CD8")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_segx_attention_effects.svg", bbox_inches="tight")
    plt.close(fig)


def fig5_summary(summary: pd.DataFrame) -> None:
    rows = summary[
        summary["metric"].isin([
            "delta_high_minus_low",
            "ck_stratified_delta_high_minus_low",
            "delta_high_minus_low_cd8_cam",
            "ck_stratified_delta_high_minus_low_cd8_cam",
        ])
    ].copy()
    rows["order"] = rows["label"].map({
        "Raw high-minus-low CD8 delta": 0,
        "CK-stratified CD8 delta": 1,
        "SegX CD8-CAM high-minus-low delta": 2,
        "SegX CK-stratified delta": 3,
    })
    rows = rows.sort_values(["order", "round"])
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    y_positions = np.arange(len(rows))[::-1]
    for y, (_, row) in zip(y_positions, rows.iterrows()):
        cohort = row["round"]
        ax.scatter(float(row["median"]), y, color=COLORS[cohort], s=38, zorder=3)
        if pd.notna(row.get("bootstrap_ci_low")) and pd.notna(row.get("bootstrap_ci_high")):
            ax.plot(
                [float(row["bootstrap_ci_low"]), float(row["bootstrap_ci_high"])],
                [y, y],
                color=COLORS[cohort],
                linewidth=2,
                alpha=0.75,
            )
    labels = [f"{row['label']} ({COHORT_LABELS[row['round']]})" for _, row in rows.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Median high-minus-low effect")
    ax.set_title("Effect-size summary across rounds")
    ax.grid(True, axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_effect_size_summary.svg", bbox_inches="tight")
    plt.close(fig)


def build_figures(data: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig1_raw_tile_density(data["raw_tile_metrics_512"])
    fig2_raw_sample_effects(data["raw_sample_primary_512"])
    fig3_tumor_adjusted(data["tumor_adjusted_sample_512"])
    fig4_segx_attention(data["segx_sample_summary"])
    fig5_summary(data["summary_statistics"])


def stat_text(cohort: str) -> dict[str, str]:
    raw_rho = get_stat(cohort, "raw", "spearman_rho", "raw_all_tiles")
    raw_delta = get_stat(cohort, "raw", "delta_high_minus_low", "raw_all_tiles")
    partial = get_stat(cohort, "tumor", "partial_spearman_actin_cd8_given_ck")
    residual = get_stat(cohort, "tumor", "residual_delta_high_minus_low")
    strat = get_stat(cohort, "tumor", "ck_stratified_delta_high_minus_low")
    segx_delta = get_stat(cohort, "segx", "delta_high_minus_low_cd8_cam")
    segx_partial = get_stat(cohort, "segx", "partial_spearman_actin_vs_cd8_cam_given_ck")
    segx_strat = get_stat(cohort, "segx", "ck_stratified_delta_high_minus_low_cd8_cam")
    return {
        "n_raw": str(int(raw_rho["n_samples"])),
        "raw_rho": format_num(raw_rho["median"], 4),
        "raw_delta": format_num(raw_delta["median"], 4),
        "raw_delta_p": format_num(raw_delta["wilcoxon_p"], 4),
        "partial": format_num(partial["median"], 4),
        "residual": format_num(residual["median"], 4),
        "strat": format_num(strat["median"], 4),
        "strat_p": format_num(strat["wilcoxon_p"], 4),
        "segx_delta": format_num(segx_delta["median"], 4),
        "segx_delta_p": format_num(segx_delta["wilcoxon_p"], 4),
        "segx_partial": format_num(segx_partial["median"], 4),
        "segx_strat": format_num(segx_strat["median"], 4),
        "segx_strat_p": format_num(segx_strat["wilcoxon_p"], 4),
    }


def write_report() -> None:
    r1 = stat_text("r1")
    r2 = stat_text("r2")
    summary = {
        "r1": r1,
        "r2": r2,
    }
    (ROOT / "summary_values.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# On-chip Distribution Analysis

## Results: R1 Local CD8 Distribution After Accounting for Tumor Context
We tested whether local CD8/CAR-T signal is depleted from actin-rich neighborhoods while accounting for tumor distribution. In the R1 cohort, all train, validation, and test samples with matched binary masks were analyzed (n = {r1['n_raw']}). Actin was used as the stromal/CAF-proxy channel, CD8 as the CAR-T proxy, and CK as the tumor-context channel. Native-resolution masks were summarized using non-overlapping tiles at 256, 512, and 1024 px, with 512 px treated as the primary scale. The primary raw endpoint compared per-tile actin fraction with CD8 fraction; tumor-aware endpoints then controlled CK using partial Spearman correlation, CK-adjusted CD8 residuals, and CK-stratified high- vs low-actin comparisons. In parallel, predicted-class SegX-GradCAM was recomputed for the R1 PDO-change classifier, multiplied by the CD8 mask, and summarized by local actin and CK context in CD8-positive tiles.

R1 did not support the expected CAF-high/CD8-low exclusion pattern. At the 512 px scale, the raw per-sample actin-CD8 Spearman rho was positive (median {r1['raw_rho']}), and high-actin tiles had higher, not lower, CD8 fraction than low-actin tiles (median high-minus-low delta {r1['raw_delta']}; one-sided p(delta < 0) = {r1['raw_delta_p']}). This positive association weakened but remained positive after accounting for tumor context: partial rho(actin, CD8 | CK) was {r1['partial']}, CK-adjusted residual delta was {r1['residual']}, and CK-stratified delta was {r1['strat']} (p(delta < 0) = {r1['strat_p']}). Model attention showed the same direction rather than an opposite one: CD8 SegX-GradCAM attention was higher in high-actin tiles (median delta {r1['segx_delta']}; p(delta < 0) = {r1['segx_delta_p']}), and this persisted after CK adjustment (partial rho | CK {r1['segx_partial']}; CK-stratified CD8-CAM delta {r1['segx_strat']}; p(delta < 0) = {r1['segx_strat_p']}). Together, these results indicate that the R1 masks and model attention do not show CD8 avoidance of actin-rich regions after tumor distribution is considered.

![Figure 1. Local actin and CD8 density across 512 px tiles.](figures/fig1_raw_tile_density.svg)

![Figure 2. R1/R2 sample-level raw association endpoints.](figures/fig2_raw_sample_effects.svg)

![Figure 3. Tumor-adjusted raw mask endpoints.](figures/fig3_tumor_adjusted_effects.svg)

## Results: R2 Replication and Cross-round Consistency
The same analysis was repeated in the processed R2 cohort using all train and validation samples (n = {r2['n_raw']}). The raw-mask analysis used the identical actin/CD8/CK endpoint definitions and the same 512 px primary tile scale. For model-facing validation, predicted-class SegX-GradCAM was recomputed using the R2 optimal PDO-change classifier with CD8, CD68, actin, and CK inputs, then CD8-restricted attention was tested against local actin context with and without CK adjustment. This section therefore asks whether the R1 finding is reproduced in an independently processed R2 dataset and whether the model-attention readout is directionally consistent across rounds.

R2 was directionally consistent with R1. Raw local actin and CD8 were again positively associated (median rho {r2['raw_rho']}), and high-actin tiles had higher local CD8 than low-actin tiles (median delta {r2['raw_delta']}; p(delta < 0) = {r2['raw_delta_p']}). After controlling tumor/CK distribution, the association remained non-negative: partial rho(actin, CD8 | CK) was {r2['partial']}, CK-adjusted residual delta was {r2['residual']}, and CK-stratified delta was {r2['strat']} (p(delta < 0) = {r2['strat_p']}). R2 SegX-GradCAM also matched the R1 direction: CD8 attention was higher in high-actin context (median delta {r2['segx_delta']}; p(delta < 0) = {r2['segx_delta_p']}), and the CK-aware attention endpoints remained positive (partial rho | CK {r2['segx_partial']}; CK-stratified CD8-CAM delta {r2['segx_strat']}; p(delta < 0) = {r2['segx_strat_p']}). Thus, the R2 repeat does not rescue the CAF-high/CD8-low hypothesis; instead, both rounds show that local CD8 density and model CD8 attention tend to co-occur with actin-rich or actin-nearby contexts after tumor distribution is considered.

![Figure 4. SegX-GradCAM CD8 attention relative to actin and CK context.](figures/fig4_segx_attention_effects.svg)

![Figure 5. Effect-size summary across raw, tumor-adjusted, and SegX endpoints.](figures/fig5_effect_size_summary.svg)

## Reproducibility Notes
The code snapshots used to generate these analyses are stored in `code/`. The source result tables are stored under `results/r1/` and `results/r2/`. Plotting-ready CSV files are stored in `plotting_data/`, with one CSV corresponding to each figure family. Full-resolution raw pixel distance-gradient analysis was intentionally skipped for these exports because the R1 native masks made that endpoint slow; the tumor-aware conclusions here are based on tile-level CK adjustment and CK-stratified comparisons.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    copy_code_snapshots()
    data = export_plotting_data()
    build_figures(data)
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
