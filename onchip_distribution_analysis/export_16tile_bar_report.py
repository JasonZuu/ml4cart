"""Export 16x16-tile bar plots and report.

Figures are within-dataset bar plots only: low-actin versus high-actin local
regions in R1, and the same comparison in R2. Bars show the sample-level
median, with Q1-Q3 intervals. No paired points or lines are drawn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("on-chip_distribution-analysis")
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

COHORTS = ("r1", "r2")
COHORT_LABELS = {"r1": "R1", "r2": "R2"}
LOW_COLOR = "#377A6B"
HIGH_COLOR = "#B24A64"
FIGURE_DATA_FILES = {
    ("raw_cd8_density_16tile", "r1"): "fig_r1_raw_cd8_density_16tile_data.csv",
    ("segx_cd8cam_16tile", "r1"): "fig_r1_segx_cd8cam_16tile_data.csv",
    ("raw_cd8_density_16tile", "r2"): "fig_r2_raw_cd8_density_16tile_data.csv",
    ("segx_cd8cam_16tile", "r2"): "fig_r2_segx_cd8cam_16tile_data.csv",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt(value, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}g}"


def raw_dir(cohort: str) -> Path:
    return RESULTS_DIR / cohort / "raw_mask_16tile"


def segx_dir(cohort: str) -> Path:
    return RESULTS_DIR / cohort / "segxgradcam_16tile"


def load_raw_sample(cohort: str) -> pd.DataFrame:
    df = read_csv(raw_dir(cohort) / "raw16_sample_stats.csv")
    df["round"] = cohort
    return df


def load_segx_sample(cohort: str) -> pd.DataFrame:
    df = read_csv(segx_dir(cohort) / "segx_sample_stats.csv")
    df["round"] = cohort
    return df


def load_raw_stats(cohort: str) -> pd.DataFrame:
    return read_csv(raw_dir(cohort) / "raw16_cohort_stats.csv")


def load_segx_stats(cohort: str) -> pd.DataFrame:
    return read_csv(segx_dir(cohort) / "segx_cohort_stats.csv")


def stat_row(cohort: str, source: str, metric: str) -> dict:
    df = load_raw_stats(cohort) if source == "raw" else load_segx_stats(cohort)
    rows = df[df["metric"].eq(metric)]
    if rows.empty:
        raise KeyError((cohort, source, metric))
    return rows.iloc[0].to_dict()


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "mean": float("nan"),
            "sd": float("nan"),
        }
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    return {
        "n": int(vals.size),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "mean": float(np.mean(vals)),
        "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
    }


def paired_long(df: pd.DataFrame, cohort: str, low_col: str, high_col: str, id_col: str, value_col: str, delta_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        base = {"round": cohort, "sample_id": row[id_col], "split": row.get("split", "")}
        rows.append({**base, "actin_group": "low_actin", value_col: row[low_col], "high_minus_low": row[delta_col]})
        rows.append({**base, "actin_group": "high_actin", value_col: row[high_col], "high_minus_low": row[delta_col]})
    return pd.DataFrame(rows)


def bar_summary_from_long(figure: str, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        for group in ("low_actin", "high_actin"):
            vals = pd.to_numeric(df[(df["round"] == cohort) & (df["actin_group"] == group)][value_col], errors="coerce")
            summary = distribution_summary(vals.to_numpy(dtype=np.float64))
            rows.append({
                "figure": figure,
                "round": cohort,
                "actin_group": group,
                "value_name": value_col,
                **summary,
            })
    return pd.DataFrame(rows)


def export_figure_data(bar_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for (figure, cohort), filename in FIGURE_DATA_FILES.items():
        out = bar_summary[
            bar_summary["figure"].eq(figure) & bar_summary["round"].eq(cohort)
        ].copy()
        out["actin_region"] = out["actin_group"].map({
            "low_actin": "actin-sparse Q1",
            "high_actin": "actin-dense Q4",
        })
        out["error_low"] = out["median"] - out["q1"]
        out["error_high"] = out["q3"] - out["median"]
        out = out[
            [
                "figure",
                "round",
                "actin_region",
                "value_name",
                "n",
                "median",
                "q1",
                "q3",
                "error_low",
                "error_high",
            ]
        ]
        out.to_csv(FIG_DIR / filename, index=False)


def export_plotting_data() -> dict[str, pd.DataFrame]:
    raw_samples = pd.concat([load_raw_sample(c) for c in COHORTS], ignore_index=True)
    segx_samples = pd.concat([load_segx_sample(c) for c in COHORTS], ignore_index=True)

    raw_long = pd.concat([
        paired_long(
            load_raw_sample(cohort),
            cohort,
            "cd8_low_actin_q1_mean",
            "cd8_high_actin_q4_mean",
            "source_image_id",
            "cd8_fraction",
            "delta_high_minus_low",
        )
        for cohort in COHORTS
    ], ignore_index=True)
    segx_long = pd.concat([
        paired_long(
            load_segx_sample(cohort),
            cohort,
            "low_caf_cd8_cam_mean",
            "high_caf_cd8_cam_mean",
            "image_id",
            "cd8_cam_mean",
            "delta_high_minus_low_cd8_cam",
        )
        for cohort in COHORTS
    ], ignore_index=True)

    bar_summary = pd.concat([
        bar_summary_from_long("raw_cd8_density_16tile", raw_long, "cd8_fraction"),
        bar_summary_from_long("segx_cd8cam_16tile", segx_long, "cd8_cam_mean"),
    ], ignore_index=True)

    stat_rows = []
    for cohort in COHORTS:
        for source, metric, label in [
            ("raw", "delta_high_minus_low", "Raw CD8 difference between actin-dense and actin-sparse tiles"),
            ("raw", "residual_delta_high_minus_low", "CK-adjusted raw CD8 difference between actin-dense and actin-sparse tiles"),
            ("raw", "ck_stratified_delta_high_minus_low", "CK-stratified raw CD8 difference between actin-dense and actin-sparse tiles"),
            ("segx", "delta_high_minus_low_cd8_cam", "SegX CD8-CAM difference between actin-dense and actin-sparse tiles"),
            ("segx", "residual_delta_high_minus_low_cd8_cam_given_ck", "CK-adjusted SegX CD8-CAM difference between actin-dense and actin-sparse tiles"),
            ("segx", "ck_stratified_delta_high_minus_low_cd8_cam", "CK-stratified SegX CD8-CAM difference between actin-dense and actin-sparse tiles"),
        ]:
            row = stat_row(cohort, source, metric)
            stat_rows.append({
                "round": cohort,
                "source": source,
                "metric": metric,
                "label": label,
                "n_samples": row.get("n_samples", ""),
                "mean": row.get("mean", np.nan),
                "median": row.get("median", np.nan),
                "sd": row.get("sd", np.nan),
                "sem": row.get("sem", np.nan),
            })
    summary_stats = pd.DataFrame(stat_rows)

    export_figure_data(bar_summary)
    data = {
        "raw16_sample_stats": raw_samples,
        "segx16_sample_stats": segx_samples,
        "raw16_cd8_low_high": raw_long,
        "segx16_cd8cam_low_high": segx_long,
        "barplot_summary_16tile": bar_summary,
        "summary_statistics_16tile": summary_stats,
    }
    return data


def plot_bar_for_cohort(
    cohort: str,
    long_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_name: str,
) -> None:
    sub = long_df[long_df["round"] == cohort]
    medians = []
    err_low = []
    err_high = []
    for group in ("low_actin", "high_actin"):
        vals = pd.to_numeric(sub[sub["actin_group"] == group][value_col], errors="coerce").to_numpy(dtype=np.float64)
        summary = distribution_summary(vals)
        medians.append(summary["median"])
        err_low.append(summary["median"] - summary["q1"])
        err_high.append(summary["q3"] - summary["median"])

    fig, ax = plt.subplots(figsize=(3.8, 4.2), dpi=180)
    ax.bar(
        [0, 1],
        medians,
        yerr=np.asarray([err_low, err_high], dtype=np.float64),
        color=[LOW_COLOR, HIGH_COLOR],
        edgecolor="0.25",
        linewidth=0.9,
        width=0.62,
        capsize=5,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "0.2"},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Actin-sparse\nQ1", "Actin-dense\nQ4"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def build_figures(data: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    for cohort in COHORTS:
        plot_bar_for_cohort(
            cohort,
            data["raw16_cd8_low_high"],
            "cd8_fraction",
            "CD8 fraction",
            f"{COHORT_LABELS[cohort]} raw mask: 16x16 tiles",
            f"fig_{cohort}_raw_cd8_density_16tile.svg",
        )
        plot_bar_for_cohort(
            cohort,
            data["segx16_cd8cam_low_high"],
            "cd8_cam_mean",
            "CD8-restricted GradCAM intensity",
            f"{COHORT_LABELS[cohort]} SegX-GradCAM: 16x16 tiles",
            f"fig_{cohort}_segx_cd8cam_16tile.svg",
        )


def column_summary(df: pd.DataFrame, column: str) -> dict[str, float | int]:
    return distribution_summary(pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64))


def iqr_text(summary: dict[str, float | int]) -> str:
    return f"{fmt(summary['q1'])}-{fmt(summary['q3'])}"


def stats_text(cohort: str) -> dict[str, str]:
    raw_samples = load_raw_sample(cohort)
    segx_samples = load_segx_sample(cohort)
    raw_rho = stat_row(cohort, "raw", "spearman_actin_cd8")
    raw_delta = stat_row(cohort, "raw", "delta_high_minus_low")
    raw_partial = stat_row(cohort, "raw", "partial_spearman_actin_cd8_given_ck")
    raw_resid = stat_row(cohort, "raw", "residual_delta_high_minus_low")
    raw_strat = stat_row(cohort, "raw", "ck_stratified_delta_high_minus_low")
    segx_delta = stat_row(cohort, "segx", "delta_high_minus_low_cd8_cam")
    segx_partial = stat_row(cohort, "segx", "partial_spearman_actin_vs_cd8_cam_given_ck")
    segx_resid = stat_row(cohort, "segx", "residual_delta_high_minus_low_cd8_cam_given_ck")
    segx_strat = stat_row(cohort, "segx", "ck_stratified_delta_high_minus_low_cd8_cam")
    raw_low = column_summary(raw_samples, "cd8_low_actin_q1_mean")
    raw_high = column_summary(raw_samples, "cd8_high_actin_q4_mean")
    segx_low = column_summary(segx_samples, "low_caf_cd8_cam_mean")
    segx_high = column_summary(segx_samples, "high_caf_cd8_cam_mean")
    return {
        "n": str(int(raw_delta["n_samples"])),
        "segx_n": str(int(segx_delta["n_samples"])),
        "raw_low": fmt(raw_low["median"]),
        "raw_low_iqr": iqr_text(raw_low),
        "raw_high": fmt(raw_high["median"]),
        "raw_high_iqr": iqr_text(raw_high),
        "raw_rho": fmt(raw_rho["median"]),
        "raw_delta": fmt(raw_delta["median"]),
        "raw_partial": fmt(raw_partial["median"]),
        "raw_resid": fmt(raw_resid["median"]),
        "raw_strat": fmt(raw_strat["median"]),
        "segx_low": fmt(segx_low["median"]),
        "segx_low_iqr": iqr_text(segx_low),
        "segx_high": fmt(segx_high["median"]),
        "segx_high_iqr": iqr_text(segx_high),
        "segx_delta": fmt(segx_delta["median"]),
        "segx_partial": fmt(segx_partial["median"]),
        "segx_resid": fmt(segx_resid["median"]),
        "segx_strat": fmt(segx_strat["median"]),
    }


def write_report() -> None:
    r1 = stats_text("r1")
    r2 = stats_text("r2")
    (ROOT / "summary_values.json").write_text(json.dumps({"r1": r1, "r2": r2}, indent=2), encoding="utf-8")
    report = f"""# On-chip Distribution Analysis

## R1: CD8 distribution across actin-defined microregions
To quantify local variation in CD8 signal across actin-defined microenvironments, native-resolution binary actin, CD8 and CK masks were partitioned into non-overlapping 16 x 16 pixel tiles. Actin density was calculated as the fraction of actin-positive pixels in each tile; CD8 and CK densities were computed by the same positive-pixel fraction. Within each sample, actin-sparse and actin-dense microregions were defined as the bottom and top quartiles of tile-level actin density, respectively. R1 included all train, validation and test samples with matched actin, CD8 and CK masks (n = {r1['n']}). For the model-attribution analysis, predicted-class GradCAM was computed once over the full 512 x 512 model input, restricted to CD8-positive pixels, and then summarized as the mean continuous GradCAM intensity within each 16 x 16 tile.

In R1, the median sample-level CD8 fraction was {r1['raw_low']} in actin-sparse tiles (Q1-Q3, {r1['raw_low_iqr']}) and {r1['raw_high']} in actin-dense tiles (Q1-Q3, {r1['raw_high_iqr']}). The median within-sample difference between actin-dense and actin-sparse microregions was {r1['raw_delta']}, and the median per-sample Spearman correlation between actin and CD8 density was {r1['raw_rho']}. This pattern was retained after incorporating tumor distribution: the median partial correlation of actin and CD8 density conditional on CK density was {r1['raw_partial']}, the median CK-adjusted residual difference was {r1['raw_resid']}, and the median CK-stratified difference was {r1['raw_strat']}. The CD8-restricted SegX-GradCAM analysis showed a corresponding shift in model attribution. Median GradCAM intensity was {r1['segx_low']} in actin-sparse CD8-positive tiles (Q1-Q3, {r1['segx_low_iqr']}) and {r1['segx_high']} in actin-dense CD8-positive tiles (Q1-Q3, {r1['segx_high_iqr']}), with a median within-sample difference of {r1['segx_delta']}. CK-aware SegX summaries were also positive (partial correlation conditional on CK, {r1['segx_partial']}; CK-adjusted residual difference, {r1['segx_resid']}; CK-stratified difference, {r1['segx_strat']}). Thus, in R1, CD8 signal and CD8-restricted model attribution were higher in actin-dense than actin-sparse microregions at the 16 x 16 tile scale.

![Figure 1. R1 raw CD8 density in 16 x 16 actin-sparse and actin-dense tiles. Bars show the median across samples; error bars show Q1-Q3.](figures/fig_r1_raw_cd8_density_16tile.svg)

![Figure 2. R1 CD8-restricted SegX-GradCAM intensity summarized in 16 x 16 actin-sparse and actin-dense tiles. Bars show the median across samples; error bars show Q1-Q3.](figures/fig_r1_segx_cd8cam_16tile.svg)

## R2: CD8 distribution across actin-defined microregions
The same 16 x 16 tile analysis was applied to R2 using all train and validation samples with matched actin, CD8 and CK masks (n = {r2['n']}). Raw mask densities were computed from native-resolution masks as positive-pixel fractions per tile, and actin-sparse and actin-dense microregions were defined within each sample as the lowest and highest quartiles of actin density. Whole-image predicted-class GradCAM was similarly restricted to CD8-positive pixels before the resulting continuous attribution map was summarized over the same 16 x 16 tiles.

In R2, the median sample-level CD8 fraction was {r2['raw_low']} in actin-sparse tiles (Q1-Q3, {r2['raw_low_iqr']}) and {r2['raw_high']} in actin-dense tiles (Q1-Q3, {r2['raw_high_iqr']}). The median within-sample difference between actin-dense and actin-sparse microregions was {r2['raw_delta']}, with a median Spearman correlation of {r2['raw_rho']}. Tumor-aware summaries gave the same directionality: the median partial correlation conditional on CK density was {r2['raw_partial']}, the median CK-adjusted residual difference was {r2['raw_resid']}, and the median CK-stratified difference was {r2['raw_strat']}. In the SegX-GradCAM analysis, median CD8-restricted attribution was {r2['segx_low']} in actin-sparse CD8-positive tiles (Q1-Q3, {r2['segx_low_iqr']}) and {r2['segx_high']} in actin-dense CD8-positive tiles (Q1-Q3, {r2['segx_high_iqr']}), with a median within-sample difference of {r2['segx_delta']}. CK-aware attribution summaries remained positive (partial correlation conditional on CK, {r2['segx_partial']}; CK-adjusted residual difference, {r2['segx_resid']}; CK-stratified difference, {r2['segx_strat']}). Therefore, in R2, actin-dense microregions contained higher raw CD8 signal and higher CD8-restricted model attribution than actin-sparse microregions under this 16 x 16 tile definition.

![Figure 3. R2 raw CD8 density in 16 x 16 actin-sparse and actin-dense tiles. Bars show the median across samples; error bars show Q1-Q3.](figures/fig_r2_raw_cd8_density_16tile.svg)

![Figure 4. R2 CD8-restricted SegX-GradCAM intensity summarized in 16 x 16 actin-sparse and actin-dense tiles. Bars show the median across samples; error bars show Q1-Q3.](figures/fig_r2_segx_cd8cam_16tile.svg)
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")


def main() -> int:
    data = export_plotting_data()
    build_figures(data)
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
