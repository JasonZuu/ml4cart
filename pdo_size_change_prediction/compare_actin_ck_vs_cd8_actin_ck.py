"""Direct comparison of `actin_ck` vs `cd8_actin_ck` PDO-change models.

Both models reach identical val accuracy (6/8) and macro-F1 because their
argmax predictions agree on every val sample.  The interesting signal lives
in the *softmax distributions*: how much probability mass each model puts on
the true bin and how spread that mass is across neighbouring bins.

This script consumes the `prob_distribution.csv` files already produced by
`draw_prob_distribution.py` for both models and emits:

  * `<split>_per_sample.csv`   — per-sample probability-based metrics
  * `<split>_summary.csv`      — aggregate metrics (mean over samples)
  * `<split>_prob_overlay.{png,svg}`     — 12-bin curves overlaid
  * `<split>_ptrue_bars.{png,svg}`       — P(true bin) per sample
  * `<split>_metric_summary.{png,svg}`   — aggregate metric bars
  * `comparison_report.md`     — text summary

Run from repo root:

    python onchip_pdochange_prediction/compare_actin_ck_vs_cd8_actin_ck.py
"""
import argparse
import csv
import math
import sys
from pathlib import Path

# Ensure repo root is in path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.pdochange_data import PDO_CHANGE_BIN_LABELS


MODEL_A = "actin_ck"           # 2-channel: actin + CK19
MODEL_B = "cd8_actin_ck"       # 3-channel: CD8 + actin + CK19
NUM_BINS = len(PDO_CHANGE_BIN_LABELS)

# Same grouping used by draw_prob_distribution.py
GROUPS = [
    ("Shrinkage  (PDO < -20%)",       0,  4),
    ("Near-zero  (-20 <= PDO < 20%)", 5,  6),
    ("Growth     (PDO >= 20%)",        7, 11),
]

# Consistent colours across all figures
COLOR_A = "#1f77b4"   # blue   - actin_ck
COLOR_B = "#d62728"   # red    - cd8_actin_ck

EPS = 1e-12


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare actin_ck vs cd8_actin_ck softmax distributions."
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("onchip_pdochange_prediction/results/r1"),
        help="Root holding the two model directories (actin_ck/ and cd8_actin_ck/).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory.  Defaults to <results-root>/comparison_actin_ck_vs_cd8_actin_ck.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["val", "r4"],
        choices=["val", "r4"],
        help="Which splits to compare.  'val' uses val_analysis/, 'r4' uses r4_drug_eval/.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

SPLIT_TO_SUBDIR = {
    "val": "val_analysis",
    "r4":  "r4_drug_eval",
}


def _prob_csv_path(results_root: Path, model: str, split: str) -> Path:
    return results_root / model / SPLIT_TO_SUBDIR[split] / "prob_distribution.csv"


def load_prob_records(csv_path: Path) -> list[dict]:
    """Read a `prob_distribution.csv` as a list of records.

    Each record contains: image_id, true_bin_idx (int), pred_bin_idx (int),
    probs (np.ndarray of length NUM_BINS).
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing prob CSV: {csv_path}")
    prob_cols = [f"prob_{lbl}" for lbl in PDO_CHANGE_BIN_LABELS]
    records: list[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            probs = np.array([float(row[c]) for c in prob_cols], dtype=np.float64)
            # Re-normalise defensively in case of FP drift in the CSV writer
            s = float(probs.sum())
            if s > 0:
                probs = probs / s
            records.append({
                "image_id":     str(row["image_id"]),
                "true_bin_idx": int(row["true_bin_idx"]),
                "pred_bin_idx": int(row["pred_bin_idx"]),
                "probs":        probs,
            })
    return records


def align_records(recs_a: list[dict], recs_b: list[dict]) -> list[tuple[dict, dict]]:
    """Pair records from model A and model B by image_id, preserving order of A."""
    by_id_b = {r["image_id"]: r for r in recs_b}
    pairs: list[tuple[dict, dict]] = []
    for ra in recs_a:
        rb = by_id_b.get(ra["image_id"])
        if rb is None:
            raise KeyError(f"Image {ra['image_id']!r} present in {MODEL_A} but not {MODEL_B}.")
        if rb["true_bin_idx"] != ra["true_bin_idx"]:
            raise ValueError(
                f"True-bin mismatch for {ra['image_id']!r}: "
                f"{ra['true_bin_idx']} vs {rb['true_bin_idx']}"
            )
        pairs.append((ra, rb))
    return pairs


# ---------------------------------------------------------------------------
# Probability-based metrics
# ---------------------------------------------------------------------------

def compute_metrics(rec: dict) -> dict:
    """Return a dict of probability-aware metrics for a single record.

    Notes:
      - Brier and RPS are *proper scoring rules* for the categorical outcome
        (RPS being the appropriate one for ordinal bins).
      - EAD is the expected |pred_bin - true_bin| under the predicted
        distribution; useful because PDO-change bins are ordinal.
    """
    probs = rec["probs"]
    true_idx = rec["true_bin_idx"]
    p_true = float(probs[true_idx])

    # One-hot
    one_hot = np.zeros(NUM_BINS, dtype=np.float64)
    one_hot[true_idx] = 1.0

    # Brier score (sum of squared errors)
    brier = float(np.sum((probs - one_hot) ** 2))

    # Negative log-likelihood
    nll = float(-math.log(max(p_true, EPS)))

    # Ranked probability score (lower = better)
    cdf_pred = np.cumsum(probs)
    cdf_true = np.cumsum(one_hot)
    rps = float(np.sum((cdf_pred - cdf_true) ** 2))

    # Expected absolute bin distance under predicted distribution
    bin_idx = np.arange(NUM_BINS, dtype=np.float64)
    ead = float(np.sum(probs * np.abs(bin_idx - true_idx)))

    # Argmax error (in bins) - reported for completeness
    pred_idx = int(np.argmax(probs))
    abs_bin_err = abs(pred_idx - true_idx)

    return {
        "p_true": p_true,
        "brier": brier,
        "nll": nll,
        "rps": rps,
        "ead": ead,
        "pred_bin_idx": pred_idx,
        "abs_bin_err": abs_bin_err,
    }


def group_for_true_bin(idx: int) -> str:
    for name, lo, hi in GROUPS:
        if lo <= idx <= hi:
            return name
    return "(out of range)"


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

PER_SAMPLE_FIELDS = [
    "image_id", "group", "true_bin_idx", "true_bin_label",
    "p_true_A", "p_true_B", "delta_p_true",
    "brier_A", "brier_B", "delta_brier",
    "nll_A", "nll_B", "delta_nll",
    "rps_A", "rps_B", "delta_rps",
    "ead_A", "ead_B", "delta_ead",
    "pred_bin_A", "pred_bin_B", "argmax_match",
    "correct_A", "correct_B",
]


def write_per_sample_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SAMPLE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  Saved {out_path}")


SUMMARY_FIELDS = [
    "metric", "model_A_actin_ck", "model_B_cd8_actin_ck",
    "delta_B_minus_A", "better_for_B",
]


def write_summary_csv(summary: dict, out_path: Path) -> None:
    """summary[metric] = (val_A, val_B, lower_is_better)"""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for metric, (va, vb, lower_is_better) in summary.items():
            delta = vb - va
            if lower_is_better:
                better_b = delta < 0
            else:
                better_b = delta > 0
            w.writerow({
                "metric": metric,
                "model_A_actin_ck": f"{va:.6f}",
                "model_B_cd8_actin_ck": f"{vb:.6f}",
                "delta_B_minus_A": f"{delta:+.6f}",
                "better_for_B": "yes" if better_b else "no",
            })
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: Path, stem: str) -> None:
    for ext in ("png", "svg"):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_ptrue_bars(per_sample: list[dict], split: str, out_dir: Path) -> None:
    """Side-by-side bar chart of P(true bin) for every sample.

    Higher = the model puts more mass on the correct bin.
    """
    n = len(per_sample)
    x = np.arange(n)
    width = 0.38
    pa = np.array([r["p_true_A"] for r in per_sample], dtype=np.float64)
    pb = np.array([r["p_true_B"] for r in per_sample], dtype=np.float64)
    labels = [r["image_id"].split("_", 1)[-1] for r in per_sample]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.85 * n + 3.5), 4.2))
    ax.bar(x - width / 2, pa, width, color=COLOR_A,
           label=f"{MODEL_A} (actin + CK19)")
    ax.bar(x + width / 2, pb, width, color=COLOR_B,
           label=f"{MODEL_B} (CD8 + actin + CK19)")

    # Annotate deltas above each pair
    for xi, va, vb in zip(x, pa, pb):
        delta = vb - va
        sign = "+" if delta >= 0 else ""
        top = max(va, vb)
        ax.text(xi, top + 0.025, f"{sign}{delta:.2f}",
                ha="center", va="bottom", fontsize=7,
                color="black" if delta >= 0 else "#666")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P(true bin)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"Probability mass on the true bin per sample - {split} set\n"
        f"Higher is better.  Delta = B - A annotated above each pair."
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, out_dir, f"{split}_ptrue_bars")
    print(f"  Saved {split}_ptrue_bars.{{png,svg}}")


def plot_prob_overlay(pairs: list[tuple[dict, dict]], split: str, out_dir: Path) -> None:
    """Per-sample 12-bin softmax curves overlaid for both models.

    Layout: small multiples, one panel per sample, grouped by PDO-change category.
    """
    # Order panels by group, then sample order
    indexed = []
    for i, (ra, rb) in enumerate(pairs):
        g_idx = next((k for k, (_, lo, hi) in enumerate(GROUPS)
                      if lo <= ra["true_bin_idx"] <= hi), len(GROUPS))
        indexed.append((g_idx, i, ra, rb))
    indexed.sort(key=lambda t: (t[0], t[1]))

    n = len(indexed)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.3 * ncols, 2.6 * nrows + 0.6),
                             sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    x = np.arange(NUM_BINS)

    for k, (_, _, ra, rb) in enumerate(indexed):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        true_idx = ra["true_bin_idx"]
        ax.axvspan(true_idx - 0.5, true_idx + 0.5,
                   color="#dddddd", alpha=0.6, zorder=0)
        ax.plot(x, ra["probs"], marker="o", markersize=4, linewidth=1.5,
                color=COLOR_A, label=MODEL_A)
        ax.plot(x, rb["probs"], marker="s", markersize=4, linewidth=1.5,
                color=COLOR_B, label=MODEL_B)
        ax.set_title(ra["image_id"].split("_", 1)[-1] +
                     f"  (true={PDO_CHANGE_BIN_LABELS[true_idx]})",
                     fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PDO_CHANGE_BIN_LABELS, rotation=60,
                           ha="right", fontsize=5.5)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)

    # Hide any spare axes
    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].set_visible(False)

    # One legend for the whole figure
    handles = [
        plt.Line2D([0], [0], color=COLOR_A, marker="o", label=f"{MODEL_A} (actin + CK19)"),
        plt.Line2D([0], [0], color=COLOR_B, marker="s", label=f"{MODEL_B} (CD8 + actin + CK19)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.suptitle(f"Per-sample softmax probability distributions - {split} set",
                 fontsize=11, y=1.05)
    fig.tight_layout()
    _save_fig(fig, out_dir, f"{split}_prob_overlay")
    print(f"  Saved {split}_prob_overlay.{{png,svg}}")


def plot_metric_summary(summary: dict, split: str, out_dir: Path) -> None:
    """Aggregate metric bar chart with the two models side-by-side."""
    metrics = list(summary.keys())
    n = len(metrics)
    x = np.arange(n)
    width = 0.38
    va = np.array([summary[m][0] for m in metrics], dtype=np.float64)
    vb = np.array([summary[m][1] for m in metrics], dtype=np.float64)
    lower_is_better = [summary[m][2] for m in metrics]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * n + 2.0), 4.0))
    ax.bar(x - width / 2, va, width, color=COLOR_A,
           label=f"{MODEL_A} (actin + CK19)")
    ax.bar(x + width / 2, vb, width, color=COLOR_B,
           label=f"{MODEL_B} (CD8 + actin + CK19)")

    for xi, a, b, lib in zip(x, va, vb, lower_is_better):
        delta = b - a
        better_b = (delta < 0) if lib else (delta > 0)
        color = "#137a13" if better_b else "#a51111"
        sign = "+" if delta >= 0 else ""
        ax.text(xi, max(a, b) + 0.02 * max(va.max(), vb.max(), 1e-9),
                f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=8, color=color)

    labels = [
        m + ("\n(lower better)" if lib else "\n(higher better)")
        for m, lib in zip(metrics, lower_is_better)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Metric value (mean over samples)")
    ax.set_title(f"Aggregate probability-based metrics - {split} set")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, out_dir, f"{split}_metric_summary")
    print(f"  Saved {split}_metric_summary.{{png,svg}}")


# ---------------------------------------------------------------------------
# Per-split driver
# ---------------------------------------------------------------------------

def compare_split(split: str, results_root: Path, out_dir: Path) -> dict:
    print(f"\n=== Comparing split: {split} ===")
    csv_a = _prob_csv_path(results_root, MODEL_A, split)
    csv_b = _prob_csv_path(results_root, MODEL_B, split)
    print(f"  A: {csv_a}")
    print(f"  B: {csv_b}")
    recs_a = load_prob_records(csv_a)
    recs_b = load_prob_records(csv_b)
    pairs = align_records(recs_a, recs_b)
    print(f"  Aligned {len(pairs)} samples.")

    per_sample_rows: list[dict] = []
    metrics_a = {"p_true": [], "brier": [], "nll": [], "rps": [], "ead": [], "abs_bin_err": []}
    metrics_b = {k: [] for k in metrics_a}
    correct_a = 0
    correct_b = 0
    argmax_match = 0

    for ra, rb in pairs:
        ma = compute_metrics(ra)
        mb = compute_metrics(rb)
        for k in metrics_a:
            metrics_a[k].append(ma[k])
            metrics_b[k].append(mb[k])
        ca = (ma["pred_bin_idx"] == ra["true_bin_idx"])
        cb = (mb["pred_bin_idx"] == rb["true_bin_idx"])
        correct_a += int(ca)
        correct_b += int(cb)
        argmax_match += int(ma["pred_bin_idx"] == mb["pred_bin_idx"])

        per_sample_rows.append({
            "image_id":      ra["image_id"],
            "group":         group_for_true_bin(ra["true_bin_idx"]),
            "true_bin_idx":  ra["true_bin_idx"],
            "true_bin_label": PDO_CHANGE_BIN_LABELS[ra["true_bin_idx"]],
            "p_true_A":      f"{ma['p_true']:.6f}",
            "p_true_B":      f"{mb['p_true']:.6f}",
            "delta_p_true":  f"{mb['p_true'] - ma['p_true']:+.6f}",
            "brier_A":       f"{ma['brier']:.6f}",
            "brier_B":       f"{mb['brier']:.6f}",
            "delta_brier":   f"{mb['brier'] - ma['brier']:+.6f}",
            "nll_A":         f"{ma['nll']:.6f}",
            "nll_B":         f"{mb['nll']:.6f}",
            "delta_nll":     f"{mb['nll'] - ma['nll']:+.6f}",
            "rps_A":         f"{ma['rps']:.6f}",
            "rps_B":         f"{mb['rps']:.6f}",
            "delta_rps":     f"{mb['rps'] - ma['rps']:+.6f}",
            "ead_A":         f"{ma['ead']:.6f}",
            "ead_B":         f"{mb['ead']:.6f}",
            "delta_ead":     f"{mb['ead'] - ma['ead']:+.6f}",
            "pred_bin_A":    PDO_CHANGE_BIN_LABELS[ma["pred_bin_idx"]],
            "pred_bin_B":    PDO_CHANGE_BIN_LABELS[mb["pred_bin_idx"]],
            "argmax_match":  "yes" if ma["pred_bin_idx"] == mb["pred_bin_idx"] else "no",
            "correct_A":     "yes" if ca else "no",
            "correct_B":     "yes" if cb else "no",
        })

    write_per_sample_csv(per_sample_rows, out_dir / f"{split}_per_sample.csv")

    # (mean_A, mean_B, lower_is_better)
    summary = {
        "P(true bin)": (
            float(np.mean(metrics_a["p_true"])),
            float(np.mean(metrics_b["p_true"])),
            False,
        ),
        "Brier": (
            float(np.mean(metrics_a["brier"])),
            float(np.mean(metrics_b["brier"])),
            True,
        ),
        "NLL": (
            float(np.mean(metrics_a["nll"])),
            float(np.mean(metrics_b["nll"])),
            True,
        ),
        "RPS": (
            float(np.mean(metrics_a["rps"])),
            float(np.mean(metrics_b["rps"])),
            True,
        ),
        "Expected |bin err|": (
            float(np.mean(metrics_a["ead"])),
            float(np.mean(metrics_b["ead"])),
            True,
        ),
    }
    write_summary_csv(summary, out_dir / f"{split}_summary.csv")

    plot_ptrue_bars(per_sample_rows, split, out_dir)
    plot_prob_overlay(pairs, split, out_dir)
    plot_metric_summary(summary, split, out_dir)

    return {
        "n_samples":     len(pairs),
        "correct_A":     correct_a,
        "correct_B":     correct_b,
        "argmax_match":  argmax_match,
        "summary":       summary,
        "per_sample":    per_sample_rows,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(split_results: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Comparison: `actin_ck` vs `cd8_actin_ck`")
    lines.append("")
    lines.append("Both models share the same backbone (ResNet-18, 12 ordinal PDO-change bins).")
    lines.append("They differ only in the input channels:")
    lines.append("")
    lines.append("- **Model A — `actin_ck`**: 2 channels (actin + CK19)")
    lines.append("- **Model B — `cd8_actin_ck`**: 3 channels (CD8 + actin + CK19)")
    lines.append("")
    lines.append("Argmax accuracy and macro-F1 are identical on the val set, so this analysis")
    lines.append("focuses on the *softmax distributions* using probability-aware scoring rules.")
    lines.append("")
    lines.append("Lower is better for Brier, NLL, RPS, and Expected |bin err|.  ")
    lines.append("Higher is better for P(true bin).")
    lines.append("")

    for split, res in split_results.items():
        lines.append(f"## Split: `{split}`")
        lines.append("")
        lines.append(f"- Samples: **{res['n_samples']}**")
        lines.append(f"- Argmax-correct A: {res['correct_A']} / {res['n_samples']}")
        lines.append(f"- Argmax-correct B: {res['correct_B']} / {res['n_samples']}")
        lines.append(f"- Argmax agreement A vs B: {res['argmax_match']} / {res['n_samples']}")
        lines.append("")
        lines.append("### Aggregate metrics")
        lines.append("")
        lines.append("| Metric | A (actin_ck) | B (cd8_actin_ck) | delta (B - A) | Better for B? |")
        lines.append("|---|---:|---:|---:|:---:|")
        for metric, (va, vb, lib) in res["summary"].items():
            delta = vb - va
            better = (delta < 0) if lib else (delta > 0)
            arrow = "yes" if better else "no"
            lines.append(
                f"| {metric} | {va:.4f} | {vb:.4f} | "
                f"{delta:+.4f} | {arrow} |"
            )
        lines.append("")
        lines.append("### Per-sample P(true bin)")
        lines.append("")
        lines.append("| Sample | True bin | P(true) A | P(true) B | delta | A correct | B correct |")
        lines.append("|---|---|---:|---:|---:|:---:|:---:|")
        for r in res["per_sample"]:
            lines.append(
                "| {sid} | {tl} | {pa} | {pb} | {dp} | {ca} | {cb} |".format(
                    sid=r["image_id"],
                    tl=r["true_bin_label"],
                    pa=r["p_true_A"],
                    pb=r["p_true_B"],
                    dp=r["delta_p_true"],
                    ca=r["correct_A"],
                    cb=r["correct_B"],
                )
            )
        lines.append("")
        lines.append(f"Figures: `{split}_prob_overlay.{{png,svg}}`, "
                     f"`{split}_ptrue_bars.{{png,svg}}`, "
                     f"`{split}_metric_summary.{{png,svg}}`.")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir) if args.output_dir is not None \
        else results_root / "comparison_actin_ck_vs_cd8_actin_ck"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {out_dir}")

    split_results: dict = {}
    for split in args.splits:
        split_results[split] = compare_split(split, results_root, out_dir)

    write_markdown_report(split_results, out_dir / "comparison_report.md")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
