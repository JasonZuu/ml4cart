"""Softmax probability distribution plot for val-set samples.

Samples are grouped into three PDO-change categories and shown in separate
subplots so intra-group variation and inter-group differences are both visible:

  Group 0 — Shrinkage   (PDO < −20%, bins 0–4)
  Group 1 — Near-zero   (−20 ≤ PDO < 20%, bins 5–6)
  Group 2 — Growth      (PDO ≥ 20%, bins 7–11)

Usage (run from repo root):
    python onchip_pdochange_prediction/draw_prob_distribution.py --version cd8_only
    python onchip_pdochange_prediction/draw_prob_distribution.py --version cd8_actin_ck
    python onchip_pdochange_prediction/draw_prob_distribution.py --version cd8_cd68_actin_ck
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.seed import set_random_seed
from common.pdochange_data import (
    MaskImageTransformations,
    OnchipPDOChangeDataset,
    PDO_CHANGE_BIN_LABELS,
    load_pdo_change_labels,
    load_split_json,
)
from common.pdochange_model import PDOChangeResNetClassifier
from common.pdochange_training import resolve_device


VERSION_TO_MASKS = {
    "cd8_only": ["cd8"],
    "cd8_actin_ck": ["cd8", "actin", "ck"],
    "cd8_cd68_actin_ck": ["cd8", "cd68", "actin", "ck"],
    "actin_ck": ["actin", "ck"],
}

# Three semantic groups spanning the 12 bins.
# Each entry: (display name, first_bin_inclusive, last_bin_inclusive, colormap name)
GROUPS = [
    ("Shrinkage  (PDO < −20%)",        0,  4, "Reds"),
    ("Near-zero  (−20 ≤ PDO < 20%)",   5,  6, "Blues"),
    ("Growth     (PDO ≥ 20%)",          7, 11, "Greens"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Grouped softmax probability distribution plot for val set"
    )
    parser.add_argument("--version", choices=sorted(VERSION_TO_MASKS.keys()), default="actin_ck")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data"))
    parser.add_argument("--split-json", type=Path, default=Path("data/On-chip_Data/data_split.json"))
    parser.add_argument("--label-json", type=Path, default=Path("data/On-chip_Data/pdo_change_label.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("onchip_pdochange_prediction/results"),
                        help="Root results directory; version subdir is appended automatically")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_prob_csv(records: list[dict], out_dir: Path) -> None:
    """Save per-sample softmax probabilities to CSV.

    Columns: image_id, true_bin_idx, true_bin_label, pred_bin_idx, pred_bin_label,
             followed by one column per bin named prob_{bin_label}.
    This is sufficient to regenerate the probability distribution figure
    without re-running the model.
    """
    prob_cols = [f"prob_{lbl}" for lbl in PDO_CHANGE_BIN_LABELS]
    fieldnames = ["image_id", "true_bin_idx", "true_bin_label",
                  "pred_bin_idx", "pred_bin_label"] + prob_cols
    out_path = out_dir / "prob_distribution.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {
                "image_id": rec["image_id"],
                "true_bin_idx": rec["true_bin_idx"],
                "true_bin_label": PDO_CHANGE_BIN_LABELS[rec["true_bin_idx"]],
                "pred_bin_idx": rec["pred_bin_idx"],
                "pred_bin_label": PDO_CHANGE_BIN_LABELS[rec["pred_bin_idx"]],
            }
            for col, p in zip(prob_cols, rec["probs"]):
                row[col] = f"{p:.6f}"
            writer.writerow(row)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Grouped probability distribution plot
# ---------------------------------------------------------------------------

def plot_grouped_prob_distribution(records: list[dict], out_dir: Path, version: str) -> None:
    """Three-panel figure grouping val samples by PDO-change category."""
    x_positions = np.arange(len(PDO_CHANGE_BIN_LABELS))
    non_empty_groups = [
        (group_name, bin_lo, bin_hi, cmap_name)
        for group_name, bin_lo, bin_hi, cmap_name in GROUPS
        if any(bin_lo <= r["true_bin_idx"] <= bin_hi for r in records)
    ]
    if not non_empty_groups:
        raise ValueError("No non-empty PDO-change groups found for probability plot.")

    fig, axes = plt.subplots(1, len(non_empty_groups), figsize=(6 * len(non_empty_groups), 5), sharey=True)
    if len(non_empty_groups) == 1:
        axes = [axes]
    fig.suptitle(
        f"Softmax Probability Distributions — Validation Set  [{version}]",
        fontsize=13, y=1.01,
    )

    for ax_idx, (group_name, bin_lo, bin_hi, cmap_name) in enumerate(non_empty_groups):
        ax = axes[ax_idx]
        group_records = [r for r in records if bin_lo <= r["true_bin_idx"] <= bin_hi]

        # Shade the "expected" bin range for this group
        ax.axvspan(bin_lo - 0.5, bin_hi + 0.5, color="lightgrey", alpha=0.35, zorder=0,
                   label="True bin range")

        if group_records:
            # Spread colours evenly across the group's colormap (avoid very pale extremes)
            n = len(group_records)
            cmap = plt.cm.get_cmap(cmap_name)
            color_vals = np.linspace(0.45, 0.85, n) if n > 1 else [0.65]

            for i, rec in enumerate(group_records):
                color = cmap(color_vals[i])
                short_id = rec["image_id"].replace("chip-r3_", "")  # shorten label
                ax.plot(
                    x_positions,
                    rec["probs"],
                    marker="o",
                    linewidth=1.8,
                    markersize=5,
                    color=color,
                    alpha=0.9,
                    label=short_id,
                )
                # Mark true class with a filled star
                true_bin = rec["true_bin_idx"]
                ax.plot(true_bin, rec["probs"][true_bin],
                        marker="*", markersize=12, color=color, zorder=5)

        ax.set_title(group_name, fontsize=10, pad=6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(PDO_CHANGE_BIN_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("PDO Change Bin", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Softmax Probability", fontsize=9)
        n_samples = len(group_records)
        ax.legend(loc="upper right", fontsize=7,
                  title=f"{n_samples} sample{'s' if n_samples != 1 else ''}",
                  title_fontsize=7)

    fig.tight_layout()
    out_path = out_dir / "prob_distribution.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    set_random_seed(int(args.seed))

    version = str(args.version)
    mask_names = VERSION_TO_MASKS[version]
    if args.model_path is not None:
        checkpoint = Path(args.model_path)
        model_dir = checkpoint.parent
        out_dir = Path(args.output_dir) if args.output_dir is not None else model_dir / "val_analysis"
    else:
        model_dir = Path(args.output_dir) / version
        out_dir = model_dir / "val_analysis"
        checkpoint = model_dir / "best_model.pth"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    split_payload = load_split_json(args.split_json)
    val_ids = split_payload.get("val", []) or []
    if not val_ids:
        raise ValueError("No val IDs in split JSON.")
    pdo_change_labels = load_pdo_change_labels(args.label_json)
    transformations = MaskImageTransformations(
        image_size=int(args.image_size), normalize_mean=0.5, normalize_std=0.5
    )
    val_dataset = OnchipPDOChangeDataset(
        masks_dir=args.masks_dir,
        image_ids=list(val_ids),
        pdo_change_labels=pdo_change_labels,
        mask_names=mask_names,
        transform=transformations.validation_transformations,
    )
    print(f"Loaded {len(val_dataset)} val samples for version={version}")

    # Load model
    device = resolve_device(str(args.device))
    model = PDOChangeResNetClassifier(
        in_channels=len(mask_names),
        num_classes=len(PDO_CHANGE_BIN_LABELS),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint}")

    # Forward pass (no gradients needed)
    records: list[dict] = []
    with torch.no_grad():
        for idx in range(len(val_dataset)):
            image_tensor, true_label, image_id = val_dataset[idx]
            inp = image_tensor.unsqueeze(0).to(device)
            pred_logits = model(inp)
            probs = torch.softmax(pred_logits, dim=1)[0].cpu().numpy()
            pred_class = int(pred_logits.argmax(dim=1).item())
            true_bin_idx = int(true_label.item())
            records.append({
                "image_id": str(image_id),
                "true_bin_idx": true_bin_idx,
                "pred_bin_idx": pred_class,
                "probs": probs,
            })

    save_prob_csv(records, out_dir)
    plot_grouped_prob_distribution(records, out_dir, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
