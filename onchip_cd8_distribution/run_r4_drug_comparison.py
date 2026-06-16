"""Run trained UNet on Chip-R4 patient groups and compare predicted CD8 density.

Patient groups (from data_split.json): test_nyu285, test_nyu318, test_nyu774

Outputs per drug group:
  - Per-sample visualizations (input masks, prediction, ground-truth label)
  - comparison_table.csv  (case_id, patient, drug, day, pred_density, label_density)
  - density_by_drug.png   (box plot grouped by drug)
  - density_by_patient_day.png  (line plot per patient across days, colored by drug)
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.cd8_training import COLOR_MAP, resolve_device
from .dataset import OnchipCD8Dataset, load_split_json
from .run_inference import (
    _build_dataset_with_mask_fallback,
    mask_to_rgb,
    make_combined_rgb,
    save_combined_with_legend,
)
from common.cd8_model import UNet


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run per-drug group inference on Chip-R4 and compare CD8 distributions."
    )
    parser.add_argument(
        "--model-path", type=Path,
        default=Path("onchip_cd8_distribution/results/seed-1_lr-0p0001_bs-8_wd-0p0001_base-32_input-dapi_actin_ck/best_model.pth"),
    )
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data"))
    parser.add_argument(
        "--split-json", type=Path, default=Path("data/On-chip_Data/data_split.json")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("onchip_cd8_distribution/results/r4_drug_comparison"),
    )
    parser.add_argument(
        "--drug-groups",
        type=str,
        default="test_nyu285,test_nyu318,test_nyu774",
        help="Comma-separated split keys to compare (must exist in split JSON)",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--mask-names", nargs="+", default=["dapi", "actin", "ck"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Case ID parsing
# ---------------------------------------------------------------------------

def _parse_case_id_r4(case_id: str) -> tuple[str, str, str]:
    """Extract (patient, drug_original_case, day) from 'chip-r4_nyu285_fap-d1'.

    Returns e.g. ('285', 'FAP', 'd1').
    Raises ValueError if the case_id does not match the R4 pattern.
    """
    # Expected: chip-r4_nyu{patient}_{drug_lower}-{day}
    if "chip-r4_nyu" not in case_id:
        raise ValueError(f"Not an R4 case ID: {case_id}")
    after_nyu = case_id.split("chip-r4_nyu", 1)[1]  # e.g. "285_fap-d1"
    patient, drug_day = after_nyu.split("_", 1)       # "285", "fap-d1"
    drug_lower, day = drug_day.split("-", 1)           # "fap", "d1"
    # Canonicalize drug label for display
    drug_display = {"fap": "FAP", "igg": "IgG", "iareg": "iAREG"}.get(drug_lower, drug_lower.upper())
    return patient, drug_display, day


# ---------------------------------------------------------------------------
# Inference for one drug group
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict_group(
    model: UNet,
    masks_dir: Path,
    case_ids: list[str],
    mask_names: list[str],
    image_size: int,
    base_channels: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    drug_label: str,
    output_drug_dir: Path,
) -> list[dict]:
    """Run inference on one drug group. Returns per-sample records and saves visualizations."""
    state_dict = model.state_dict()
    model_in_channels = int(
        state_dict.get("enc1.block.0.weight", torch.empty(1, len(mask_names))).shape[1]
    )
    dataset, actual_mask_names = _build_dataset_with_mask_fallback(
        masks_dir=masks_dir,
        case_ids=list(case_ids),
        image_size=image_size,
        model_in_channels=model_in_channels,
        requested_mask_names=list(mask_names),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    records: list[dict] = []
    seen_case_counts: dict[str, int] = {}

    for x_norm, y, case_ids_batch in tqdm(loader, desc=f"Inference ({drug_label})"):
        x_norm = x_norm.to(device)
        y = y.to(device)
        probs = torch.sigmoid(model(x_norm))
        x_raw = (x_norm * 0.5 + 0.5).clamp(0.0, 1.0)
        x_np = x_raw.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()

        for i, case_id in enumerate(case_ids_batch):
            k = seen_case_counts.get(case_id, 0)
            seen_case_counts[case_id] = k + 1
            folder_name = case_id if k == 0 else f"{case_id}_{k}"
            sample_dir = output_drug_dir / folder_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Build input mask dict and save visualizations
            input_masks_dict: dict[str, np.ndarray] = {}
            for ch_idx, name in enumerate(actual_mask_names):
                mask_2d = np.clip(x_np[i, ch_idx], 0.0, 1.0)
                input_masks_dict[str(name)] = mask_2d
                color = COLOR_MAP.get(str(name).lower(), (255, 255, 255))
                Image.fromarray(mask_to_rgb(mask_2d, color), mode="RGB").save(
                    sample_dir / f"{name}_mask.png"
                )
            combined = make_combined_rgb(input_masks_dict, list(actual_mask_names))
            save_combined_with_legend(combined, list(actual_mask_names), sample_dir / "combined_input_mask.png")

            label_2d = np.clip(labels_np[i, 0], 0.0, 1.0)
            pred_2d = np.clip(probs_np[i, 0], 0.0, 1.0)
            cd8_color = COLOR_MAP.get("cd8", (255, 77, 37))
            Image.fromarray(mask_to_rgb(label_2d, cd8_color), mode="RGB").save(sample_dir / "label_cd8.png")
            Image.fromarray(mask_to_rgb(pred_2d, cd8_color), mode="RGB").save(sample_dir / "pred_cd8.png")

            # Compute density metrics
            pred_density = float(pred_2d.mean())
            label_density = float(label_2d.mean())

            try:
                patient, drug_display, day = _parse_case_id_r4(case_id)
            except ValueError:
                patient, drug_display, day = "unknown", drug_label, "unknown"

            records.append(
                {
                    "case_id": case_id,
                    "patient": patient,
                    "drug": drug_display,
                    "day": day,
                    "pred_density": pred_density,
                    "label_density": label_density,
                }
            )

    return records


# ---------------------------------------------------------------------------
# Output: CSV table
# ---------------------------------------------------------------------------

def build_comparison_table(all_records: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "comparison_table.csv"
    fieldnames = ["case_id", "patient", "drug", "day", "pred_density", "label_density"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(all_records, key=lambda r: (r["drug"], r["patient"], r["day"])):
            writer.writerow(rec)
    print(f"Saved {csv_path}")


# ---------------------------------------------------------------------------
# Output: plots
# ---------------------------------------------------------------------------

DRUG_COLORS = {"FAP": "#e41a1c", "IgG": "#377eb8", "iAREG": "#4daf4a"}
DRUG_ORDER = ["FAP", "IgG", "iAREG"]


def plot_density_by_drug(all_records: list[dict], output_dir: Path) -> None:
    """Box plot of predicted and label CD8 density grouped by drug."""
    drugs = [d for d in DRUG_ORDER if any(r["drug"] == d for r in all_records)]
    pred_by_drug = {d: [r["pred_density"] for r in all_records if r["drug"] == d] for d in drugs}
    label_by_drug = {d: [r["label_density"] for r in all_records if r["drug"] == d] for d in drugs}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    for ax, data_by_drug, title in zip(
        axes,
        [pred_by_drug, label_by_drug],
        ["Predicted CD8 Density", "Ground-Truth CD8 Density"],
    ):
        values = [data_by_drug[d] for d in drugs]
        bp = ax.boxplot(values, patch_artist=True, labels=drugs)
        for patch, drug in zip(bp["boxes"], drugs):
            patch.set_facecolor(DRUG_COLORS.get(drug, "#888888"))
            patch.set_alpha(0.7)
        ax.set_ylabel("Mean pixel density (0–1)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("CD8 Density by Drug Group (Chip-R4)", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "density_by_drug.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_density_by_patient_day(all_records: list[dict], output_dir: Path) -> None:
    """Line plot per patient: predicted CD8 density across days, one line per drug."""
    patients = sorted({r["patient"] for r in all_records})
    drugs = [d for d in DRUG_ORDER if any(r["drug"] == d for r in all_records)]

    n_patients = len(patients)
    fig, axes = plt.subplots(1, n_patients, figsize=(5 * n_patients, 4), sharey=True)
    if n_patients == 1:
        axes = [axes]

    for ax, patient in zip(axes, patients):
        for drug in drugs:
            recs = sorted(
                [r for r in all_records if r["patient"] == patient and r["drug"] == drug],
                key=lambda r: r["day"],
            )
            if not recs:
                continue
            x = [r["day"] for r in recs]
            y_pred = [r["pred_density"] for r in recs]
            y_label = [r["label_density"] for r in recs]
            color = DRUG_COLORS.get(drug, "#888888")
            ax.plot(x, y_pred, marker="o", color=color, label=drug, linewidth=2)
            ax.plot(x, y_label, marker="s", color=color, linestyle="--", alpha=0.5)

        ax.set_title(f"NYU {patient}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Predicted CD8 density")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Predicted CD8 Density Across Days by Drug (Chip-R4)\n"
        "solid=predicted, dashed=ground-truth",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = output_dir / "density_by_patient_day.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(all_records: list[dict]) -> None:
    drugs = [d for d in DRUG_ORDER if any(r["drug"] == d for r in all_records)]
    print("\n" + "=" * 60)
    print(f"{'Drug':<10} {'N':>4} {'Pred density':>14} {'Label density':>14}")
    print("-" * 60)
    for drug in drugs:
        recs = [r for r in all_records if r["drug"] == drug]
        if not recs:
            continue
        preds = [r["pred_density"] for r in recs]
        labels = [r["label_density"] for r in recs]
        print(
            f"{drug:<10} {len(recs):>4} "
            f"{np.mean(preds):>10.4f}±{np.std(preds):.3f} "
            f"{np.mean(labels):>10.4f}±{np.std(labels):.3f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    mask_names = list(args.mask_names)
    if len(mask_names) == 1 and "," in mask_names[0]:
        mask_names = [s.strip() for s in mask_names[0].split(",") if s.strip()]

    drug_groups = [g.strip() for g in args.drug_groups.split(",") if g.strip()]

    device = resolve_device(args.device)
    state_dict = torch.load(args.model_path, map_location=device)
    model_in_channels = int(
        state_dict.get("enc1.block.0.weight", torch.empty(1, len(mask_names))).shape[1]
    )
    model = UNet(
        in_channels=model_in_channels,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    split_payload = load_split_json(args.split_json)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for group_key in drug_groups:
        case_ids = split_payload.get(group_key, []) or []
        if not case_ids:
            print(f"[warn] No case IDs found for split '{group_key}' in {args.split_json}")
            continue

        # Derive a short display label from the key (e.g. "test_nyu285" -> "nyu285")
        drug_label = group_key.replace("test_", "")
        output_drug_dir = args.output_dir / drug_label

        print(f"\n--- Drug group: {drug_label} ({len(case_ids)} samples) ---")
        records = _predict_group(
            model=model,
            masks_dir=args.masks_dir,
            case_ids=list(case_ids),
            mask_names=mask_names,
            image_size=args.image_size,
            base_channels=args.base_channels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            drug_label=drug_label,
            output_drug_dir=output_drug_dir,
        )
        all_records.extend(records)

    if not all_records:
        print("[error] No records produced. Check model path and split JSON.")
        return 1

    build_comparison_table(all_records, args.output_dir)
    plot_density_by_drug(all_records, args.output_dir)
    plot_density_by_patient_day(all_records, args.output_dir)
    print_summary(all_records)

    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
