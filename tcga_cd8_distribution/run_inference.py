import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.cd8_training import resolve_device
from .dataset import INPUT_MASK_NAMES, TCGACD8Dataset, load_split_json
from common.cd8_model import UNet


COLOR_MAP = {
    "cd8":    (255,  77,  37),   # #FF4D25
    "cd4":    ( 22,  80, 170),   # #1650AA
    "cd68":   (240, 230,  96),   # #F0E660
    "ck":     (231, 219, 145),   # #E7DB91
    "actin":  (153, 214, 224),   # #99D6E0
    "pd-1":   (119,  32,  29),   # #77201D
    "tissue": (255, 255, 255),   # white
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run inference and save colorized mask visualizations.")
    parser.add_argument("--model-path", type=Path, default="wsi_cd8_distribution/results/seed-1_lr-0p0005_bs-16_wd-0p01_base-32/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--masks-dir", type=Path, default=Path("data_f/TCGA_WSI_masks"))
    parser.add_argument("--split-json", type=Path, default=Path("data_f/data_split.json"))
    parser.add_argument("--split-set", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: model_path.parent / {split_set}_predictions)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--mask-names", nargs="+", default=list(INPUT_MASK_NAMES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def mask_to_rgb(mask: np.ndarray, color: tuple) -> np.ndarray:
    """Convert a 2D float mask [0,1] to an RGB image using the given color."""
    r, g, b = color
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (mask * r).astype(np.uint8)
    rgb[..., 1] = (mask * g).astype(np.uint8)
    rgb[..., 2] = (mask * b).astype(np.uint8)
    return rgb


def make_combined_rgb(input_masks: dict, mask_names: list) -> np.ndarray:
    """Create a combined overlay: tissue as white background, then cell-type masks on top."""
    first = next(iter(input_masks.values()))
    h, w = first.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    # Tissue as white background first
    if "tissue" in input_masks:
        alpha = input_masks["tissue"][..., np.newaxis]
        tissue_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        canvas = canvas * (1.0 - alpha) + tissue_color * alpha

    # Cell-type masks on top in order
    for name in mask_names:
        if name == "tissue" or name not in input_masks:
            continue
        color = np.array(COLOR_MAP.get(name, (255, 255, 255)), dtype=np.float32) / 255.0
        alpha = input_masks[name][..., np.newaxis]
        canvas = canvas * (1.0 - alpha) + color * alpha

    return (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)


def _legend_order(mask_names: list[str]) -> list[str]:
    names = list(mask_names)
    ordered = []
    if "tissue" in names:
        ordered.append("tissue")
    for name in names:
        if name != "tissue":
            ordered.append(name)
    return ordered


def _display_name(name: str) -> str:
    return str(name).replace("_", " ").upper()


def save_combined_with_legend(combined_rgb: np.ndarray, mask_names: list[str], out_path: Path) -> None:
    h, w = int(combined_rgb.shape[0]), int(combined_rgb.shape[1])
    fig_w = 10.0
    fig_h = max(4.0, fig_w * (h / max(float(w), 1.0)))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(combined_rgb)
    ax.set_axis_off()

    handles = []
    for name in _legend_order(mask_names):
        color = np.array(COLOR_MAP.get(name, (255, 255, 255)), dtype=np.float32) / 255.0
        handles.append(Patch(facecolor=tuple(color.tolist()), edgecolor="black", label=_display_name(name)))
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_inference(
    model_path: Path,
    masks_dir: Path,
    split_json: Path,
    split_set: str,
    output_dir: Path,
    image_size: int,
    base_channels: int,
    mask_names: list,
    batch_size: int,
    num_workers: int,
    device_str: str,
) -> None:
    device = resolve_device(device_str)

    split_payload = load_split_json(split_json)
    case_ids = split_payload.get(split_set, []) or []
    if not case_ids:
        raise ValueError(f"No case IDs found for split '{split_set}' in {split_json}")

    dataset = TCGACD8Dataset(
        masks_dir=masks_dir,
        case_ids=case_ids,
        input_mask_names=mask_names,
        target_mask_name="cd8",
        image_size=image_size,
        augment=False,
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=len(mask_names), out_channels=1, base_channels=base_channels).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_case_counts: dict = {}

    for x_norm, y, case_ids_batch in tqdm(loader, desc=f"Inference ({split_set})"):
        x_norm = x_norm.to(device)
        y = y.to(device)
        logits = model(x_norm)
        probs = torch.sigmoid(logits)

        # Denormalize inputs: (x - 0.5) / 0.5 was applied, so reverse: x_raw = x_norm * 0.5 + 0.5
        x_raw = (x_norm * 0.5 + 0.5).clamp(0.0, 1.0)

        x_np = x_raw.detach().cpu().numpy()        # (B, C, H, W)
        y_np = y.detach().cpu().numpy()             # (B, 1, H, W)
        probs_np = probs.detach().cpu().numpy()     # (B, 1, H, W)

        for i, case_id in enumerate(case_ids_batch):
            k = seen_case_counts.get(case_id, 0)
            seen_case_counts[case_id] = k + 1
            folder_name = case_id if k == 0 else f"{case_id}_{k}"
            sample_dir = output_dir / folder_name
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save each input mask as colorized RGB
            input_masks_dict: dict = {}
            for ch_idx, name in enumerate(mask_names):
                mask_2d = np.clip(x_np[i, ch_idx], 0.0, 1.0)
                input_masks_dict[name] = mask_2d
                color = COLOR_MAP.get(name, (255, 255, 255))
                rgb = mask_to_rgb(mask_2d, color)
                safe_name = name.replace("-", "-")  # keep as-is (pd-1 stays pd-1)
                Image.fromarray(rgb, mode="RGB").save(sample_dir / f"{safe_name}_mask.png")

            # Save combined mask
            combined = make_combined_rgb(input_masks_dict, mask_names)
            save_combined_with_legend(combined, mask_names, sample_dir / "combined_mask.png")

            # Save label and prediction as CD8-colored RGB
            cd8_color = COLOR_MAP["cd8"]
            label_2d = np.clip(y_np[i, 0], 0.0, 1.0)
            pred_2d = np.clip(probs_np[i, 0], 0.0, 1.0)
            Image.fromarray(mask_to_rgb(label_2d, cd8_color), mode="RGB").save(sample_dir / "label_cd8.png")
            Image.fromarray(mask_to_rgb(pred_2d, cd8_color), mode="RGB").save(sample_dir / "pred_cd8.png")

    print(f"Saved visualizations for {len(seen_case_counts)} samples to {output_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)

    mask_names = list(args.mask_names)
    if len(mask_names) == 1 and "," in mask_names[0]:
        mask_names = [s.strip() for s in mask_names[0].split(",") if s.strip()]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.model_path.parent / f"{args.split_set}_predictions"

    run_inference(
        model_path=args.model_path,
        masks_dir=args.masks_dir,
        split_json=args.split_json,
        split_set=args.split_set,
        output_dir=output_dir,
        image_size=args.image_size,
        base_channels=args.base_channels,
        mask_names=mask_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
