"""SegX-GradCAM visualisation for on-chip PDO-change prediction models.

Standard GradCAM is computed w.r.t. the final ResNet conv block (layer4).
The SegX augmentation multiplies the raw CAM by each channel's own binary
mask so that only cells positive in that channel are highlighted.

Scalar target for backward: logit of the predicted class.

Output per sample saved to results/{version}/val_analysis/{sample_id}/:
  - gradcam.png              — multi-panel combined figure
      cd8_only:     1×2  [CD8 mask | SegX-GradCAM CD8]
      cd8_actin_ck: 2×3  row0=[CD8|Actin|CK masks], row1=[SegX-GradCAMs]
      cd8_cd68_actin_ck: 2×4 row0=[CD8|CD68|Actin|CK masks], row1=[SegX-GradCAMs]
  - cd8_mask.png             — raw grayscale CD8 mask
  - segx_gradcam_cd8.png     — SegX heatmap for CD8 channel
  - [actin_mask.png]         — Actin mask (cd8_actin_ck only)
  - [segx_gradcam_actin.png] — SegX heatmap for Actin channel (cd8_actin_ck only)
  - [ck_mask.png]            — CK mask (cd8_actin_ck only)
  - [segx_gradcam_ck.png]    — SegX heatmap for CK channel (cd8_actin_ck only)

gradcam_summary.csv is saved at val_analysis/ (not per-sample).

Usage (run from repo root):
    python onchip_pdochange_prediction/draw_segx_gradcam.py --version cd8_only
    python onchip_pdochange_prediction/draw_segx_gradcam.py --version cd8_actin_ck
    python onchip_pdochange_prediction/draw_segx_gradcam.py --version cd8_cd68_actin_ck
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

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
    "actin_only": ["actin"],
    "actin_ck": ["actin", "ck"],
    "cd8_only": ["cd8"],
    "cd8_actin_ck": ["cd8", "actin", "ck"],
    "cd8_cd68_actin_ck": ["cd8", "cd68", "actin", "ck"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SegX-GradCAM figures for val-set samples"
    )
    parser.add_argument(
        "--version",
        choices=sorted(VERSION_TO_MASKS.keys()),
        default="actin_ck",
        help="Input mask version, including R2 four-channel cd8_cd68_actin_ck.",
    )
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data"))
    parser.add_argument("--split-json", type=Path, default=Path("data/On-chip_Data/data_split.json"))
    parser.add_argument("--label-json", type=Path, default=Path("data/On-chip_Data/pdo_change_label.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("onchip_pdochange_prediction/results"),
                        help="Root results directory; version subdir is appended automatically")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="GradCAM heatmap blend weight (0=mask only, 1=heatmap only)")
    parser.add_argument("--cd8-threshold", type=float, default=0.05,
                        help="Denorm threshold for CD8-positive binary mask")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# GradCAM helpers  (adapted from wsi_survival_analysis/draw_gradcam.py)
# ---------------------------------------------------------------------------

class _GradCAMHooks:
    """Registers forward/backward hooks on a target layer."""

    def __init__(self, layer: torch.nn.Module):
        self._fmaps: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self._fwd_handle = layer.register_forward_hook(self._save_fmaps)
        self._bwd_handle = layer.register_full_backward_hook(self._save_grads)

    def _save_fmaps(self, _module, _inp, output):
        self._fmaps = output

    def _save_grads(self, _module, _grad_in, grad_out):
        self._grads = grad_out[0]

    def compute(self) -> np.ndarray:
        """Return (H, W) GradCAM map in [0, 1], averaged over the batch."""
        assert self._fmaps is not None and self._grads is not None
        weights = self._grads.mean(dim=(2, 3))                          # (B, C)
        cam = (weights[:, :, None, None] * self._fmaps).sum(dim=1)     # (B, H, W)
        cam = F.relu(cam)
        cam_np = cam.mean(dim=0).detach().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def _resize_cam(cam_lowres: np.ndarray, h: int, w: int) -> np.ndarray:
    cam_t = torch.from_numpy(cam_lowres)[None, None].float()
    cam_t = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
    return cam_t[0, 0].numpy()


def _cam_to_rgb(cam: np.ndarray) -> np.ndarray:
    return plt.cm.jet(cam)[:, :, :3]


def _cam_metrics_in_mask(cam_full: np.ndarray, binary_mask: np.ndarray) -> tuple[float, float]:
    """Return (mean CAM inside mask, fraction of mask pixels with CAM > 0.5).

    Uses the global (un-per-channel-normalised) cam_full so that channels are comparable.
    Returns (0.0, 0.0) when the mask is empty.
    """
    n_pos = binary_mask.sum()
    if n_pos == 0:
        return 0.0, 0.0
    cam_in_mask = cam_full * binary_mask
    mean_val = float(cam_in_mask.sum() / n_pos)
    hi_frac = float(((cam_full > 0.5) * binary_mask).sum() / n_pos)
    return mean_val, hi_frac


def _make_segx_overlay(
    gray: np.ndarray,
    cam_full: np.ndarray,
    cd8_binary: np.ndarray,
    alpha: float,
) -> np.ndarray:
    cam_filtered = cam_full * cd8_binary
    cam_max = cam_filtered.max()
    if cam_max > 0:
        cam_filtered = cam_filtered / cam_max
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    cam_rgb = _cam_to_rgb(cam_filtered)
    mask3 = np.stack([cd8_binary, cd8_binary, cd8_binary], axis=-1)
    overlay = np.where(mask3 > 0, (1 - alpha) * gray_rgb + alpha * cam_rgb, gray_rgb)
    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-sample figure saving
# ---------------------------------------------------------------------------

def save_gradcam_figure(rec: dict, mask_names: list[str], out_dir: Path, alpha: float) -> None:
    image_id = rec["image_id"]
    img_np = rec["img_np"]              # (C, H, W) float32 in [0, 1]
    segx_overlays = rec["segx_overlays"]  # list of (H, W, 3) uint8, one per channel

    # Per-sample subfolder
    sample_dir = out_dir / image_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(mask_names)
    channel_titles = [n.upper() for n in mask_names]

    # Save individual files for every channel
    for ch_idx, ch_name in enumerate(mask_names):
        plt.imsave(sample_dir / f"{ch_name}_mask.png", img_np[ch_idx], cmap="gray", vmin=0, vmax=1)
        plt.imsave(sample_dir / f"segx_gradcam_{ch_name}.png", segx_overlays[ch_idx])

    correct_str = "CORRECT" if rec["true_bin_idx"] == rec["pred_bin_idx"] else "WRONG"
    suptitle = (
        f"{image_id}  |  True: {rec['true_bin_label']}  |  "
        f"Pred: {rec['pred_bin_label']} ({rec['pred_confidence']:.1%})  [{correct_str}]"
    )

    sm = ScalarMappable(cmap="jet", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    if n_channels == 1:
        # cd8_only: 1×2 layout [mask | SegX-GradCAM]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
        axes[0].imshow(img_np[0], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"{channel_titles[0]} mask", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(segx_overlays[0])
        axes[1].set_title(f"{channel_titles[0]} cells attended\n(SegX-GradCAM)", fontsize=11)
        axes[1].axis("off")
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Attention intensity", fontsize=9)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["Low", "Mid", "High"])
    else:
        # Multi-channel layout. For R2 this is 2×4:
        #   row 0: masks
        #   row 1: SegX-GradCAMs
        fig, axes = plt.subplots(
            2, n_channels, figsize=(5 * n_channels, 10), dpi=120, constrained_layout=True
        )
        for ch_idx in range(n_channels):
            axes[0, ch_idx].imshow(img_np[ch_idx], cmap="gray", vmin=0, vmax=1)
            axes[0, ch_idx].set_title(f"{channel_titles[ch_idx]} mask", fontsize=11)
            axes[0, ch_idx].axis("off")
            axes[1, ch_idx].imshow(segx_overlays[ch_idx])
            axes[1, ch_idx].set_title(
                f"{channel_titles[ch_idx]} cells attended\n(SegX-GradCAM)", fontsize=11
            )
            axes[1, ch_idx].axis("off")
        # Single colorbar spanning the bottom row
        cbar = fig.colorbar(sm, ax=axes[1, :].tolist(), fraction=0.02, pad=0.04)
        cbar.set_label("Attention intensity", fontsize=9)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["Low", "Mid", "High"])

    fig.suptitle(suptitle, fontsize=10, y=1.01 if n_channels > 1 else 1.0)
    if n_channels == 1:
        fig.tight_layout()
    fig.savefig(sample_dir / "gradcam.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {sample_dir}/gradcam.png  [{correct_str}]")


def save_dual_cam_figure(rec: dict, mask_names: list[str], out_dir: Path, alpha: float) -> None:
    """3×n_channels figure comparing pred-class vs true-class GradCAM for wrong predictions.

    Row 0: input masks
    Row 1: SegX-GradCAM (predicted class)
    Row 2: SegX-GradCAM (true class)
    """
    image_id = rec["image_id"]
    img_np = rec["img_np"]
    segx_pred = rec["segx_overlays"]
    segx_true = rec["segx_overlays_true"]

    sample_dir = out_dir / image_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    n_channels = len(mask_names)
    channel_titles = [n.upper() for n in mask_names]
    sm = ScalarMappable(cmap="jet", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    fig, axes = plt.subplots(
        3, n_channels, figsize=(5 * n_channels, 15), dpi=120, constrained_layout=True
    )
    for ch_idx in range(n_channels):
        axes[0, ch_idx].imshow(img_np[ch_idx], cmap="gray", vmin=0, vmax=1)
        axes[0, ch_idx].set_title(f"{channel_titles[ch_idx]} mask", fontsize=11)
        axes[0, ch_idx].axis("off")
        axes[1, ch_idx].imshow(segx_pred[ch_idx])
        axes[1, ch_idx].set_title(
            f"{channel_titles[ch_idx]}\npred: {rec['pred_bin_label']}", fontsize=10
        )
        axes[1, ch_idx].axis("off")
        axes[2, ch_idx].imshow(segx_true[ch_idx])
        axes[2, ch_idx].set_title(
            f"{channel_titles[ch_idx]}\ntrue: {rec['true_bin_label']}", fontsize=10
        )
        axes[2, ch_idx].axis("off")
    fig.colorbar(sm, ax=axes[1:, :].ravel().tolist(), fraction=0.015, pad=0.04,
                 label="Attention intensity")
    fig.suptitle(
        f"{image_id}  |  True: {rec['true_bin_label']}  |  "
        f"Pred: {rec['pred_bin_label']} ({rec['pred_confidence']:.1%})  [WRONG]\n"
        "Row 1 = pred-class CAM  •  Row 2 = true-class CAM",
        fontsize=10,
    )
    out_path = sample_dir / "gradcam_pred_vs_true.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}  [dual-CAM]")


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

    alpha = float(args.alpha)
    cd8_threshold = float(args.cd8_threshold)

    # Per-sample GradCAM inference (requires grad; batch_size=1)
    records: list[dict] = []
    for idx in range(len(val_dataset)):
        image_tensor, true_label, image_id = val_dataset[idx]
        inp = image_tensor.unsqueeze(0).to(device).detach().requires_grad_(True)

        hooks = _GradCAMHooks(model.backbone.layer4)
        pred_logits = model(inp)                                        # (1, 12)
        pred_class = int(pred_logits.argmax(dim=1).item())
        pred_logits[0, pred_class].backward()
        model.zero_grad()

        cam_lowres = hooks.compute()
        hooks.remove()

        with torch.no_grad():
            probs = torch.softmax(pred_logits.detach(), dim=1)[0].cpu().numpy()

        # Denormalize: NormalizeByChannels uses (x - 0.5) / 0.5 → inverse is x * 0.5 + 0.5
        img_np = np.clip(inp[0].detach().cpu().numpy() * 0.5 + 0.5, 0.0, 1.0)  # (C, H, W)

        h, w = img_np.shape[1], img_np.shape[2]
        cam_full = _resize_cam(cam_lowres, h, w)

        # Binary masks and per-channel overlays (pred-class CAM)
        binary_masks = [
            (img_np[ch_idx] > cd8_threshold).astype(np.float32)
            for ch_idx in range(len(mask_names))
        ]
        segx_overlays = [
            _make_segx_overlay(img_np[ch_idx], cam_full, binary_masks[ch_idx], alpha)
            for ch_idx in range(len(mask_names))
        ]

        # Exp B: per-channel CAM intensity metrics (using global cam_full for comparability)
        cam_metrics: dict[str, float] = {}
        for ch_idx, ch_name in enumerate(mask_names):
            mean_val, hi_frac = _cam_metrics_in_mask(cam_full, binary_masks[ch_idx])
            cam_metrics[f"{ch_name}_cam_mean"] = mean_val
            cam_metrics[f"{ch_name}_cam_hi_frac"] = hi_frac

        true_bin_idx = int(true_label.item())

        # Exp C: true-class CAM for mispredicted samples
        segx_overlays_true: list | None = None
        if pred_class != true_bin_idx:
            hooks_true = _GradCAMHooks(model.backbone.layer4)
            pred_logits_true = model(inp)                               # fresh forward pass
            pred_logits_true[0, true_bin_idx].backward()
            model.zero_grad()
            cam_true_lowres = hooks_true.compute()
            hooks_true.remove()
            cam_true_full = _resize_cam(cam_true_lowres, h, w)
            segx_overlays_true = [
                _make_segx_overlay(img_np[ch_idx], cam_true_full, binary_masks[ch_idx], alpha)
                for ch_idx in range(len(mask_names))
            ]

        records.append({
            "image_id": str(image_id),
            "true_bin_idx": true_bin_idx,
            "true_bin_label": PDO_CHANGE_BIN_LABELS[true_bin_idx],
            "pred_bin_idx": pred_class,
            "pred_bin_label": PDO_CHANGE_BIN_LABELS[pred_class],
            "pred_confidence": float(probs[pred_class]),
            "img_np": img_np,
            "segx_overlays": segx_overlays,
            "segx_overlays_true": segx_overlays_true,
            **cam_metrics,
        })

    # Save per-sample figures
    print()
    for rec in records:
        save_gradcam_figure(rec, mask_names, out_dir, alpha)
        if rec["segx_overlays_true"] is not None:
            save_dual_cam_figure(rec, mask_names, out_dir, alpha)

    # Summary CSV — base columns + per-channel CAM intensity metrics (Exp B)
    cam_metric_fields = [
        f"{ch}_{metric}"
        for ch in mask_names
        for metric in ("cam_mean", "cam_hi_frac")
    ]
    csv_path = out_dir / "gradcam_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "true_bin_idx", "true_bin_label",
            "pred_bin_idx", "pred_bin_label", "pred_confidence", "correct",
            *cam_metric_fields,
        ])
        writer.writeheader()
        for rec in records:
            row = {
                "image_id": rec["image_id"],
                "true_bin_idx": rec["true_bin_idx"],
                "true_bin_label": rec["true_bin_label"],
                "pred_bin_idx": rec["pred_bin_idx"],
                "pred_bin_label": rec["pred_bin_label"],
                "pred_confidence": f"{rec['pred_confidence']:.6f}",
                "correct": rec["true_bin_idx"] == rec["pred_bin_idx"],
            }
            for field in cam_metric_fields:
                row[field] = f"{rec[field]:.6f}"
            writer.writerow(row)
    print(f"\nSaved gradcam_summary.csv")
    print(f"All outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
