"""SegX-GradCAM CD8 attention versus local CAF/actin context.

This analysis asks a model-facing version of the local CAF-CD8 question:
when the PDO-change model predicts a PDO-size-change bin, is its CD8
SegX-GradCAM attention concentrated in CD8 regions with little nearby actin
(CAF proxy), or in CD8 regions with more nearby actin?

The analysis recomputes predicted-class Grad-CAM for the R2 optimal model on
the validation set, multiplies the CAM by the CD8 mask for SegX-CD8 attention,
and summarizes attention by local actin context.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr, wilcoxon

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
from onchip_distribution_analysis.analyze_caf_cd8_local_combined import (
    _ck_stratified_delta,
    _partial_spearman_given_ck,
    _residualize_on_covariate,
)


MASK_NAMES = ["cd8", "cd68", "actin", "ck"]

TILE_FIELDS = [
    "image_id",
    "split",
    "true_bin_idx",
    "true_bin_label",
    "pred_bin_idx",
    "pred_bin_label",
    "pred_confidence",
    "correct",
    "tile_size",
    "tile_row",
    "tile_col",
    "y0",
    "x0",
    "actin_frac",
    "cd8_frac",
    "ck_frac",
    "cd8_pixels",
    "cd8_cam_mean",
    "cd8_attention_mass",
    "caf_group",
]

SAMPLE_FIELDS = [
    "image_id",
    "split",
    "true_bin_idx",
    "true_bin_label",
    "pred_bin_idx",
    "pred_bin_label",
    "pred_confidence",
    "correct",
    "tile_size",
    "n_cd8_tiles",
    "actin_q25",
    "actin_q75",
    "mean_actin_frac_cd8_tiles",
    "attention_weighted_actin_frac",
    "attention_weighted_minus_unweighted_actin",
    "spearman_actin_vs_cd8_cam",
    "spearman_actin_vs_cd8_cam_p",
    "partial_spearman_actin_vs_cd8_cam_given_ck",
    "low_caf_cd8_cam_mean",
    "high_caf_cd8_cam_mean",
    "delta_high_minus_low_cd8_cam",
    "residual_low_caf_cd8_cam_mean",
    "residual_high_caf_cd8_cam_mean",
    "residual_delta_high_minus_low_cd8_cam_given_ck",
    "ck_stratified_delta_high_minus_low_cd8_cam",
    "n_ck_strata",
    "n_ck_stratified_tiles",
    "low_caf_attention_mass_frac",
    "high_caf_attention_mass_frac",
    "mean_cd8_distance_to_actin",
    "attention_weighted_cd8_distance_to_actin",
    "attention_weighted_minus_unweighted_distance",
]

DISTANCE_FIELDS = [
    "image_id",
    "split",
    "true_bin_idx",
    "true_bin_label",
    "pred_bin_idx",
    "pred_bin_label",
    "pred_confidence",
    "correct",
    "band_index",
    "band_label",
    "distance_lo",
    "distance_hi",
    "cd8_pixel_count",
    "cd8_cam_mean",
    "cd8_attention_mass",
    "cd8_attention_mass_frac",
]

COHORT_FIELDS = [
    "metric",
    "alternative",
    "n_samples",
    "mean",
    "median",
    "wilcoxon_statistic",
    "wilcoxon_p",
    "interpretation",
]


class GradCAMHooks:
    def __init__(self, layer: torch.nn.Module):
        self.fmaps: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        self.fwd_handle = layer.register_forward_hook(self._save_fmaps)
        self.bwd_handle = layer.register_full_backward_hook(self._save_grads)

    def _save_fmaps(self, _module, _inp, output):
        self.fmaps = output

    def _save_grads(self, _module, _grad_in, grad_out):
        self.grads = grad_out[0]

    def compute(self) -> np.ndarray:
        if self.fmaps is None or self.grads is None:
            raise RuntimeError("GradCAM hooks did not capture feature maps/gradients.")
        weights = self.grads.mean(dim=(2, 3))
        cam = (weights[:, :, None, None] * self.fmaps).sum(dim=1)
        cam = F.relu(cam)
        cam_np = cam.mean(dim=0).detach().cpu().numpy()
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze SegX-CD8 attention by local actin/CAF context.")
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data_R2"))
    parser.add_argument("--split-json", type=Path, default=Path("data/On-chip_Data_R2/data_split.json"))
    parser.add_argument("--label-json", type=Path, default=Path("data/On-chip_Data_R2/pdo_change_label.json"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("onchip_pdochange_prediction/results/r2/optimal/best_model.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onchip_pdochange_prediction/results/r2/optimal/caf_cd8_local_analysis/segxgradcam_cd8_caf"),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=["train", "val", "test", "all"])
    parser.add_argument("--mask-names", nargs="+", default=MASK_NAMES)
    parser.add_argument("--dataset-label", type=str, default="R2 validation")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--distance-bands", nargs="+", type=int, default=[0, 8, 16, 32, 64, 128])
    parser.add_argument("--mask-threshold", type=float, default=0.05)
    parser.add_argument("--min-cd8-pixels", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def resize_cam(cam_lowres: np.ndarray, h: int, w: int) -> np.ndarray:
    cam_t = torch.from_numpy(cam_lowres)[None, None].float()
    cam_t = F.interpolate(cam_t, size=(h, w), mode="bilinear", align_corners=False)
    return cam_t[0, 0].numpy()


def cam_to_rgb(cam: np.ndarray) -> np.ndarray:
    return plt.cm.jet(np.clip(cam, 0.0, 1.0))[:, :, :3]


def safe_spearman(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    rho, p_val = spearmanr(x, y)
    return float(rho), float(p_val)


def safe_wilcoxon(values: list[float], alternative: str) -> tuple[float, float]:
    vals = np.array(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if np.allclose(vals, 0.0):
        return 0.0, 1.0
    stat, p_val = wilcoxon(vals, alternative=alternative)
    return float(stat), float(p_val)


def format_value(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
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


def compute_pred_cam(model, image_tensor: torch.Tensor, true_bin_idx: int, device) -> dict:
    inp = image_tensor.unsqueeze(0).to(device).detach().requires_grad_(True)
    hooks = GradCAMHooks(model.backbone.layer4)
    logits = model(inp)
    pred_bin_idx = int(logits.argmax(dim=1).item())
    score = logits[0, pred_bin_idx]
    model.zero_grad(set_to_none=True)
    score.backward()
    cam_lowres = hooks.compute()
    hooks.remove()
    probs = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
    img_np = np.clip(inp[0].detach().cpu().numpy() * 0.5 + 0.5, 0.0, 1.0)
    h, w = img_np.shape[1], img_np.shape[2]
    return {
        "img_np": img_np,
        "cam_full": resize_cam(cam_lowres, h, w),
        "pred_bin_idx": pred_bin_idx,
        "pred_bin_label": PDO_CHANGE_BIN_LABELS[pred_bin_idx],
        "pred_confidence": float(probs[pred_bin_idx]),
        "true_bin_idx": int(true_bin_idx),
        "true_bin_label": PDO_CHANGE_BIN_LABELS[int(true_bin_idx)],
        "correct": pred_bin_idx == int(true_bin_idx),
    }


def compute_tile_rows(base: dict, img_np: np.ndarray, cam_full: np.ndarray, tile_size: int, threshold: float, min_cd8_pixels: int) -> tuple[list[dict], dict]:
    cd8 = img_np[MASK_NAMES.index("cd8")] > threshold
    actin = img_np[MASK_NAMES.index("actin")] > threshold
    ck = img_np[MASK_NAMES.index("ck")] > threshold
    segx_cd8 = cam_full * cd8.astype(np.float32)
    h, w = cd8.shape
    n_rows = h // int(tile_size)
    n_cols = w // int(tile_size)
    rows: list[dict] = []
    for tile_row in range(n_rows):
        y0 = tile_row * int(tile_size)
        y1 = y0 + int(tile_size)
        for tile_col in range(n_cols):
            x0 = tile_col * int(tile_size)
            x1 = x0 + int(tile_size)
            cd8_tile = cd8[y0:y1, x0:x1]
            cd8_pixels = int(cd8_tile.sum())
            if cd8_pixels < int(min_cd8_pixels):
                continue
            actin_tile = actin[y0:y1, x0:x1]
            ck_tile = ck[y0:y1, x0:x1]
            cam_tile = cam_full[y0:y1, x0:x1]
            segx_tile = segx_cd8[y0:y1, x0:x1]
            rows.append({
                **base,
                "tile_size": int(tile_size),
                "tile_row": int(tile_row),
                "tile_col": int(tile_col),
                "y0": int(y0),
                "x0": int(x0),
                "actin_frac": float(actin_tile.mean()),
                "cd8_frac": float(cd8_tile.mean()),
                "ck_frac": float(ck_tile.mean()),
                "cd8_pixels": cd8_pixels,
                "cd8_cam_mean": float(cam_tile[cd8_tile].mean()),
                "cd8_attention_mass": float(segx_tile.sum()),
                "caf_group": "middle",
            })
    return rows, {"cd8": cd8, "actin": actin, "ck": ck, "segx_cd8": segx_cd8}


def summarize_sample(base: dict, rows: list[dict], masks: dict, distance_bands: list[int]) -> tuple[dict, list[dict]]:
    if not rows:
        empty = {
            **base,
            "tile_size": "",
            "n_cd8_tiles": 0,
            "actin_q25": float("nan"),
            "actin_q75": float("nan"),
            "mean_actin_frac_cd8_tiles": float("nan"),
            "attention_weighted_actin_frac": float("nan"),
            "attention_weighted_minus_unweighted_actin": float("nan"),
            "spearman_actin_vs_cd8_cam": float("nan"),
            "spearman_actin_vs_cd8_cam_p": float("nan"),
            "partial_spearman_actin_vs_cd8_cam_given_ck": float("nan"),
            "low_caf_cd8_cam_mean": float("nan"),
            "high_caf_cd8_cam_mean": float("nan"),
            "delta_high_minus_low_cd8_cam": float("nan"),
            "residual_low_caf_cd8_cam_mean": float("nan"),
            "residual_high_caf_cd8_cam_mean": float("nan"),
            "residual_delta_high_minus_low_cd8_cam_given_ck": float("nan"),
            "ck_stratified_delta_high_minus_low_cd8_cam": float("nan"),
            "n_ck_strata": 0,
            "n_ck_stratified_tiles": 0,
            "low_caf_attention_mass_frac": float("nan"),
            "high_caf_attention_mass_frac": float("nan"),
            "mean_cd8_distance_to_actin": float("nan"),
            "attention_weighted_cd8_distance_to_actin": float("nan"),
            "attention_weighted_minus_unweighted_distance": float("nan"),
        }
        return empty, []

    actin_frac = np.array([r["actin_frac"] for r in rows], dtype=np.float64)
    cd8_cam = np.array([r["cd8_cam_mean"] for r in rows], dtype=np.float64)
    ck_frac = np.array([r["ck_frac"] for r in rows], dtype=np.float64)
    masses = np.array([r["cd8_attention_mass"] for r in rows], dtype=np.float64)
    q25, q75 = np.quantile(actin_frac, [0.25, 0.75])
    for row in rows:
        if float(row["actin_frac"]) <= q25:
            row["caf_group"] = "low"
        elif float(row["actin_frac"]) >= q75:
            row["caf_group"] = "high"
    low = [r for r in rows if r["caf_group"] == "low"]
    high = [r for r in rows if r["caf_group"] == "high"]
    total_mass = float(masses.sum())
    rho, p_val = safe_spearman(actin_frac, cd8_cam)
    partial_rho = _partial_spearman_given_ck(actin_frac, cd8_cam, ck_frac)
    cd8_cam_resid = _residualize_on_covariate(cd8_cam, ck_frac)
    low_resid = cd8_cam_resid[actin_frac <= q25]
    high_resid = cd8_cam_resid[actin_frac >= q75]
    low_resid = low_resid[np.isfinite(low_resid)]
    high_resid = high_resid[np.isfinite(high_resid)]
    low_resid_mean = float(low_resid.mean()) if low_resid.size else float("nan")
    high_resid_mean = float(high_resid.mean()) if high_resid.size else float("nan")
    residual_delta = (
        float(high_resid_mean - low_resid_mean)
        if math.isfinite(high_resid_mean) and math.isfinite(low_resid_mean)
        else float("nan")
    )
    stratified_delta, n_ck_strata, n_ck_stratified_tiles = _ck_stratified_delta(actin_frac, cd8_cam, ck_frac)
    unweighted_actin = float(actin_frac.mean())
    weighted_actin = float(np.average(actin_frac, weights=masses)) if total_mass > 0 else float("nan")

    cd8 = masks["cd8"]
    actin = masks["actin"]
    segx_cd8 = masks["segx_cd8"]
    dist = distance_transform_edt(~actin)
    cd8_dist = dist[cd8]
    cd8_attention = segx_cd8[cd8]
    mean_dist = float(cd8_dist.mean()) if cd8_dist.size else float("nan")
    weighted_dist = float(np.average(cd8_dist, weights=cd8_attention)) if cd8_dist.size and cd8_attention.sum() > 0 else float("nan")

    distance_rows = compute_distance_rows(base, cd8_dist, cd8_attention, distance_bands)
    tile_size = int(rows[0]["tile_size"])
    summary = {
        **base,
        "tile_size": tile_size,
        "n_cd8_tiles": int(len(rows)),
        "actin_q25": float(q25),
        "actin_q75": float(q75),
        "mean_actin_frac_cd8_tiles": unweighted_actin,
        "attention_weighted_actin_frac": weighted_actin,
        "attention_weighted_minus_unweighted_actin": float(weighted_actin - unweighted_actin) if math.isfinite(weighted_actin) else float("nan"),
        "spearman_actin_vs_cd8_cam": rho,
        "spearman_actin_vs_cd8_cam_p": p_val,
        "partial_spearman_actin_vs_cd8_cam_given_ck": partial_rho,
        "low_caf_cd8_cam_mean": float(np.mean([r["cd8_cam_mean"] for r in low])) if low else float("nan"),
        "high_caf_cd8_cam_mean": float(np.mean([r["cd8_cam_mean"] for r in high])) if high else float("nan"),
        "delta_high_minus_low_cd8_cam": float(np.mean([r["cd8_cam_mean"] for r in high]) - np.mean([r["cd8_cam_mean"] for r in low])) if low and high else float("nan"),
        "residual_low_caf_cd8_cam_mean": low_resid_mean,
        "residual_high_caf_cd8_cam_mean": high_resid_mean,
        "residual_delta_high_minus_low_cd8_cam_given_ck": residual_delta,
        "ck_stratified_delta_high_minus_low_cd8_cam": stratified_delta,
        "n_ck_strata": n_ck_strata,
        "n_ck_stratified_tiles": n_ck_stratified_tiles,
        "low_caf_attention_mass_frac": float(sum(r["cd8_attention_mass"] for r in low) / total_mass) if total_mass > 0 else float("nan"),
        "high_caf_attention_mass_frac": float(sum(r["cd8_attention_mass"] for r in high) / total_mass) if total_mass > 0 else float("nan"),
        "mean_cd8_distance_to_actin": mean_dist,
        "attention_weighted_cd8_distance_to_actin": weighted_dist,
        "attention_weighted_minus_unweighted_distance": float(weighted_dist - mean_dist) if math.isfinite(weighted_dist) and math.isfinite(mean_dist) else float("nan"),
    }
    return summary, distance_rows


def compute_distance_rows(base: dict, cd8_dist: np.ndarray, cd8_attention: np.ndarray, bands: list[int]) -> list[dict]:
    rows: list[dict] = []
    if cd8_dist.size == 0:
        return rows
    edges = sorted({int(x) for x in bands})
    if not edges or edges[0] != 0:
        edges = [0, *edges]
    total_mass = float(cd8_attention.sum())
    for idx, lo in enumerate(edges):
        hi = edges[idx + 1] if idx + 1 < len(edges) else math.inf
        if math.isinf(hi):
            sel = cd8_dist >= lo
            label = f">={lo}"
            hi_value = ""
        else:
            sel = (cd8_dist >= lo) & (cd8_dist < hi)
            label = f"{lo}-{hi}"
            hi_value = int(hi)
        pixel_count = int(sel.sum())
        mass = float(cd8_attention[sel].sum()) if pixel_count else 0.0
        rows.append({
            **base,
            "band_index": int(idx),
            "band_label": label,
            "distance_lo": int(lo),
            "distance_hi": hi_value,
            "cd8_pixel_count": pixel_count,
            "cd8_cam_mean": float(cd8_attention[sel].mean()) if pixel_count else float("nan"),
            "cd8_attention_mass": mass,
            "cd8_attention_mass_frac": float(mass / total_mass) if total_mass > 0 else float("nan"),
        })
    return rows


def cohort_row(metric: str, values: list[float], alternative: str, interpretation: str) -> dict:
    vals = np.array(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    stat, p_val = safe_wilcoxon(vals.tolist(), alternative=alternative)
    return {
        "metric": metric,
        "alternative": alternative,
        "n_samples": int(vals.size),
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "median": float(np.median(vals)) if vals.size else float("nan"),
        "wilcoxon_statistic": stat,
        "wilcoxon_p": p_val,
        "interpretation": interpretation,
    }


def build_cohort_stats(sample_rows: list[dict]) -> list[dict]:
    return [
        cohort_row(
            "delta_high_minus_low_cd8_cam",
            [r["delta_high_minus_low_cd8_cam"] for r in sample_rows],
            "less",
            "negative means predicted-class CD8 attention is higher in low-CAF tiles",
        ),
        cohort_row(
            "attention_weighted_minus_unweighted_actin",
            [r["attention_weighted_minus_unweighted_actin"] for r in sample_rows],
            "less",
            "negative means model-attended CD8 tiles have less actin than typical CD8 tiles",
        ),
        cohort_row(
            "spearman_actin_vs_cd8_cam",
            [r["spearman_actin_vs_cd8_cam"] for r in sample_rows],
            "less",
            "negative means CD8 attention decreases as local actin fraction increases",
        ),
        cohort_row(
            "partial_spearman_actin_vs_cd8_cam_given_ck",
            [r["partial_spearman_actin_vs_cd8_cam_given_ck"] for r in sample_rows],
            "less",
            "negative means CD8 attention decreases as local actin fraction increases after controlling CK",
        ),
        cohort_row(
            "residual_delta_high_minus_low_cd8_cam_given_ck",
            [r["residual_delta_high_minus_low_cd8_cam_given_ck"] for r in sample_rows],
            "less",
            "negative means CK-adjusted CD8 attention is higher in low-CAF tiles",
        ),
        cohort_row(
            "ck_stratified_delta_high_minus_low_cd8_cam",
            [r["ck_stratified_delta_high_minus_low_cd8_cam"] for r in sample_rows],
            "less",
            "negative means CD8 attention is higher in low-CAF tiles within matched CK strata",
        ),
        cohort_row(
            "attention_weighted_minus_unweighted_distance",
            [r["attention_weighted_minus_unweighted_distance"] for r in sample_rows],
            "greater",
            "positive means model-attended CD8 pixels are farther from actin than typical CD8 pixels",
        ),
    ]


def save_sample_figure(out_dir: Path, image_id: str, img_np: np.ndarray, cam_full: np.ndarray, masks: dict, rows: list[dict], summary: dict) -> None:
    sample_dir = out_dir / "sample_figures"
    sample_dir.mkdir(parents=True, exist_ok=True)
    actin = img_np[MASK_NAMES.index("actin")]
    cd8 = img_np[MASK_NAMES.index("cd8")]
    cd8_binary = masks["cd8"]
    segx_cd8 = masks["segx_cd8"]
    segx_display = segx_cd8 / segx_cd8.max() if segx_cd8.max() > 0 else segx_cd8

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=130)
    axes[0].imshow(actin, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Actin / CAF proxy")
    axes[1].imshow(cd8, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("CD8 / CAR-T proxy")
    axes[2].imshow(cam_full, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Pred-class Grad-CAM")
    axes[3].imshow(cd8_binary, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(cam_to_rgb(segx_display), alpha=np.where(cd8_binary, 0.85, 0.0))
    axes[3].set_title("CD8 SegX-CAM by CAF tiles")
    for ax in axes:
        ax.axis("off")

    for row in rows:
        if row["caf_group"] not in {"low", "high"}:
            continue
        edge = "#2ca02c" if row["caf_group"] == "low" else "#d627b0"
        rect = patches.Rectangle(
            (float(row["x0"]), float(row["y0"])),
            float(row["tile_size"]),
            float(row["tile_size"]),
            linewidth=0.8,
            edgecolor=edge,
            facecolor="none",
            alpha=0.9,
        )
        axes[3].add_patch(rect)

    title = (
        f"{image_id} | true {summary['true_bin_label']} | pred {summary['pred_bin_label']} "
        f"({float(summary['pred_confidence']):.1%}) | "
        f"high-low CD8 CAM delta={float(summary['delta_high_minus_low_cd8_cam']):.3f}"
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(sample_dir / f"{image_id}_segx_cd8_caf_context.png", bbox_inches="tight")
    plt.close(fig)


def plot_low_high(sample_rows: list[dict], out_dir: Path) -> None:
    labels = [str(r["image_id"]).replace("chip-r1_", "") for r in sample_rows]
    low = np.array([r["low_caf_cd8_cam_mean"] for r in sample_rows], dtype=np.float64)
    high = np.array([r["high_caf_cd8_cam_mean"] for r in sample_rows], dtype=np.float64)
    x = np.arange(len(sample_rows))
    fig, ax = plt.subplots(figsize=(max(7, len(sample_rows) * 0.75), 4), dpi=140)
    for i, (lo, hi) in enumerate(zip(low, high)):
        ax.plot([i - 0.12, i + 0.12], [lo, hi], color="0.6", linewidth=1.0)
    ax.scatter(x - 0.12, low, label="Low-CAF CD8 tiles", color="#2ca02c")
    ax.scatter(x + 0.12, high, label="High-CAF CD8 tiles", color="#d627b0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean predicted-class CAM on CD8-positive pixels")
    ax.set_title("CD8 SegX-GradCAM attention in low- vs high-CAF tiles")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cd8_segx_attention_low_vs_high_caf.png", bbox_inches="tight")
    plt.close(fig)


def plot_weighted_context(sample_rows: list[dict], out_dir: Path) -> None:
    labels = [str(r["image_id"]).replace("chip-r1_", "") for r in sample_rows]
    unweighted = np.array([r["mean_actin_frac_cd8_tiles"] for r in sample_rows], dtype=np.float64)
    weighted = np.array([r["attention_weighted_actin_frac"] for r in sample_rows], dtype=np.float64)
    x = np.arange(len(sample_rows))
    fig, ax = plt.subplots(figsize=(max(7, len(sample_rows) * 0.75), 4), dpi=140)
    for i, (u, w) in enumerate(zip(unweighted, weighted)):
        ax.plot([i - 0.12, i + 0.12], [u, w], color="0.6", linewidth=1.0)
    ax.scatter(x - 0.12, unweighted, label="All CD8 tiles", color="0.25")
    ax.scatter(x + 0.12, weighted, label="CAM-weighted CD8 tiles", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Local actin fraction")
    ax.set_title("Actin context of CD8 tiles: unweighted vs model-attended")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_weighted_caf_vs_unweighted.png", bbox_inches="tight")
    plt.close(fig)


def plot_distance(distance_rows: list[dict], out_dir: Path) -> None:
    grouped: dict[int, list[dict]] = {}
    for row in distance_rows:
        grouped.setdefault(int(row["band_index"]), []).append(row)
    labels = []
    means = []
    sems = []
    for idx in sorted(grouped):
        rows = grouped[idx]
        vals = np.array([r["cd8_cam_mean"] for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        labels.append(rows[0]["band_label"])
        means.append(float(vals.mean()) if vals.size else float("nan"))
        sems.append(float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.errorbar(np.arange(len(labels)), means, yerr=sems, marker="o", capsize=3)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("CD8 pixel distance from actin-positive pixels (512px input)")
    ax.set_ylabel("Mean predicted-class CAM on CD8 pixels")
    ax.set_title("CD8 SegX-GradCAM attention by distance from CAF/actin")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cd8_segx_attention_by_distance.png", bbox_inches="tight")
    plt.close(fig)


def write_report(out_dir: Path, sample_rows: list[dict], cohort_rows: list[dict], plots_enabled: bool = True) -> None:
    by_metric = {r["metric"]: r for r in cohort_rows}
    delta = by_metric["delta_high_minus_low_cd8_cam"]
    weighted_actin = by_metric["attention_weighted_minus_unweighted_actin"]
    weighted_dist = by_metric["attention_weighted_minus_unweighted_distance"]
    partial_ck = by_metric["partial_spearman_actin_vs_cd8_cam_given_ck"]
    residual_ck = by_metric["residual_delta_high_minus_low_cd8_cam_given_ck"]
    stratified_ck = by_metric["ck_stratified_delta_high_minus_low_cd8_cam"]
    lines = [
        "# SegX-GradCAM CD8 Attention vs CAF Context",
        "",
        "## Question",
        "When the PDO-change model predicts PDO size change, does it attend more to CD8 signal in low-CAF/actin neighborhoods, or in high-CAF/actin neighborhoods?",
        "",
        "## Main Readout",
        f"- Samples: {len(sample_rows)} samples",
        "- CAM target: model predicted class",
        "- CD8 attention: predicted-class Grad-CAM multiplied by the CD8 mask",
        "- Local CAF context: actin fraction in 64px tiles containing CD8",
        "",
        "## Results",
        f"- High-minus-low CAF CD8-CAM delta median: {float(delta['median']):.4f}, p(delta < 0) = {float(delta['wilcoxon_p']):.4g}",
        f"- CAM-weighted minus unweighted actin context median: {float(weighted_actin['median']):.4f}, p(< 0) = {float(weighted_actin['wilcoxon_p']):.4g}",
        f"- CAM-weighted minus unweighted CD8 distance-to-actin median: {float(weighted_dist['median']):.4f}, p(> 0) = {float(weighted_dist['wilcoxon_p']):.4g}",
        "",
        "## Tumor-Adjusted Readout",
        f"- Partial rho(actin, CD8-CAM | CK) median: {float(partial_ck['median']):.4f}, p(rho < 0) = {float(partial_ck['wilcoxon_p']):.4g}",
        f"- CK-adjusted CD8-CAM residual high-minus-low CAF delta median: {float(residual_ck['median']):.4f}, p(delta < 0) = {float(residual_ck['wilcoxon_p']):.4g}",
        f"- CK-stratified high-minus-low CAF CD8-CAM delta median: {float(stratified_ck['median']):.4f}, p(delta < 0) = {float(stratified_ck['wilcoxon_p']):.4g}",
        "",
    ]
    if float(delta["median"]) < 0 and float(delta["wilcoxon_p"]) < 0.05:
        lines.append("Interpretation: model CD8 attention is significantly higher in low-CAF CD8 tiles.")
    elif float(delta["median"]) > 0 and float(delta["wilcoxon_p"]) > 0.95:
        lines.append("Interpretation: model CD8 attention trends higher in high-CAF CD8 tiles, not low-CAF tiles.")
    else:
        lines.append("Interpretation: model CD8 attention does not show a clear low-CAF preference.")
    lines.extend([
        "",
        "## Files",
        "- `segx_tile_attention.csv`",
        "- `segx_sample_stats.csv`",
        "- `segx_distance_attention.csv`",
        "- `segx_cohort_stats.csv` / `segx_cohort_stats.json`",
    ])
    if plots_enabled:
        lines.extend([
            "- `cd8_segx_attention_low_vs_high_caf.png`",
            "- `attention_weighted_caf_vs_unweighted.png`",
            "- `cd8_segx_attention_by_distance.png`",
            "- `sample_figures/*_segx_cd8_caf_context.png`",
        ])
    (out_dir / "segx_cd8_caf_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_splits(names: list[str]) -> list[str]:
    if "all" in names:
        return ["train", "val", "test"]
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    global MASK_NAMES
    MASK_NAMES = list(args.mask_names)
    set_random_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_payload = load_split_json(args.split_json)
    image_ids: list[str] = []
    image_id_to_split: dict[str, str] = {}
    for split_name in selected_splits(args.splits):
        for image_id in split_payload.get(split_name, []) or []:
            image_id = str(image_id)
            if image_id in image_id_to_split:
                continue
            image_id_to_split[image_id] = split_name
            image_ids.append(image_id)
    if not image_ids:
        raise ValueError("No image IDs found for requested splits.")
    if args.limit is not None:
        image_ids = image_ids[: int(args.limit)]
    labels = load_pdo_change_labels(args.label_json)
    transformations = MaskImageTransformations(
        image_size=int(args.image_size), normalize_mean=0.5, normalize_std=0.5
    )
    dataset = OnchipPDOChangeDataset(
        masks_dir=args.masks_dir,
        image_ids=image_ids,
        pdo_change_labels=labels,
        mask_names=MASK_NAMES,
        transform=transformations.validation_transformations,
    )

    device = resolve_device(str(args.device))
    model = PDOChangeResNetClassifier(
        in_channels=len(MASK_NAMES),
        num_classes=len(PDO_CHANGE_BIN_LABELS),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    tile_rows_all: list[dict] = []
    sample_rows: list[dict] = []
    distance_rows_all: list[dict] = []
    for idx in range(len(dataset)):
        image_tensor, true_label, image_id = dataset[idx]
        cam_rec = compute_pred_cam(model, image_tensor, int(true_label.item()), device)
        base = {
            "image_id": str(image_id),
            "split": image_id_to_split.get(str(image_id), ""),
            "true_bin_idx": cam_rec["true_bin_idx"],
            "true_bin_label": cam_rec["true_bin_label"],
            "pred_bin_idx": cam_rec["pred_bin_idx"],
            "pred_bin_label": cam_rec["pred_bin_label"],
            "pred_confidence": cam_rec["pred_confidence"],
            "correct": cam_rec["correct"],
        }
        tile_rows, masks = compute_tile_rows(
            base,
            cam_rec["img_np"],
            cam_rec["cam_full"],
            int(args.tile_size),
            float(args.mask_threshold),
            int(args.min_cd8_pixels),
        )
        summary, distance_rows = summarize_sample(base, tile_rows, masks, args.distance_bands)
        tile_rows_all.extend(tile_rows)
        sample_rows.append(summary)
        distance_rows_all.extend(distance_rows)
        if not args.no_plots:
            save_sample_figure(out_dir, str(image_id), cam_rec["img_np"], cam_rec["cam_full"], masks, tile_rows, summary)
        print(f"Processed {idx + 1}/{len(dataset)}: {image_id}", flush=True)

    cohort_rows = build_cohort_stats(sample_rows)
    write_csv(out_dir / "segx_tile_attention.csv", tile_rows_all, TILE_FIELDS)
    write_csv(out_dir / "segx_sample_stats.csv", sample_rows, SAMPLE_FIELDS)
    write_csv(out_dir / "segx_distance_attention.csv", distance_rows_all, DISTANCE_FIELDS)
    write_csv(out_dir / "segx_cohort_stats.csv", cohort_rows, COHORT_FIELDS)
    with (out_dir / "segx_cohort_stats.json").open("w", encoding="utf-8") as f:
        json.dump([{k: format_value(v) for k, v in row.items()} for row in cohort_rows], f, indent=2)

    if not args.no_plots:
        plot_low_high(sample_rows, out_dir)
        plot_weighted_context(sample_rows, out_dir)
        plot_distance(distance_rows_all, out_dir)
    write_report(out_dir, sample_rows, cohort_rows, plots_enabled=not args.no_plots)

    metric_by_name = {r["metric"]: r for r in cohort_rows}
    delta = metric_by_name["delta_high_minus_low_cd8_cam"]
    print()
    print(f"Analyzed {len(sample_rows)} samples. Outputs in: {out_dir}")
    print(
        "CD8 SegX-CAM high-minus-low CAF delta: "
        f"median={float(delta['median']):.4f}, p(delta < 0)={float(delta['wilcoxon_p']):.4g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
