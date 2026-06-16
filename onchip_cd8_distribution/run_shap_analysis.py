import argparse
import json
import os
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.cd8_training import resolve_device
from common.seed import set_random_seed
from .dataset import INPUT_MASK_NAMES, OnchipCD8Dataset, load_split_json
from common.cd8_model import UNet


TARGET_MASKS_DEFAULT = ["dapi", "actin", "ck"]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "")


def _display_mask_name_for_shap(name: str) -> str:
    key = _normalize_name(name)
    mapping = {
        "dapi": "DAPI",
        "actin": "ACTIN",
        "ck": "CK",
    }
    return mapping.get(key, str(name).upper())


def _pick_mask_names(model_in_channels: int, requested_mask_names: list[str]) -> list[str]:
    if len(requested_mask_names) == model_in_channels:
        return list(requested_mask_names)
    presets = {
        3: ["dapi", "actin", "ck"],
        2: ["actin", "ck"],
        1: ["actin"],
    }
    if model_in_channels in presets:
        return list(presets[model_in_channels])
    if model_in_channels == len(INPUT_MASK_NAMES):
        return list(INPUT_MASK_NAMES)
    return [f"ch{i}" for i in range(model_in_channels)]


def _candidate_mask_name_sets(model_in_channels: int, requested_mask_names: list[str]) -> list[list[str]]:
    candidates = []
    primary = _pick_mask_names(model_in_channels, requested_mask_names)
    candidates.append(list(primary))
    presets = {
        3: ["dapi", "actin", "ck"],
        2: ["actin", "ck"],
        1: ["actin"],
    }
    if model_in_channels in presets:
        candidates.append(list(presets[model_in_channels]))
    if model_in_channels == len(INPUT_MASK_NAMES):
        candidates.append(list(INPUT_MASK_NAMES))
    unique = []
    seen = set()
    for names in candidates:
        key = tuple(_normalize_name(x) for x in names)
        if key in seen:
            continue
        seen.add(key)
        unique.append(names)
    return unique


def _build_dataset_with_mask_fallback(
    masks_dir: Path,
    case_ids: list[str],
    image_size: int,
    model_in_channels: int,
    requested_mask_names: list[str],
) -> tuple[OnchipCD8Dataset, list[str]]:
    last_err = None
    for names in _candidate_mask_name_sets(model_in_channels, requested_mask_names):
        try:
            ds = OnchipCD8Dataset(
                masks_dir=masks_dir,
                case_ids=list(case_ids),
                input_mask_names=list(names),
                target_mask_name="cd8",
                image_size=image_size,
                augment=False,
                normalize=True,
            )
            return ds, list(names)
        except ValueError as e:
            last_err = e
    if last_err is not None:
        raise ValueError(
            f"{last_err} model_in_channels={model_in_channels}, requested_mask_names={list(requested_mask_names)}, "
            f"tried={_candidate_mask_name_sets(model_in_channels, requested_mask_names)}"
        ) from last_err
    raise RuntimeError("Failed to build dataset.")


def _collect_inputs(loader: DataLoader, device: torch.device, max_items: int) -> torch.Tensor:
    xs = []
    count = 0
    for x, _y, _case_ids in loader:
        if count >= max_items:
            break
        take = min(int(x.shape[0]), max_items - count)
        xs.append(x[:take])
        count += take
    if not xs:
        raise ValueError("No samples found for SHAP analysis.")
    return torch.cat(xs, dim=0).to(device)


def _resize_for_shap(x: torch.Tensor, target_size: int | None) -> torch.Tensor:
    if target_size is None:
        return x
    target = int(target_size)
    h, w = int(x.shape[2]), int(x.shape[3])
    if h == target and w == target:
        return x
    if target <= 0:
        return x
    mode = "area" if target < min(h, w) else "bilinear"
    if mode == "area":
        return F.interpolate(x, size=(target, target), mode=mode)
    return F.interpolate(x, size=(target, target), mode=mode, align_corners=False)


def _extract_gradient_shap_values(shap_values, expected_channels: int | None = None) -> np.ndarray:
    vals = shap_values[0] if isinstance(shap_values, (list, tuple)) else shap_values
    if isinstance(vals, list):
        vals = np.asarray(vals)
    if torch.is_tensor(vals):
        tensor_vals = cast(torch.Tensor, vals)
        vals = tensor_vals.cpu().numpy()
    vals = np.asarray(vals)
    if vals.ndim == 5 and vals.shape[-1] == 1:
        vals = vals[..., 0]
    elif vals.ndim == 5 and vals.shape[1] == 1:
        vals = vals[:, 0, ...]
    if vals.ndim != 4:
        raise ValueError(f"Unexpected Gradient SHAP output shape: {vals.shape}")
    if expected_channels is not None:
        if vals.shape[1] == int(expected_channels):
            return vals
        if vals.shape[-1] == int(expected_channels):
            return np.transpose(vals, (0, 3, 1, 2))
        raise ValueError(f"Cannot resolve SHAP channel axis from shape {vals.shape}, expected_channels={expected_channels}")
    return vals


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run SHAP analysis for on-chip CD8 segmentation.")
    parser.add_argument("--model-path", type=Path, default=Path("onchip_cd8_distribution/results/lr-0p0001_bs-8_wd-0p0001_base-64_input-dapi_actin_ck/best_model.pth"))
    parser.add_argument("--masks-dir", type=Path, default=Path("data/On-chip_Data"))
    parser.add_argument("--split-json", type=Path, default=Path("data/On-chip_Data/data_split.json"))
    parser.add_argument("--split-set", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--mask-names", nargs="+", default=list(INPUT_MASK_NAMES))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shap-image-size", type=int, default=256)
    parser.add_argument("--shap-background-size", type=int, default=8)
    parser.add_argument("--shap-eval-size", type=int, default=8)
    parser.add_argument("--shap-target-masks", nargs="+", default=list(TARGET_MASKS_DEFAULT))
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def run_shap_analysis(
    model_path: Path,
    masks_dir: Path,
    split_json: Path,
    split_set: str,
    output_dir: Path,
    image_size: int,
    base_channels: int,
    mask_names: list[str],
    batch_size: int,
    num_workers: int,
    device_str: str,
    shap_image_size: int,
    background_size: int,
    eval_size: int,
    target_masks: list[str],
    seed: int,
) -> dict:
    matplotlib.use("Agg")
    set_random_seed(int(seed))
    torch.backends.cudnn.enabled = torch.cuda.is_available()
    device = resolve_device(device_str)
    state_dict = torch.load(model_path, map_location=device)
    model_in_channels = int(state_dict.get("enc1.block.0.weight", torch.empty(1, len(mask_names))).shape[1])
    split_payload = load_split_json(split_json)
    case_ids = split_payload.get(split_set, []) or []
    if not case_ids:
        raise ValueError(f"No case IDs found for split '{split_set}' in {split_json}")
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
    model = UNet(in_channels=model_in_channels, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    max_items = max(int(background_size), int(eval_size))
    x_all = _collect_inputs(loader, device, max_items=max_items)
    bg_n = min(int(background_size), int(x_all.shape[0]))
    ev_n = min(int(eval_size), int(x_all.shape[0]))
    background = _resize_for_shap(x_all[:bg_n], target_size=shap_image_size)
    eval_x = _resize_for_shap(x_all[:ev_n], target_size=shap_image_size)

    class WrappedModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            y = self.base_model(x)
            return torch.sigmoid(y).mean(dim=(1, 2, 3)).unsqueeze(1)

    wrapped = WrappedModel(model).to(device)
    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(eval_x)
    shap_vals = _extract_gradient_shap_values(shap_values, expected_channels=model_in_channels)
    channel_importance = np.mean(np.abs(shap_vals), axis=(0, 2, 3))
    target_keys = {_normalize_name(x) for x in list(target_masks)}
    rows_all = []
    rows = []
    for idx, name in enumerate(actual_mask_names):
        row = {"feature": _display_mask_name_for_shap(name), "importance": float(channel_importance[idx])}
        rows_all.append(row)
        if _normalize_name(name) in target_keys:
            rows.append(row)
    if not rows:
        raise ValueError("None of target_masks matched mask_names.")
    df = pd.DataFrame(rows)
    total_importance = float(df["importance"].sum())
    if total_importance > 0:
        df["contribution_pct"] = (df["importance"] / total_importance) * 100.0
    else:
        df["contribution_pct"] = 0.0
    sorted_idx = np.argsort(-np.abs(df["importance"].to_numpy()))
    df = df.iloc[sorted_idx].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "shap_feature_importance.csv"
    png_path = output_dir / "shap_feature_importance.png"
    svg_path = output_dir / "shap_feature_importance.svg"
    summary_path = output_dir / "shap_feature_importance_summary.json"
    df.to_csv(csv_path, index=False)
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.barplot(data=df, x="importance", y="feature", palette="viridis")
    plt.title(f"SHAP Importance ({split_set})", fontsize=13, pad=10)
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, dpi=300, bbox_inches="tight")
    plt.close()
    summary_payload = {
        "model_in_channels": int(model_in_channels),
        "actual_mask_names": list(actual_mask_names),
        "target_masks": list(target_masks),
        "all_features": rows_all,
        "selected_features": [
            {
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
                "contribution_pct": float(row["contribution_pct"]),
            }
            for row in df.to_dict(orient="records")
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    return {
        "csv": str(csv_path),
        "png": str(png_path),
        "svg": str(svg_path),
        "summary_json": str(summary_path),
        "n_features": int(len(df)),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    mask_names = list(args.mask_names)
    if len(mask_names) == 1 and "," in mask_names[0]:
        mask_names = [s.strip() for s in mask_names[0].split(",") if s.strip()]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.model_path.parent / f"{args.split_set}_shap"
    result = run_shap_analysis(
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
        shap_image_size=int(args.shap_image_size),
        background_size=int(args.shap_background_size),
        eval_size=int(args.shap_eval_size),
        target_masks=list(args.shap_target_masks),
        seed=int(args.seed),
    )
    print(result)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
