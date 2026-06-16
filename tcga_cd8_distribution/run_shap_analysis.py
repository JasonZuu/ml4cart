import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns 
from torch.utils.data import DataLoader

from .dataset import INPUT_MASK_NAMES, TCGACD8Dataset, load_split_json
from common.cd8_model import UNet
from common.seed import set_random_seed


TARGET_MASKS_DEFAULT = ["cd4", "cd68", "ck", "actin", "pd-1"]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "")


def _display_name(name: str) -> str:
    key = _normalize_name(name)
    mapping = {
        "cd4": "CD4",
        "cd68": "CD68",
        "ck": "CK",
        "actin": "ACTIN",
        "pd1": "PD1",
        "tissue": "TISSUE",
    }
    return mapping.get(key, str(name).upper())


def _resolve_device(device_str: str) -> torch.device:
    if str(device_str).lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if str(device_str).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_checkpoint(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def _detect_in_channels(state_dict: dict, fallback: int) -> int:
    key = "enc1.block.0.weight"
    if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) >= 2:
        return int(state_dict[key].shape[1])
    return int(fallback)


def _pick_mask_names(model_in_channels: int, requested_mask_names: list[str]) -> list[str]:
    if len(requested_mask_names) == model_in_channels:
        return list(requested_mask_names)
    if model_in_channels == len(INPUT_MASK_NAMES):
        return list(INPUT_MASK_NAMES)
    return [f"ch{i}" for i in range(model_in_channels)]


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


def _extract_gradient_shap_values(shap_values, expected_channels: int | None = None) -> np.ndarray:
    vals = shap_values[0] if isinstance(shap_values, (list, tuple)) else shap_values
    if isinstance(vals, list):
        vals = np.asarray(vals)
    if hasattr(vals, "detach"):
        vals = vals.detach().cpu().numpy()
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", type=Path, default=Path("data_f/TCGA_WSI_masks"))
    parser.add_argument("--split-json", type=Path, default=Path("data_f/data_split.json"))
    parser.add_argument("--weights-dir", type=Path, default=Path("wsi_cd8_distribution/results/seed-1_lr-0p0005_bs-16_wd-0p01_base-32"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default="wsi_cd8_distribution/results/seed-1_lr-0p0005_bs-16_wd-0p01_base-32")
    parser.add_argument("--split-name", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--shap-image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask-names", nargs="+", default=["cd4", "cd68", "ck", "actin", "pd-1"])
    parser.add_argument("--target-masks", nargs="+", default=["cd4", "cd68", "ck", "actin", "pd-1"])
    parser.add_argument("--background-size", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=8)
    return parser.parse_args(argv)


def run_shap_analysis(args):
    set_random_seed(int(args.seed))
    torch.backends.cudnn.enabled = torch.cuda.is_available()
    device = _resolve_device(str(args.device))
    model_path = Path(args.model_path) if args.model_path is not None else Path(args.weights_dir) / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")

    state_dict = _read_checkpoint(model_path, device)
    model_in_channels = _detect_in_channels(state_dict, fallback=len(list(args.mask_names)))
    mask_names = _pick_mask_names(model_in_channels, list(args.mask_names))

    split_payload = load_split_json(Path(args.split_json))
    case_ids = split_payload.get(str(args.split_name), []) or []
    if not case_ids:
        raise ValueError(f"Split '{args.split_name}' has no case_ids.")

    dataset = TCGACD8Dataset(
        masks_dir=Path(args.masks_dir),
        case_ids=list(case_ids),
        input_mask_names=mask_names,
        target_mask_name="cd8",
        image_size=int(args.image_size),
        augment=False,
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(in_channels=model_in_channels, out_channels=1, base_channels=int(args.base_channels)).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    max_items = max(int(args.background_size), int(args.eval_size))
    x_all = _collect_inputs(loader, device, max_items=max_items)
    bg_n = min(int(args.background_size), int(x_all.shape[0]))
    ev_n = min(int(args.eval_size), int(x_all.shape[0]))
    background = _resize_for_shap(x_all[:bg_n], target_size=args.shap_image_size)
    eval_x = _resize_for_shap(x_all[:ev_n], target_size=args.shap_image_size)

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
    rows = []
    target_keys = {_normalize_name(x) for x in list(args.target_masks)}
    for idx, name in enumerate(mask_names):
        nkey = _normalize_name(name)
        if nkey in target_keys:
            rows.append(
                {
                    "feature": _display_name(name),
                    "importance": float(channel_importance[idx]),
                }
            )
    if not rows:
        raise ValueError("None of target_masks matched mask_names.")

    df = pd.DataFrame(rows)
    df = df.reindex(df["importance"].abs().sort_values(ascending=False).index)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(args.weights_dir) / "plots" / f"{args.split_name} results"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.barplot(data=df, x="importance", y="feature", palette="viridis")
    plt.title(f"SHAP Importance ({args.split_name})", fontsize=13, pad=10)
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "shap_feature_importance.svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote: {output_dir / 'shap_feature_importance.csv'}")
    print(f"wrote: {output_dir / 'shap_feature_importance.png'}")
    print(f"wrote: {output_dir / 'shap_feature_importance.svg'}")


def main(argv=None) -> int:
    args = parse_args(argv)
    run_shap_analysis(args)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
