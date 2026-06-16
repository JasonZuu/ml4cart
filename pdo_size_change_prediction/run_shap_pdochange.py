"""SHAP channel-importance analysis for on-chip PDO-change classifiers."""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


DISPLAY_NAMES = {
    "cd8": "CD8",
    "cd68": "CD68",
    "actin": "ACTIN",
    "ck": "CK",
}


class PredictedClassLogitWrapper(nn.Module):
    """Return the predicted-class logit for each sample."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        pred = torch.argmax(logits, dim=1)
        pred_logits = logits[torch.arange(len(pred), device=logits.device), pred]
        return pred_logits.unsqueeze(1)


def _collect_dataset_tensor(dataset: OnchipPDOChangeDataset, limit: int | None = None) -> tuple[torch.Tensor, list[str]]:
    images: list[torch.Tensor] = []
    image_ids: list[str] = []
    n = len(dataset) if limit is None else min(len(dataset), int(limit))
    for idx in range(n):
        image, _label, image_id = dataset[idx]
        images.append(image)
        image_ids.append(str(image_id))
    if not images:
        raise ValueError("No samples available for SHAP analysis.")
    return torch.stack(images, dim=0), image_ids


def _build_background_tensor(
    masks_dir: Path,
    train_ids: list[str],
    pdo_change_labels: dict[str, float],
    mask_names: list[str],
    transformations: MaskImageTransformations,
    num_background: int,
) -> tuple[torch.Tensor, int]:
    if not train_ids:
        raise ValueError("No train IDs available for SHAP background.")
    train_dataset = OnchipPDOChangeDataset(
        masks_dir=masks_dir,
        image_ids=list(train_ids),
        pdo_change_labels=pdo_change_labels,
        mask_names=mask_names,
        transform=transformations.validation_transformations,
    )
    n_background = min(len(train_dataset), int(num_background))
    generator = torch.Generator().manual_seed(1)
    indices = torch.randperm(len(train_dataset), generator=generator)[:n_background].tolist()
    images = [train_dataset[idx][0] for idx in indices]
    return torch.stack(images, dim=0), n_background


def _extract_shap_array(shap_values, n_channels: int) -> np.ndarray:
    arr = shap_values[0] if isinstance(shap_values, list) else shap_values
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected SHAP array with 4 dimensions, got shape {arr.shape}")
    if arr.shape[1] != n_channels and arr.shape[-1] == n_channels:
        arr = np.moveaxis(arr, -1, 1)
    if arr.shape[1] != n_channels:
        raise ValueError(f"Expected channel dimension {n_channels}, got SHAP shape {arr.shape}")
    return arr


def _write_outputs(
    channel_importance: np.ndarray,
    mask_names: list[str],
    output_dir: Path,
    n_background: int,
    shap_nsamples: int,
    num_bootstrap: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    display_names = [DISPLAY_NAMES.get(name, name.upper()) for name in mask_names]
    means = channel_importance.mean(axis=0)
    stds = channel_importance.std(axis=0)
    n_eval = int(channel_importance.shape[0])
    rng = np.random.default_rng(1)
    bootstrap_means = np.empty((int(num_bootstrap), len(mask_names)), dtype=np.float64)
    for boot_idx in range(int(num_bootstrap)):
        sample_idx = rng.integers(0, n_eval, size=n_eval)
        bootstrap_means[boot_idx] = channel_importance[sample_idx].mean(axis=0)
    bootstrap_se = bootstrap_means.std(axis=0)
    ci_low = np.percentile(bootstrap_means, 2.5, axis=0)
    ci_high = np.percentile(bootstrap_means, 97.5, axis=0)
    ci_low = np.clip(ci_low, 0.0, None)

    csv_path = output_dir / "shap_feature_importance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "channel",
                "mean_abs_shap",
                "std_abs_shap",
                "bootstrap_se",
                "ci95_low",
                "ci95_high",
            ]
        )
        for channel, mean_val, std_val, se_val, low_val, high_val in zip(
            display_names, means, stds, bootstrap_se, ci_low, ci_high
        ):
            writer.writerow(
                [
                    channel,
                    f"{float(mean_val):.10f}",
                    f"{float(std_val):.10f}",
                    f"{float(se_val):.10f}",
                    f"{float(low_val):.10f}",
                    f"{float(high_val):.10f}",
                ]
            )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    x = np.arange(len(display_names))
    yerr = np.vstack([means - ci_low, ci_high - means])
    yerr = np.clip(yerr, 0.0, None)
    ax.bar(x, means, yerr=yerr, capsize=5, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel("Mean absolute SHAP")
    ax.set_title("PDO-change channel importance\nerror bars: bootstrap 95% CI")
    ax.set_ylim(bottom=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_feature_importance.png", bbox_inches="tight")
    fig.savefig(output_dir / "shap_feature_importance.svg", bbox_inches="tight")
    plt.close(fig)

    summary = {
        channel: {
            "mean": float(mean_val),
            "std": float(std_val),
            "bootstrap_se": float(se_val),
            "ci95_low": float(low_val),
            "ci95_high": float(high_val),
            "n_eval": n_eval,
            "n_background": int(n_background),
            "shap_nsamples": int(shap_nsamples),
            "num_bootstrap": int(num_bootstrap),
        }
        for channel, mean_val, std_val, se_val, low_val, high_val in zip(
            display_names, means, stds, bootstrap_se, ci_low, ci_high
        )
    }
    with (output_dir / "shap_feature_importance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_shap_analysis(
    model_path: Path,
    masks_dir: Path,
    split_json: Path,
    label_json: Path,
    mask_names: list[str],
    split_set: str = "val",
    output_dir: Path | None = None,
    device: str = "cuda",
    num_background: int = 40,
    num_eval: int | None = 8,
    shap_nsamples: int = 1000,
    num_bootstrap: int = 10000,
) -> dict:
    import shap

    set_random_seed(1)
    model_path = Path(model_path)
    masks_dir = Path(masks_dir)
    split_json = Path(split_json)
    label_json = Path(label_json)
    output_dir = Path(output_dir) if output_dir is not None else model_path.parent / "shap_analysis"

    split_payload = load_split_json(split_json)
    eval_ids = split_payload.get(split_set, []) or []
    if not eval_ids:
        raise ValueError(f"No IDs found for split set: {split_set}")
    pdo_change_labels = load_pdo_change_labels(label_json)
    transformations = MaskImageTransformations(image_size=512, normalize_mean=0.5, normalize_std=0.5)

    eval_dataset = OnchipPDOChangeDataset(
        masks_dir=masks_dir,
        image_ids=list(eval_ids),
        pdo_change_labels=pdo_change_labels,
        mask_names=mask_names,
        transform=transformations.validation_transformations,
    )
    eval_tensor, eval_image_ids = _collect_dataset_tensor(eval_dataset, limit=num_eval)

    try:
        background_tensor, n_background = _build_background_tensor(
            masks_dir=masks_dir,
            train_ids=list(split_payload.get("train", []) or []),
            pdo_change_labels=pdo_change_labels,
            mask_names=mask_names,
            transformations=transformations,
            num_background=num_background,
        )
    except Exception as exc:
        print(f"Warning: using eval samples as SHAP background because train background failed: {exc}")
        background_tensor = eval_tensor[: min(len(eval_tensor), int(num_background))]
        n_background = int(background_tensor.shape[0])

    target_device = resolve_device(str(device))
    model = PDOChangeResNetClassifier(
        in_channels=len(mask_names),
        num_classes=len(PDO_CHANGE_BIN_LABELS),
        hidden_dim=256,
        dropout=0.2,
    ).to(target_device)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    background_tensor = background_tensor.to(target_device)
    eval_tensor = eval_tensor.to(target_device)
    wrapped_model = PredictedClassLogitWrapper(model).to(target_device)
    wrapped_model.eval()

    print(
        f"Running SHAP on {len(eval_image_ids)} {split_set} samples "
        f"with {n_background} background samples, nsamples={int(shap_nsamples)}, "
        f"masks={mask_names}"
    )
    explainer = shap.GradientExplainer(wrapped_model, background_tensor)
    shap_values = explainer.shap_values(eval_tensor, nsamples=int(shap_nsamples), rseed=1)
    shap_arr = _extract_shap_array(shap_values, n_channels=len(mask_names))
    channel_importance = np.abs(shap_arr).mean(axis=(2, 3))

    summary = _write_outputs(
        channel_importance=channel_importance,
        mask_names=mask_names,
        output_dir=output_dir,
        n_background=n_background,
        shap_nsamples=int(shap_nsamples),
        num_bootstrap=int(num_bootstrap),
    )
    print(f"Saved SHAP analysis to: {output_dir}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run SHAP channel-importance analysis for PDO-change models.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--label-json", type=Path, required=True)
    parser.add_argument("--mask-names", nargs="+", required=True)
    parser.add_argument("--split-set", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-background", type=int, default=40)
    parser.add_argument("--num-eval", type=int, default=8)
    parser.add_argument("--shap-nsamples", type=int, default=1000)
    parser.add_argument("--num-bootstrap", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_shap_analysis(
        model_path=args.model_path,
        masks_dir=args.masks_dir,
        split_json=args.split_json,
        label_json=args.label_json,
        mask_names=list(args.mask_names),
        split_set=str(args.split_set),
        output_dir=args.output_dir,
        device=str(args.device),
        num_background=int(args.num_background),
        num_eval=int(args.num_eval),
        shap_nsamples=int(args.shap_nsamples),
        num_bootstrap=int(args.num_bootstrap),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
