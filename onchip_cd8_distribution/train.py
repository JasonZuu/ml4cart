import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.cd8_training import (
    epoch_eval,
    epoch_train,
    evaluate_and_save_val_predictions,
    resolve_device,
)
from common.paths import onchip_data_dir
from common.seed import set_random_seed
from .dataset import OnchipCD8Dataset, load_split_json
from common.cd8_model import UNet


PRESETS = {
    "actin_ck": ["actin", "ck"],
    "dapi_actin_ck": ["dapi", "actin", "ck"],
}
TRAIN_INPUT_MASK_NAMES = PRESETS["actin_ck"]


def _soft_dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = _soft_dice_loss_from_logits(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="actin_ck")
    parser.add_argument("--masks-dir", type=Path, default=onchip_data_dir())
    parser.add_argument("--split-json", type=Path, default=onchip_data_dir() / "data_split.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask-names", nargs="+", default=None)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--bce-weight", type=float, default=0.1)
    parser.add_argument("--dice-weight", type=float, default=0.9)
    return parser.parse_args(argv)


def train_unet(config: dict, wandb_run=None) -> dict:
    masks_dir = Path(config["masks_dir"])
    split_json = Path(config["split_json"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(int(config["seed"]))
    split_payload = load_split_json(split_json)
    train_case_ids = split_payload.get("train", []) or []
    val_case_ids = split_payload.get("val", []) or []
    if not train_case_ids:
        raise ValueError("data_split.json has no train case_ids.")
    if not val_case_ids:
        raise ValueError("data_split.json has no val case_ids.")

    mask_names = list(config["mask_names"])
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    train_dataset = OnchipCD8Dataset(
        masks_dir=masks_dir,
        case_ids=train_case_ids,
        input_mask_names=mask_names,
        target_mask_name="cd8",
        image_size=image_size,
        augment=True,
    )
    val_dataset = OnchipCD8Dataset(
        masks_dir=masks_dir,
        case_ids=val_case_ids,
        input_mask_names=mask_names,
        target_mask_name="cd8",
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = resolve_device(str(config["device"]))
    model = UNet(in_channels=len(mask_names), out_channels=1, base_channels=int(config["base_channels"])).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config.get("bce_weight", 0.5)),
        dice_weight=float(config.get("dice_weight", 0.5)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_val_dice = -1.0
    best_val_rmse = float("inf")
    best_epoch = -1
    best_model_state = None
    history = []
    early_stop = 0
    early_stop_patience = int(config.get("early_stop_patience", 10))

    epochs = int(config["epochs"])
    epoch_iter = tqdm(range(1, epochs + 1), desc="Training UNet")
    for epoch in epoch_iter:
        train_loss, train_dice, train_rmse = epoch_train(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_rmse = epoch_eval(model, val_loader, criterion, device)
        scheduler.step(val_rmse)
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_dice": train_dice,
                "val_dice": val_dice,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "learning_rate": lr_now,
            }
        )

        epoch_iter.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_dice": f"{val_dice:.4f}",
                "val_rmse": f"{val_rmse:.4f}",
            }
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/dice": train_dice,
                    "val/dice": val_dice,
                    "train/rmse": train_rmse,
                    "val/rmse": val_rmse,
                    "train/lr": lr_now,
                },
                step=epoch,
            )

        if (val_dice > best_val_dice) or (
            np.isclose(val_dice, best_val_dice) and val_rmse < best_val_rmse
        ):
            best_val_dice = val_dice
            best_val_rmse = val_rmse
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, output_dir / "best_model.pth")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= early_stop_patience:
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    val_pred_dir = output_dir / "val_predictions"
    val_summary = evaluate_and_save_val_predictions(
        model, val_loader, criterion, device, val_pred_dir, input_mask_names=mask_names
    )
    history_csv_path = output_dir / "metrics.csv"
    history_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_dice",
        "val_dice",
        "train_rmse",
        "val_rmse",
        "learning_rate",
    ]
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_columns)
        writer.writeheader()
        for row in history:
            writer.writerow({col: row.get(col) for col in history_columns})
    metrics = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
        "best_val_rmse": best_val_rmse,
        "val_loss": val_summary["val_loss"],
        "val_dice": val_summary["val_dice"],
        "val_rmse": val_summary["val_rmse"],
        "val_mae": val_summary["val_mae"],
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "input_masks": list(mask_names),
        "preset": str(config.get("preset", "")),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _args_to_config(args) -> dict:
    mask_names = list(args.mask_names) if args.mask_names is not None else list(PRESETS[args.preset])
    if len(mask_names) == 1 and isinstance(mask_names[0], str) and "," in mask_names[0]:
        mask_names = [s.strip() for s in mask_names[0].split(",") if s.strip()]
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("onchip_cd8_distribution/results") / str(args.preset)
    cfg = {
        "masks_dir": str(args.masks_dir),
        "split_json": str(args.split_json),
        "output_dir": str(output_dir),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "image_size": int(args.image_size),
        "base_channels": int(args.base_channels),
        "seed": int(args.seed),
        "device": str(args.device),
        "mask_names": mask_names,
        "early_stop_patience": int(args.early_stop_patience),
        "bce_weight": float(args.bce_weight),
        "dice_weight": float(args.dice_weight),
        "preset": str(args.preset),
    }
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _args_to_config(args)
    metrics = train_unet(cfg, wandb_run=None)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
