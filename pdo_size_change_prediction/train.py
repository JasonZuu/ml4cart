import argparse
import copy
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.paths import onchip_data_dir
from common.seed import set_random_seed
from common.pdochange_data import (
    MaskImageTransformations,
    OnchipPDOChangeDataset,
    PDO_CHANGE_BIN_LABELS,
    load_pdo_change_labels,
    load_split_json,
)
from common.pdochange_model import PDOChangeResNetClassifier
from common.pdochange_training import epoch_eval, epoch_train, resolve_device


VERSION_TO_MASKS = {
    "actin_only": ["actin"],
    "actin_ck": ["actin", "ck"],
    "cd8_only": ["cd8"],
    "cd8_actin_ck": ["cd8", "actin", "ck"],
    "cd8_cd68_actin_ck": ["cd8", "cd68", "actin", "ck"],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSION_TO_MASKS.keys()), default="cd8_only")
    parser.add_argument("--masks-dir", type=Path, default=onchip_data_dir())
    parser.add_argument("--split-json", type=Path, default=onchip_data_dir() / "data_split.json")
    parser.add_argument("--label-json", type=Path, default=onchip_data_dir() / "pdo_change_label.json")
    parser.add_argument("--output-dir", type=Path, default=Path("pdo_size_change_prediction/results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    return parser.parse_args(argv)


def train_pdochange(config: dict, wandb_run=None) -> dict:
    masks_dir = Path(config["masks_dir"])
    split_json = Path(config["split_json"])
    label_json = Path(config["label_json"])
    output_root = Path(config["output_dir"])
    version = str(config["version"])
    mask_names = list(config["mask_names"])
    output_dir = output_root / version
    if bool(config.get("output_is_run_dir", False)):
        output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(int(config["seed"]))

    split_payload = load_split_json(split_json)
    train_ids = split_payload.get("train", []) or []
    val_ids = split_payload.get("val", []) or []
    if not train_ids:
        raise ValueError("data_split.json has no train IDs.")
    if not val_ids:
        raise ValueError("data_split.json has no val IDs.")

    pdo_change_labels = load_pdo_change_labels(label_json)
    if not pdo_change_labels:
        raise ValueError(f"No valid labels in {label_json}")

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    transformations = MaskImageTransformations(image_size=image_size, normalize_mean=0.5, normalize_std=0.5)

    train_dataset = OnchipPDOChangeDataset(
        masks_dir=masks_dir,
        image_ids=list(train_ids),
        pdo_change_labels=pdo_change_labels,
        mask_names=mask_names,
        transform=transformations.train_transformations,
    )
    val_dataset = OnchipPDOChangeDataset(
        masks_dir=masks_dir,
        image_ids=list(val_ids),
        pdo_change_labels=pdo_change_labels,
        mask_names=mask_names,
        transform=transformations.validation_transformations,
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
    model = PDOChangeResNetClassifier(
        in_channels=len(mask_names),
        num_classes=len(PDO_CHANGE_BIN_LABELS),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    history = []
    early_stop = 0
    early_stop_patience = int(config.get("early_stop_patience", 15))

    epochs = int(config["epochs"])
    epoch_iter = tqdm(range(1, epochs + 1), desc=f"Training PDO change ({version})")
    for epoch in epoch_iter:
        train_loss, train_acc, train_macro_f1 = epoch_train(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_macro_f1, _, _, _, _ = epoch_eval(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)
        lr_now = float(optimizer.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_macro_f1": train_macro_f1,
            "val_macro_f1": val_macro_f1,
            "learning_rate": lr_now,
        }
        history.append(row)

        epoch_iter.set_postfix(
            {
                "train_acc": f"{train_acc:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "val_f1": f"{val_macro_f1:.4f}",
            }
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/acc": train_acc,
                    "val/acc": val_acc,
                    "train/macro_f1": train_macro_f1,
                    "val/macro_f1": val_macro_f1,
                    "train/lr": lr_now,
                },
                step=epoch,
            )

        if (val_acc > best_val_acc) or (abs(val_acc - best_val_acc) <= 1e-12 and val_loss < best_val_loss):
            best_val_acc = val_acc
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
    val_loss, val_acc, val_macro_f1, val_ids_eval, val_pred_idx, val_true_idx, val_pred_conf = epoch_eval(
        model, val_loader, criterion, device
    )

    pred_csv_path = output_dir / "val_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_id",
                "pdo_change_label_raw",
                "true_bin_idx",
                "true_bin_label",
                "pred_bin_idx",
                "pred_bin_label",
                "pred_confidence",
            ]
        )
        for i, image_id in enumerate(val_ids_eval):
            raw_value = float(pdo_change_labels.get(image_id, float("nan")))
            true_idx = int(val_true_idx[i])
            pred_idx = int(val_pred_idx[i])
            writer.writerow(
                [
                    image_id,
                    raw_value,
                    true_idx,
                    PDO_CHANGE_BIN_LABELS[true_idx],
                    pred_idx,
                    PDO_CHANGE_BIN_LABELS[pred_idx],
                    float(val_pred_conf[i]),
                ]
            )

    history_csv_path = output_dir / "metrics.csv"
    history_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_macro_f1",
        "val_macro_f1",
        "learning_rate",
    ]
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_columns)
        writer.writeheader()
        for row in history:
            writer.writerow({col: row.get(col) for col in history_columns})

    metrics = {
        "version": version,
        "mask_names": list(mask_names),
        "num_classes": len(PDO_CHANGE_BIN_LABELS),
        "bin_labels": list(PDO_CHANGE_BIN_LABELS),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _args_to_config(args) -> dict:
    version = str(args.version)
    mask_names = list(VERSION_TO_MASKS[version])
    return {
        "version": version,
        "masks_dir": str(args.masks_dir),
        "split_json": str(args.split_json),
        "label_json": str(args.label_json),
        "output_dir": str(args.output_dir),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_workers": int(args.num_workers),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "device": str(args.device),
        "early_stop_patience": int(args.early_stop_patience),
        "mask_names": mask_names,
        "output_is_run_dir": False,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _args_to_config(args)
    metrics = train_pdochange(cfg, wandb_run=None)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("WANDB_SILENT", "true")
    raise SystemExit(main())
