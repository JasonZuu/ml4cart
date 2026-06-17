from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def write_mask(case_dir: Path, name: str, size: int, offset: int = 0) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.indices((size, size))
    arr = (((xx + yy + offset) % 5) > 1).astype(np.uint8) * 255
    Image.fromarray(arr).save(case_dir / f"{name}.png")


def write_split(path: Path, train: list[str], val: list[str]) -> None:
    path.write_text(json.dumps({"train": train, "val": val}, indent=2), encoding="utf-8")


def test_tcga_cd8_train_one_epoch(tmp_path):
    from tcga_cd8_distribution.train import train_unet

    masks_dir = tmp_path / "tcga_masks"
    train_ids = ["TCGA-A-01", "TCGA-A-02"]
    val_ids = ["TCGA-A-03", "TCGA-A-04"]
    names = ["cd4", "cd68", "ck", "actin", "pd-1", "tissue", "cd8"]
    for idx, case_id in enumerate(train_ids + val_ids):
        for name in names:
            write_mask(masks_dir / case_id, name, 32, offset=idx + len(name))
    split_json = tmp_path / "tcga_split.json"
    write_split(split_json, train_ids, val_ids)

    metrics = train_unet(
        {
            "masks_dir": str(masks_dir),
            "split_json": str(split_json),
            "output_dir": str(tmp_path / "tcga_out"),
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "image_size": 32,
            "base_channels": 2,
            "seed": 1,
            "device": "cpu",
            "mask_names": ["cd4", "cd68", "ck", "actin", "pd-1", "tissue"],
            "early_stop_patience": 2,
        }
    )
    assert metrics["num_train_samples"] > 0
    assert (tmp_path / "tcga_out" / "best_model.pth").exists()


def test_onchip_cd8_train_one_epoch(tmp_path):
    from onchip_cd8_distribution.train import train_unet

    masks_dir = tmp_path / "onchip_masks"
    train_ids = ["chip-r1_case-a", "chip-r1_case-b"]
    val_ids = ["chip-r1_case-c", "chip-r1_case-d"]
    for idx, case_id in enumerate(train_ids + val_ids):
        for name in ["actin", "ck", "cd8"]:
            write_mask(masks_dir / case_id, name, 32, offset=idx + len(name))
    split_json = tmp_path / "onchip_split.json"
    write_split(split_json, train_ids, val_ids)

    metrics = train_unet(
        {
            "masks_dir": str(masks_dir),
            "split_json": str(split_json),
            "output_dir": str(tmp_path / "onchip_cd8_out"),
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "image_size": 32,
            "base_channels": 2,
            "seed": 1,
            "device": "cpu",
            "mask_names": ["actin", "ck"],
            "early_stop_patience": 2,
            "bce_weight": 0.1,
            "dice_weight": 0.9,
            "preset": "actin_ck",
        }
    )
    assert metrics["preset"] == "actin_ck"
    assert (tmp_path / "onchip_cd8_out" / "best_model.pth").exists()


def test_pdo_size_change_train_one_epoch(tmp_path):
    from pdo_size_change_prediction.train import train_pdochange

    masks_dir = tmp_path / "pdo_masks"
    train_ids = ["chip-r1_case-a", "chip-r1_case-b"]
    val_ids = ["chip-r1_case-c", "chip-r1_case-d"]
    for idx, image_id in enumerate(train_ids + val_ids):
        for name in ["actin", "ck"]:
            write_mask(masks_dir / image_id, name, 64, offset=idx + len(name))
    split_json = tmp_path / "pdo_split.json"
    write_split(split_json, train_ids, val_ids)
    label_json = tmp_path / "pdo_labels.json"
    labels = {train_ids[0]: -55.0, train_ids[1]: 15.0, val_ids[0]: -35.0, val_ids[1]: 45.0}
    label_json.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    metrics = train_pdochange(
        {
            "version": "actin_ck",
            "mask_names": ["actin", "ck"],
            "masks_dir": str(masks_dir),
            "split_json": str(split_json),
            "label_json": str(label_json),
            "output_dir": str(tmp_path / "pdo_out"),
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "num_workers": 0,
            "image_size": 64,
            "hidden_dim": 8,
            "dropout": 0.1,
            "seed": 1,
            "device": "cpu",
            "early_stop_patience": 2,
        }
    )
    assert metrics["version"] == "actin_ck"
    assert (tmp_path / "pdo_out" / "actin_ck" / "best_model.pth").exists()
