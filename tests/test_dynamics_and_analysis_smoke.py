from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def test_dynamics_model_train_one_epoch(tmp_path):
    from dynamics_model.train import main

    cases = ["CaseA", "CaseB", "CaseC", "CaseD", "CaseE", "CaseF"]
    labels = {"CaseA": 0, "CaseB": 1, "CaseC": 2, "CaseD": 0, "CaseE": 1, "CaseF": 2}
    track_ids = np.array([[f"{case}_track", 0] for case in cases], dtype=object)
    rng = np.random.default_rng(1)
    seq_path = tmp_path / "trajectory_dataset_100.npz"
    track_path = tmp_path / "track_dataset.npz"
    np.savez(seq_path, X=rng.normal(size=(len(cases), 100, 7)).astype(np.float32), track_ids=track_ids)
    np.savez(track_path, X=rng.normal(size=(len(cases), 3)).astype(np.float32), track_ids=track_ids)

    meta = {
        "label_by_case": labels,
        "pdo_size": {case: float(i + 1) for i, case in enumerate(cases)},
        "antigen": {case: float(i + 2) for i, case in enumerate(cases)},
        "COL-I": {case: float(i + 3) for i, case in enumerate(cases)},
        "COL-III": {case: float(i + 4) for i, case in enumerate(cases)},
        "COL-IV": {case: float(i + 5) for i, case in enumerate(cases)},
        "HA": {case: float(i + 6) for i, case in enumerate(cases)},
        "PD-1": {case: float(i + 7) for i, case in enumerate(cases)},
        "LAG-3": {case: float(i + 8) for i, case in enumerate(cases)},
    }
    split_json = tmp_path / "dynamics_split.json"
    split_json.write_text(
        json.dumps({"train": cases[:3], "val": cases[3:], "test": [], "meta": meta}, indent=2),
        encoding="utf-8",
    )

    rc = main(
        [
            "--seq-path",
            str(seq_path),
            "--track-path",
            str(track_path),
            "--split-json",
            str(split_json),
            "--output-dir",
            str(tmp_path / "dyn_out"),
            "--training-method",
            "ce",
            "--max-epochs",
            "1",
            "--batch-size",
            "3",
            "--hidden-size",
            "4",
            "--fusion-size",
            "4",
            "--use-wandb",
            "false",
        ]
    )
    assert rc == 0
    assert (tmp_path / "dyn_out" / "run_manual" / "best_model.pth").exists()


def test_dynamics_cluster_image_level(tmp_path, monkeypatch):
    from dynamics_analysis import cluster

    split_json = tmp_path / "split.json"
    split_json.write_text(json.dumps({"val": ["CaseA", "CaseB"]}), encoding="utf-8")
    csv_path = tmp_path / "features.csv"
    rows = []
    for i in range(8):
        rows.append(
            {
                "PREFIX": f"Case{'A' if i < 4 else 'B'}_XY{i}",
                "case_name": "CaseA" if i < 4 else "CaseB",
                "LABEL": i % 3,
                "PERIMETER_mean": 1.0 + i,
                "CIRCULARITY_mean": 0.1 + i,
                "ELLIPSE_ASPECTRATIO_mean": 0.2 + i,
                "SOLIDITY_mean": 0.3 + i,
                "SPEED_mean": 0.4 + i,
                "MEAN_SQUARE_DISPLACEMENT_mean": 0.5 + i,
                "TRACK_DISPLACEMENT_mean": 0.6 + i,
                "TRACK_STD_SPEED_mean": 0.7 + i,
                "MEAN_DIRECTIONAL_CHANGE_RATE_mean": 0.8 + i,
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    out_dir = tmp_path / "cluster_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster",
            "--split",
            "val",
            "--data_split_json",
            str(split_json),
            "--input_csv",
            str(csv_path),
            "--out_dir",
            str(out_dir),
            "--kmeans_n_clusters",
            "2",
        ],
    )
    cluster.main()
    assert any(out_dir.iterdir())


def test_onchip_raw_mask_analysis_with_tmp_cohort(tmp_path, monkeypatch):
    from onchip_distribution_analysis import analyze_raw_mask_16_tiles as raw

    masks_dir = tmp_path / "masks"
    sample_dir = masks_dir / "sample1"
    sample_dir.mkdir(parents=True)
    arr = np.zeros((32, 32), dtype=np.uint8)
    arr[:, 16:] = 255
    Image.fromarray(arr).save(sample_dir / "actin.png")
    Image.fromarray(np.rot90(arr)).save(sample_dir / "cd8.png")
    Image.fromarray(np.flipud(arr)).save(sample_dir / "ck.png")
    split_json = tmp_path / "split.json"
    split_json.write_text(json.dumps({"train": ["sample1"], "val": [], "test": []}), encoding="utf-8")
    monkeypatch.setitem(raw.COHORTS, "tmp", raw.CohortSpec("tmp", masks_dir, split_json))
    out_dir = tmp_path / "raw_out"
    rc = raw.main(["--cohort", "tmp", "--splits", "train", "--output-dir", str(out_dir), "--tile-size", "16"])
    assert rc == 0
    assert (out_dir / "raw16_sample_stats.csv").exists()
