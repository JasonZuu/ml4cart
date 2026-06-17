from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

TCGA_IDS = ["TCGA-DEMO-0001-01Z-00-DX1", "TCGA-DEMO-0002-01Z-00-DX1"]
ONCHIP_IDS = ["chip-r1_demo-001", "chip-r1_demo-002"]
ONCHIP_DIRS = ["Chip-R1_DEMO-001", "Chip-R1_DEMO-002"]
DYNAMICS_CASES = ["DemoCaseA", "DemoCaseB"]
R2_WSI_ROOT = "Chip WSI_20260416_8 Patients CART, stroma, immune, and PDO for AI_Round 1"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mask_array(size: int, sample_idx: int, channel_idx: int) -> np.ndarray:
    yy, xx = np.indices((size, size))
    if channel_idx % 4 == 0:
        arr = ((xx + sample_idx * 5) % 16) < 8
    elif channel_idx % 4 == 1:
        arr = ((yy + channel_idx * 3) % 18) < 9
    elif channel_idx % 4 == 2:
        cx = size * (0.35 + 0.18 * sample_idx)
        cy = size * (0.45 + 0.10 * channel_idx)
        arr = (xx - cx) ** 2 + (yy - cy) ** 2 < (size * 0.22) ** 2
    else:
        arr = ((xx + yy + sample_idx * 7 + channel_idx * 3) % 20) < 10
    return (arr.astype(np.uint8) * 255)


def write_mask(case_dir: Path, name: str, sample_idx: int, channel_idx: int, size: int) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_array(size, sample_idx, channel_idx)).save(case_dir / f"{name}.png")


def write_tif(path: Path, sample_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = mask_array(32, sample_idx, sample_idx + 1)
    Image.fromarray(arr).save(path)


def write_split(path: Path, ids: list[str], include_test: bool = False) -> None:
    payload = {"train": ids, "val": ids}
    if include_test:
        payload["test"] = []
    write_json(path, payload)


def write_tcga_mask_demo(base: Path) -> None:
    channels = ["cd4", "cd68", "ck", "actin", "pd-1", "tissue", "cd8"]
    for sample_idx, case_id in enumerate(TCGA_IDS):
        case_dir = base / "masks" / case_id
        for channel_idx, channel in enumerate(channels):
            write_mask(case_dir, channel, sample_idx, channel_idx, size=32)
    write_split(base / "data_split.json", TCGA_IDS)


def write_onchip_mask_demo(base: Path, include_dapi: bool = True) -> None:
    channels = ["actin", "ck", "cd8", "cd68"]
    if include_dapi:
        channels.append("dapi")
    mask_root = base / "On-chip_Data" / "Chip-R1_mask"
    for sample_idx, folder in enumerate(ONCHIP_DIRS):
        case_dir = mask_root / folder
        for channel_idx, channel in enumerate(channels):
            write_mask(case_dir, channel, sample_idx, channel_idx, size=64)
    write_json(base / "On-chip_Data" / "data_split.json", {"train": ONCHIP_IDS, "val": ONCHIP_IDS, "test": []})
    write_json(
        base / "On-chip_Data" / "image_id_mapping.json",
        {image_id: folder for image_id, folder in zip(ONCHIP_IDS, ONCHIP_DIRS)},
    )
    write_json(base / "On-chip_Data" / "pdo_change_label.json", {ONCHIP_IDS[0]: -47.5, ONCHIP_IDS[1]: 28.0})


def write_dynamics_npz_demo(base: Path) -> None:
    generated = base / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    track_ids = np.array([[f"{case}_track0", 0] for case in DYNAMICS_CASES], dtype=object)
    seq = rng.normal(size=(2, 100, 7)).astype(np.float32)
    track = rng.normal(size=(2, 3)).astype(np.float32)
    np.savez(generated / "trajectory_dataset_100.npz", X=seq, track_ids=track_ids)
    np.savez(generated / "track_dataset.npz", X=track, track_ids=track_ids)
    meta = {
        "label_by_case": {DYNAMICS_CASES[0]: 0, DYNAMICS_CASES[1]: 1},
        "size_change_by_case": {DYNAMICS_CASES[0]: -42.0, DYNAMICS_CASES[1]: 21.0},
        "pdo_size": {DYNAMICS_CASES[0]: 120.0, DYNAMICS_CASES[1]: 180.0},
        "antigen": {DYNAMICS_CASES[0]: 40.0, DYNAMICS_CASES[1]: 72.0},
        "COL-I": {DYNAMICS_CASES[0]: 0.2, DYNAMICS_CASES[1]: 0.5},
        "COL-III": {DYNAMICS_CASES[0]: 0.3, DYNAMICS_CASES[1]: 0.4},
        "COL-IV": {DYNAMICS_CASES[0]: 0.1, DYNAMICS_CASES[1]: 0.6},
        "HA": {DYNAMICS_CASES[0]: 0.2, DYNAMICS_CASES[1]: 0.7},
        "PD-1": {DYNAMICS_CASES[0]: 0.4, DYNAMICS_CASES[1]: 0.2},
        "LAG-3": {DYNAMICS_CASES[0]: 0.5, DYNAMICS_CASES[1]: 0.3},
    }
    write_json(
        base / "data_split.json",
        {"train": [DYNAMICS_CASES[0]], "val": [DYNAMICS_CASES[1]], "test": [], "meta": meta},
    )


def write_dynamics_analysis_demo(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    write_json(base / "data_split.json", {"val": DYNAMICS_CASES})
    rows = [
        {
            "PREFIX": "DemoCaseA_XY1",
            "case_name": "DemoCaseA",
            "LABEL": 0,
            "PERIMETER_mean": 11.0,
            "CIRCULARITY_mean": 0.82,
            "ELLIPSE_ASPECTRATIO_mean": 1.20,
            "SOLIDITY_mean": 0.91,
            "SPEED_mean": 0.34,
            "MEAN_SQUARE_DISPLACEMENT_mean": 0.10,
            "TRACK_DISPLACEMENT_mean": 4.1,
            "TRACK_STD_SPEED_mean": 0.05,
            "MEAN_DIRECTIONAL_CHANGE_RATE_mean": 0.12,
        },
        {
            "PREFIX": "DemoCaseB_XY1",
            "case_name": "DemoCaseB",
            "LABEL": 1,
            "PERIMETER_mean": 16.0,
            "CIRCULARITY_mean": 0.64,
            "ELLIPSE_ASPECTRATIO_mean": 1.80,
            "SOLIDITY_mean": 0.84,
            "SPEED_mean": 0.68,
            "MEAN_SQUARE_DISPLACEMENT_mean": 0.24,
            "TRACK_DISPLACEMENT_mean": 6.7,
            "TRACK_STD_SPEED_mean": 0.11,
            "MEAN_DIRECTIONAL_CHANGE_RATE_mean": 0.22,
        },
    ]
    with (base / "features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_preprocess_tcga_demo() -> None:
    base = ROOT / "preprocess" / "tcga_wsi" / "demo_data"
    write_tcga_mask_demo(base)
    for idx, case_id in enumerate(TCGA_IDS):
        slide = base / "raw_wsi" / f"download_{idx + 1}" / f"{case_id}.svs"
        slide.parent.mkdir(parents=True, exist_ok=True)
        slide.write_text("Synthetic placeholder slide for flatten_wsi demos.\n", encoding="utf-8")


def write_preprocess_onchip_demo() -> None:
    base = ROOT / "preprocess" / "onchip_wsi" / "demo_data"
    write_onchip_mask_demo(base)
    for sample_idx, patient in enumerate(["NYU001", "NYU002"], start=1):
        sample_dir = (
            base
            / "On-chip_Data_R2"
            / R2_WSI_ROOT
            / patient
            / f"BV421-CD68_488-a-SMA_PE-CD8_647-CK19_{sample_idx}_{patient}_d1"
        )
        for channel_idx, tag in enumerate(["405", "561", "640", "BF"]):
            write_tif(sample_dir / f"Demo RGB {tag}.tif", sample_idx + channel_idx)


def write_preprocess_dynamics_demo() -> None:
    base = ROOT / "preprocess" / "dynamics" / "demo_data"
    write_tif(base / "raw_images" / "DemoExperiment" / "DemoCaseA_t000_XY1.tif", 0)
    write_tif(base / "raw_images" / "DemoExperiment" / "DemoCaseB_t000_XY2.tif", 1)
    write_dynamics_npz_demo(base)


def write_demo_readme(path: Path, lines: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_all_demo_data() -> None:
    write_preprocess_tcga_demo()
    write_preprocess_onchip_demo()
    write_preprocess_dynamics_demo()

    write_tcga_mask_demo(ROOT / "tcga_cd8_distribution" / "demo_data")
    write_onchip_mask_demo(ROOT / "onchip_cd8_distribution" / "demo_data")
    write_onchip_mask_demo(ROOT / "pdo_size_change_prediction" / "demo_data")
    write_onchip_mask_demo(ROOT / "onchip_distribution_analysis" / "demo_data")
    write_dynamics_npz_demo(ROOT / "dynamics_model" / "demo_data")
    write_dynamics_analysis_demo(ROOT / "dynamics_analysis" / "demo_data")

    write_demo_readme(
        ROOT / "tcga_cd8_distribution" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two synthetic TCGA-like mask samples are stored in `masks/` with `data_split.json`.",
        ],
    )
    write_demo_readme(
        ROOT / "onchip_cd8_distribution" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two synthetic on-chip samples are stored under `On-chip_Data/Chip-R1_mask/`.",
        ],
    )
    write_demo_readme(
        ROOT / "pdo_size_change_prediction" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two synthetic on-chip samples include masks, split JSON, and PDO size-change labels.",
        ],
    )
    write_demo_readme(
        ROOT / "dynamics_model" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two synthetic tracks are stored in `generated/*.npz` with labels in `data_split.json`.",
        ],
    )
    write_demo_readme(
        ROOT / "preprocess" / "tcga_wsi" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two placeholder `.svs` files and two TCGA-like mask folders are provided for split/flatten demos.",
        ],
    )
    write_demo_readme(
        ROOT / "preprocess" / "onchip_wsi" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two processed on-chip mask samples and two raw R2-style channel exports are provided.",
        ],
    )
    write_demo_readme(
        ROOT / "preprocess" / "dynamics" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two raw `.tif` images plus tiny generated `.npz` outputs are provided.",
        ],
    )
    write_demo_readme(
        ROOT / "dynamics_analysis" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two image-level dynamics rows are provided in `features.csv` with a matching split JSON.",
        ],
    )
    write_demo_readme(
        ROOT / "onchip_distribution_analysis" / "demo_data",
        [
            "# Demo Data",
            "",
            "Two synthetic on-chip mask samples are provided for raw 16-tile analysis demos.",
        ],
    )


def main() -> int:
    write_all_demo_data()
    print(f"Wrote demo_data folders under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
