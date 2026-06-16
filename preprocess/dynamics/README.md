# Dynamics Preprocess

## Purpose

This preprocessing group prepares time-lapse microscopy data for the dynamics
response model. It organizes images by XY position, runs TrackMate tracking,
builds sequence and track-level datasets, and prepares metadata used for model
training and downstream analysis.

## Inputs

Default root:

```text
../dynamics_data/
```

Typical raw layout:

```text
dynamics_data/
  20260114_8 patients_early CAR T/
    DiD-CAR T_1_NYU318_30ms_2percent/
      *.tif
```

Generated outputs are written under:

```text
../dynamics_data/generated/
```

## Demo Data

`demo_data/` contains two synthetic examples:

```text
demo_data/
  raw_images/
    DemoExperiment/
      DemoCaseA_t000_XY1.tif
      DemoCaseB_t000_XY2.tif
  generated/
    trajectory_dataset_100.npz
    track_dataset.npz
  data_split.json
```

Use a temporary copy when testing the sorter, because sorting moves files into
`XY#` folders:

```bash
cp -r preprocess/dynamics/demo_data/raw_images /tmp/ml4cart_demo_dynamics_raw_images
python -m preprocess.dynamics.sort_images_by_xy \
  --base_folder /tmp/ml4cart_demo_dynamics_raw_images
```

The tiny generated `.npz` files are the same format used by
`dynamics_model.train`:

```bash
python -m dynamics_model.train \
  --seq-path preprocess/dynamics/demo_data/generated/trajectory_dataset_100.npz \
  --track-path preprocess/dynamics/demo_data/generated/track_dataset.npz \
  --split-json preprocess/dynamics/demo_data/data_split.json \
  --output-dir /tmp/ml4cart_demo_preprocess_dynamics_model \
  --training-method ce \
  --use-class-weight false \
  --max-epochs 1 \
  --batch-size 2 \
  --hidden-size 4 \
  --fusion-size 4 \
  --use-wandb false
```

## Step 1: Sort Images by XY

```bash
python -m preprocess.dynamics.sort_images_by_xy \
  --base_folder "../dynamics_data/20260114_8 patients_early CAR T"
```

This moves images into `XY#` folders based on filename patterns.

## Step 2: Track Cells with TrackMate

TrackMate uses Fiji/ImageJ. The Fiji path and dataset configs are defined in
`dynamics_model/config.py`.

```bash
python -m preprocess.dynamics.track_cells
```

Expected tracking outputs include per-case `*_spots.csv` and `*_tracks.csv`
files.

## Step 3: Build Model Datasets

```bash
python -m preprocess.dynamics.create_dataset \
  --data_roots ../dynamics_data
```

Run all preprocessing steps where supported:

```bash
python -m preprocess.dynamics.create_dataset \
  --run_all \
  --data_roots ../dynamics_data
```

Skip tracking if TrackMate outputs already exist:

```bash
python -m preprocess.dynamics.create_dataset \
  --run_all \
  --skip_track \
  --data_roots ../dynamics_data
```

## Step 4: Metadata Utilities

Build TME feature JSON:

```bash
python -m preprocess.dynamics.build_tme_feature_json --help
```

Impute drug metadata:

```bash
python -m preprocess.dynamics.impute_drug_meta --help
```

## Outputs

Main generated model inputs:

- `trajectory_dataset_100.npz`
- `track_dataset.npz`
- `data_split.json`
- optional metadata JSON files

These outputs feed `dynamics_model` and `dynamics_analysis`.

## Smoke Test

```bash
python -m preprocess.dynamics.create_dataset --help
```
