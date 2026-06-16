# Dynamics Analysis

## Purpose

This folder contains downstream analysis scripts for dynamics-derived features.
The analyses are independent of model training and are meant to summarize
feature structure, cluster dynamics phenotypes, and correlate TME/dynamics
features with CAR-T response.

## Inputs

Typical inputs:

- Image-level feature CSVs such as `val_dynamics.csv` and `test_dynamics.csv`.
- Split JSON such as `../dynamics_data/data_split.json`.
- Optional cell-level TrackMate CSVs:
  - `../dynamics_data/generated/unscaled_spot_features.csv`
  - `../dynamics_data/generated/unscaled_track_features.csv`
- Dynamics sequence/track `.npz` files for Spearman scripts that need model
  feature names.

## Demo Data

`demo_data/` contains two image-level dynamics rows:

```text
demo_data/
  features.csv
  data_split.json
```

`features.csv` has `PREFIX`, `case_name`, `LABEL`, and the numeric dynamics
feature columns used by clustering. `data_split.json` assigns `DemoCaseA` and
`DemoCaseB` to the `val` split. Run a minimal clustering demo from `ml4cart/`:

```bash
python -m dynamics_analysis.cluster \
  --split val \
  --data_split_json dynamics_analysis/demo_data/data_split.json \
  --input_csv dynamics_analysis/demo_data/features.csv \
  --level image \
  --reducer tsne \
  --clusterer kmeans \
  --kmeans_n_clusters 2 \
  --out_dir /tmp/ml4cart_demo_dynamics_cluster
```

## Cluster Analysis

Image-level clustering:

```bash
python -m dynamics_analysis.cluster \
  --split val \
  --data_split_json ../dynamics_data/data_split.json \
  --input_csv ../val_dynamics.csv \
  --level image \
  --reducer tsne \
  --clusterer kmeans \
  --kmeans_n_clusters 4 \
  --out_dir dynamics_analysis/cluster/val_results/image_level
```

Cell-level clustering:

```bash
python -m dynamics_analysis.cluster \
  --split test_NYU318 \
  --data_split_json ../dynamics_data/data_split.json \
  --input_csv ../test_dynamics.csv \
  --level cell \
  --spot_csv ../dynamics_data/generated/unscaled_spot_features.csv \
  --track_csv ../dynamics_data/generated/unscaled_track_features.csv \
  --clusterer dbscan \
  --dbscan_eps 0.5 \
  --dbscan_min_samples 3 \
  --out_dir dynamics_analysis/cluster/test_NYU318_results/cell_level
```

## Spearman Analyses

Bar plot:

```bash
python -m dynamics_analysis.spearman_bar \
  --seq_path ../dynamics_data/generated/trajectory_dataset_100.npz \
  --track_path ../dynamics_data/generated/track_dataset.npz \
  --split_path ../dynamics_data/data_split.json \
  --cart_csv_path ../data/CART_Data/tme_feature.json \
  --splits val test_NYU285 test_NYU318 test_NYU774 \
  --output_dir dynamics_analysis/spearman/results
```

Graph plot:

```bash
python -m dynamics_analysis.spearman_graph --help
```

All-feature analysis:

```bash
python -m dynamics_analysis.spearman_all --help
```

## Outputs

Cluster scripts write embeddings, cluster labels, heatmaps, and summary plots.
Spearman scripts write CSV summaries and publication-style plots.

## Smoke Test

```bash
python -m pytest tests/test_dynamics_and_analysis_smoke.py::test_dynamics_cluster_image_level -q
```
