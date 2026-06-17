# Demo Runs

The repository includes two-sample synthetic `demo_data/` folders for every
main module. These files are intended to demonstrate the software interfaces,
not biological performance.

All commands below are run from the `ml4cart/` directory in the `cart` conda
environment.

## Regenerate Demo Data

```bash
python scripts/create_demo_data.py
```

Typical runtime: less than 2 seconds.

## Preprocessing Demos

TCGA split generation:

```bash
python -m preprocess.tcga_wsi.split_case_ids \
  --masks-dir preprocess/tcga_wsi/demo_data/masks \
  --output-json /tmp/ml4cart_demo_tcga_split.json \
  --val-ratio 0.5 \
  --seed 1
```

Typical runtime: less than 2 seconds.

On-chip R2 mask-generation dry run:

```bash
python -m preprocess.onchip_wsi.generate_r2_masks \
  --base-dir preprocess/onchip_wsi/demo_data/On-chip_Data_R2 \
  --dry-run
```

Typical runtime: less than 2 seconds.

Dynamics image sorting on a temporary copy:

```bash
cp -r preprocess/dynamics/demo_data/raw_images /tmp/ml4cart_demo_dynamics_raw_images
python -m preprocess.dynamics.sort_images_by_xy \
  --base_folder /tmp/ml4cart_demo_dynamics_raw_images
```

Typical runtime: less than 2 seconds.

## Training Demos

TCGA CD8 distribution:

```bash
python -m tcga_cd8_distribution.train \
  --masks-dir tcga_cd8_distribution/demo_data/masks \
  --split-json tcga_cd8_distribution/demo_data/data_split.json \
  --output-dir /tmp/ml4cart_demo_tcga_cd8 \
  --mask-names cd4 cd68 ck actin pd-1 tissue \
  --image-size 32 \
  --base-channels 2 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --seed 1
```

Typical runtime: about 5 seconds.

On-chip CD8 distribution:

```bash
python -m onchip_cd8_distribution.train \
  --preset actin_ck \
  --masks-dir onchip_cd8_distribution/demo_data/On-chip_Data \
  --split-json onchip_cd8_distribution/demo_data/On-chip_Data/data_split.json \
  --output-dir /tmp/ml4cart_demo_onchip_cd8 \
  --image-size 32 \
  --base-channels 2 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --seed 1
```

Typical runtime: about 5 seconds.

PDO size-change prediction:

```bash
python -m pdo_size_change_prediction.train \
  --version actin_ck \
  --masks-dir pdo_size_change_prediction/demo_data/On-chip_Data \
  --split-json pdo_size_change_prediction/demo_data/On-chip_Data/data_split.json \
  --label-json pdo_size_change_prediction/demo_data/On-chip_Data/pdo_change_label.json \
  --output-dir /tmp/ml4cart_demo_pdo \
  --image-size 64 \
  --hidden-dim 8 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --seed 1
```

Typical runtime: about 6 seconds.

Dynamics response model:

```bash
python -m dynamics_model.train \
  --seq-path dynamics_model/demo_data/generated/trajectory_dataset_100.npz \
  --track-path dynamics_model/demo_data/generated/track_dataset.npz \
  --split-json dynamics_model/demo_data/data_split.json \
  --output-dir /tmp/ml4cart_demo_dynamics \
  --training-method ce \
  --use-class-weight false \
  --max-epochs 1 \
  --batch-size 2 \
  --hidden-size 4 \
  --fusion-size 4 \
  --use-wandb false \
  --seed 1
```

Typical runtime: about 8 seconds. Statistical warnings can appear because the
demo split has only one validation case; the command should still complete and
write `/tmp/ml4cart_demo_dynamics/run_manual/`.

## Analysis Demos

On-chip raw 16-tile analysis:

```bash
python -m onchip_distribution_analysis.analyze_raw_mask_16_tiles \
  --cohort r1 \
  --masks-dir onchip_distribution_analysis/demo_data/On-chip_Data \
  --split-json onchip_distribution_analysis/demo_data/On-chip_Data/data_split.json \
  --splits train val \
  --tile-size 16 \
  --output-dir /tmp/ml4cart_demo_onchip_raw16
```

Typical runtime: about 5 seconds.

Dynamics clustering:

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

Typical runtime: about 4 seconds.

## Full Smoke Test

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ml4cart-mpl-cache \
  python -m pytest -q -p no:cacheprovider
```

Typical runtime: 15-25 seconds. Expected result: `7 passed`.
