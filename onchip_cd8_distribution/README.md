# On-Chip CD8 Distribution

## Purpose

This experiment trains a U-Net model to predict on-chip CD8 mask distribution
from other on-chip mask channels. It is used to model where CAR-T/CD8 signal is
expected under different tumor and stromal contexts.

Two presets are supported:

- `actin_ck`: uses actin + CK as input.
- `dapi_actin_ck`: uses DAPI + actin + CK as input.

## Inputs

Default paths:

- Masks: `../data/On-chip_Data`
- Split JSON: `../data/On-chip_Data/data_split.json`

Expected mask directory structure:

```text
On-chip_Data/
  Chip-R1_mask/
    Chip-R1_NYU318-d1/
      actin.png
      ck.png
      cd8.png
      dapi.png
```

The dataset loader can also resolve case directories recursively, so split IDs
such as `chip-r1_nyu318-d1` can map to normalized mask folder names.

## Demo Data

`demo_data/` contains two synthetic on-chip samples:

```text
demo_data/
  On-chip_Data/
    Chip-R1_mask/
      Chip-R1_DEMO-001/
        actin.png ck.png cd8.png dapi.png cd68.png
      Chip-R1_DEMO-002/
        actin.png ck.png cd8.png dapi.png cd68.png
    data_split.json
    image_id_mapping.json
    pdo_change_label.json
```

The split IDs are `chip-r1_demo-001` and `chip-r1_demo-002`. Run a 1-epoch
CPU demo from `ml4cart/`:

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

To check the DAPI preset, change `--preset actin_ck` to
`--preset dapi_actin_ck`.

## Train

Train the actin + CK model:

```bash
python -m onchip_cd8_distribution.train \
  --preset actin_ck \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --output-dir onchip_cd8_distribution/results/actin_ck \
  --image-size 512 \
  --base-channels 32 \
  --epochs 60 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --bce-weight 0.1 \
  --dice-weight 0.9 \
  --device cuda \
  --seed 1
```

Train the DAPI + actin + CK model:

```bash
python -m onchip_cd8_distribution.train \
  --preset dapi_actin_ck \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --output-dir onchip_cd8_distribution/results/dapi_actin_ck \
  --epochs 60 \
  --batch-size 4 \
  --device cuda
```

Override channels manually when needed:

```bash
python -m onchip_cd8_distribution.train \
  --preset actin_ck \
  --mask-names actin ck \
  --output-dir onchip_cd8_distribution/results/custom_actin_ck
```

## Inference and Analysis

Run inference from a trained checkpoint:

```bash
python -m onchip_cd8_distribution.run_inference \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --checkpoint onchip_cd8_distribution/results/actin_ck/best_model.pth \
  --output-dir onchip_cd8_distribution/results/actin_ck/inference \
  --device cuda
```

Additional entrypoints:

```bash
python -m onchip_cd8_distribution.run_shap_analysis --help
python -m onchip_cd8_distribution.run_r4_drug_comparison --help
```

## Outputs

Training writes:

- `best_model.pth`
- `metrics.csv`
- `metrics.json`
- `val_predictions/`

Inference and analysis scripts write prediction masks, visualizations, and
comparison artifacts under the selected `--output-dir`.

## Smoke Test

```bash
python -m pytest tests/test_training_smoke.py::test_onchip_cd8_train_one_epoch -q
```
