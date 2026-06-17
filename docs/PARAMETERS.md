# Parameters And Outputs

This document records the main command-line parameters used by the publication
demo runs and the full-data entrypoints.

## Shared Training Parameters

Most training entrypoints support:

- `--output-dir`: directory where checkpoints, metrics, and predictions are
  written.
- `--device`: `cpu`, `cuda`, or an explicit CUDA device when supported.
- `--seed`: random seed.
- `--batch-size`: batch size.
- `--epochs` or `--max-epochs`: number of training epochs.

## TCGA CD8 Distribution

Demo parameters:

```bash
--image-size 32 --base-channels 2 --epochs 1 --batch-size 2 --device cpu --seed 1
```

Full-data defaults use larger masks and `--base-channels 32`. Outputs include
`best_model.pth`, `metrics.csv`, `metrics.json`, and validation prediction
PNGs.

## On-Chip CD8 Distribution

Supported presets:

- `actin_ck`
- `dapi_actin_ck`

Demo parameters:

```bash
--preset actin_ck --image-size 32 --base-channels 2 --epochs 1 --batch-size 2 --device cpu --seed 1
```

Outputs match the TCGA CD8 training outputs.

## PDO Size-Change Prediction

Supported versions:

- `actin_only`
- `actin_ck`
- `cd8_only`
- `cd8_actin_ck`
- `cd8_cd68_actin_ck`

Demo parameters:

```bash
--version actin_ck --image-size 64 --hidden-dim 8 --epochs 1 --batch-size 2 --device cpu --seed 1
```

Outputs include `best_model.pth`, `metrics.csv`, `metrics.json`, and
`val_predictions.csv`.

## Dynamics Response Prediction

Demo parameters:

```bash
--training-method ce --use-class-weight false --max-epochs 1 --batch-size 2 --hidden-size 4 --fusion-size 4 --use-wandb false --seed 1
```

`--use-class-weight false` is used for the two-sample demo because the demo does
not contain all response classes in the training split. Full training can use
the default focal loss and class weighting.

## Analysis Outputs

- `onchip_distribution_analysis.analyze_raw_mask_16_tiles` writes
  `raw16_sample_stats.csv`, `raw16_cohort_stats.csv`, and
  `raw16_cohort_stats.json`.
- `dynamics_analysis.cluster` writes embedding CSVs, cluster labels, plots, and
  summaries to the selected `--out_dir`.
