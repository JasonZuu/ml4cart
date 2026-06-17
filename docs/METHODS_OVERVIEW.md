# Methods Overview

This document summarizes the key operations performed by the software and the
general approach used by each module.

## TCGA WSI Preprocessing

The TCGA preprocessing module flattens nested WSI download folders, generates
case-level train/validation split JSON files from mask directories, runs
optional GigaTIME processing, extracts model-ready masks, and builds annotation
tables.

## TCGA CD8 Distribution Model

The TCGA CD8 model predicts a CD8 mask from non-CD8 mask channels such as CD4,
CD68, CK, actin, PD-1, and tissue. It uses a compact U-Net implementation in
`common.cd8_model` with shared training utilities in `common.cd8_training`.

## On-Chip WSI Preprocessing

The on-chip preprocessing module converts channel exports into standalone mask
folders, generates R2/R4 mask sets, and builds PDO size-change label JSON files
for downstream supervised learning.

## On-Chip CD8 Distribution Model

The on-chip CD8 model predicts CD8 masks from tumor/stromal context channels.
It supports the `actin_ck` and `dapi_actin_ck` presets and shares the same U-Net
and training loop used by the TCGA CD8 model.

## On-Chip Distribution Analysis

The distribution analysis module computes local relationships among actin,
CD8, and CK masks using native-resolution tile summaries. Additional scripts
support CAF/CD8 local analysis and attribution summaries from trained models.

## PDO Size-Change Prediction

The PDO size-change module predicts binned PDO size change from on-chip mask
channels. It uses a ResNet18 backbone adapted for arbitrary mask-channel counts
and a classifier head over fixed PDO size-change bins.

## Dynamics Preprocessing

The dynamics preprocessing module sorts time-lapse images by XY position, runs
TrackMate-based tracking when external Fiji/ImageJ dependencies are available,
and builds sequence-level and track-level datasets for model training.

## Dynamics Response Model

The dynamics response model predicts response classes from cell trajectory
sequences, track summary features, PDO size, antigen expression, and TME
metadata. The default model is a cross-attention fusion network implemented in
`dynamics_model.model.CrossAttnFusionModel`.

## Dynamics Analysis

The dynamics analysis module clusters image-level or cell-level dynamics
features and computes Spearman-style associations between dynamics/TME features
and response.

## Key Characteristics

- Module entrypoints are runnable with `python -m ...`.
- Real data paths are external and configurable through CLI arguments.
- Shared model, plotting, path, seed, and mask utilities live under `common/`.
- Included `demo_data/` folders provide minimal reproducible examples for the
  main workflows.
