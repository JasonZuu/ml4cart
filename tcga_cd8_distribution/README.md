# TCGA CD8 Distribution

## Purpose

This experiment trains a U-Net model to predict the CD8 mask distribution in
TCGA WSI-derived mask tiles. The input channels are non-CD8 masks such as CD4,
CD68, CK, actin, PD-1, and tissue; the target channel is CD8.

The goal is to learn the spatial relationship between tumor/stromal/immune
mask channels and CD8 localization in TCGA WSI-derived data.

## Inputs

Default paths:

- Masks: `../data/TCGA_Data/TCGA_WSI_masks`
- Split JSON: `../data/TCGA_Data/data_split.json`

Expected mask directory structure:

```text
TCGA_WSI_masks/
  TCGA-...-DX1/
    cd4.png
    cd68.png
    ck.png
    actin.png
    pd-1.png
    tissue.png
    cd8.png
```

The split JSON must contain at least:

```json
{
  "train": ["case_id_1"],
  "val": ["case_id_2"]
}
```

## Demo Data

`demo_data/` contains two synthetic TCGA-like samples:

```text
demo_data/
  masks/
    TCGA-DEMO-0001-01Z-00-DX1/
      cd4.png cd68.png ck.png actin.png pd-1.png tissue.png cd8.png
    TCGA-DEMO-0002-01Z-00-DX1/
      cd4.png cd68.png ck.png actin.png pd-1.png tissue.png cd8.png
  data_split.json
```

Both demo samples are listed in train and val so the tiny 1-epoch run is
stable. Run from `ml4cart/`:

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

## Train

Run from `ml4cart/`:

```bash
python -m tcga_cd8_distribution.train \
  --masks-dir ../data/TCGA_Data/TCGA_WSI_masks \
  --split-json ../data/TCGA_Data/data_split.json \
  --output-dir tcga_cd8_distribution/results \
  --mask-names cd4 cd68 ck actin pd-1 tissue \
  --image-size 512 \
  --base-channels 32 \
  --epochs 60 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --device cuda \
  --seed 1
```

For a quick CPU check:

```bash
python -m tcga_cd8_distribution.train \
  --masks-dir <tiny_mask_dir> \
  --split-json <tiny_split.json> \
  --output-dir /tmp/tcga_cd8_smoke \
  --image-size 32 \
  --base-channels 2 \
  --epochs 1 \
  --batch-size 2 \
  --device cpu
```

## Inference and Attribution

After training, use the saved `best_model.pth`:

```bash
python -m tcga_cd8_distribution.run_inference \
  --masks-dir ../data/TCGA_Data/TCGA_WSI_masks \
  --split-json ../data/TCGA_Data/data_split.json \
  --checkpoint tcga_cd8_distribution/results/best_model.pth \
  --output-dir tcga_cd8_distribution/results/inference \
  --device cuda
```

SHAP analysis entrypoint:

```bash
python -m tcga_cd8_distribution.run_shap_analysis --help
```

## Outputs

Training writes:

- `best_model.pth`
- `metrics.csv`
- `metrics.json`
- `val_predictions/` with predicted CD8 masks and labels

## Smoke Test

The project test suite includes a synthetic one-epoch training run for this
module:

```bash
python -m pytest tests/test_training_smoke.py::test_tcga_cd8_train_one_epoch -q
```
