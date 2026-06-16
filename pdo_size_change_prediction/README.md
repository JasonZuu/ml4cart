# PDO Size Change Prediction

## Purpose

This experiment predicts binned PDO size change from on-chip mask channels. It
uses a ResNet18 classifier with a modified first convolution so different mask
channel combinations can be tested.

The experiment asks which spatial mask channels are informative for PDO
response after CAR-T or drug perturbation.

## Inputs

Default paths:

- Masks: `../data/On-chip_Data`
- Split JSON: `../data/On-chip_Data/data_split.json`
- Labels: `../data/On-chip_Data/pdo_change_label.json`

The label JSON maps image IDs to numeric PDO size-change percentages:

```json
{
  "chip-r1_nyu318-d1": -47.7,
  "chip-r1_nci9-d1": 24.6
}
```

Supported model versions:

- `actin_only`
- `actin_ck`
- `cd8_only`
- `cd8_actin_ck`
- `cd8_cd68_actin_ck`

## Demo Data

`demo_data/` contains two synthetic on-chip samples with masks and PDO
size-change labels:

```text
demo_data/
  On-chip_Data/
    Chip-R1_mask/
      Chip-R1_DEMO-001/
        actin.png ck.png cd8.png cd68.png dapi.png
      Chip-R1_DEMO-002/
        actin.png ck.png cd8.png cd68.png dapi.png
    data_split.json
    pdo_change_label.json
```

The label JSON maps `chip-r1_demo-001` to `-47.5` and `chip-r1_demo-002` to
`28.0`. Run a minimal CPU demo from `ml4cart/`:

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

## Train

Train the CD8 + actin + CK model:

```bash
python -m pdo_size_change_prediction.train \
  --version cd8_actin_ck \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --label-json ../data/On-chip_Data/pdo_change_label.json \
  --output-dir pdo_size_change_prediction/results \
  --image-size 512 \
  --hidden-dim 256 \
  --dropout 0.2 \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --device cuda \
  --seed 1
```

Run all common channel versions:

```bash
python -m pdo_size_change_prediction.train --version actin_only --output-dir pdo_size_change_prediction/results
python -m pdo_size_change_prediction.train --version actin_ck --output-dir pdo_size_change_prediction/results
python -m pdo_size_change_prediction.train --version cd8_only --output-dir pdo_size_change_prediction/results
python -m pdo_size_change_prediction.train --version cd8_actin_ck --output-dir pdo_size_change_prediction/results
python -m pdo_size_change_prediction.train --version cd8_cd68_actin_ck --output-dir pdo_size_change_prediction/results
```

## Model Interpretation and Comparison

Channel importance:

```bash
python -m pdo_size_change_prediction.run_shap_pdochange \
  --checkpoint pdo_size_change_prediction/results/cd8_actin_ck/best_model.pth \
  --version cd8_actin_ck \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --label-json ../data/On-chip_Data/pdo_change_label.json \
  --output-dir pdo_size_change_prediction/results/cd8_actin_ck/shap \
  --device cuda
```

SegX-GradCAM visualization:

```bash
python -m pdo_size_change_prediction.draw_segx_gradcam --help
```

Prediction probability distribution:

```bash
python -m pdo_size_change_prediction.draw_prob_distribution --help
```

Compare two trained versions:

```bash
python -m pdo_size_change_prediction.compare_actin_ck_vs_cd8_actin_ck \
  --model-a pdo_size_change_prediction/results/actin_ck \
  --model-b pdo_size_change_prediction/results/cd8_actin_ck \
  --output-dir pdo_size_change_prediction/results/comparison
```

R4 drug evaluation:

```bash
python -m pdo_size_change_prediction.run_r4_drug_eval --help
```

## Outputs

Each version writes:

- `best_model.pth`
- `metrics.csv`
- `metrics.json`
- `val_predictions.csv`

Analysis scripts write SHAP plots, GradCAM overlays, probability plots, and
comparison tables under their selected output directories.

## Smoke Test

```bash
python -m pytest tests/test_training_smoke.py::test_pdo_size_change_train_one_epoch -q
```
