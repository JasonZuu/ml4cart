# On-Chip WSI Preprocess

## Purpose

This preprocessing group converts on-chip WSI channel exports into standalone
mask folders and metadata files used by the on-chip CD8 and PDO size-change
experiments.

It supports:

- R1/R2/R3 conversion into TCGA-style mask folders.
- R2-specific mask generation from the newer export layout.
- R4 drug-evaluation mask generation.
- PDO size-change label extraction from spreadsheet files.

## Inputs

Default roots:

```text
../data/On-chip_Data/
../data/On-chip_Data_R2/
```

Expected outputs from preprocessing include:

```text
On-chip_Data/
  Chip-R1_mask/
  Chip-R2_mask/
  Chip-R3_mask/
  Chip-R4_mask/
  data_split.json
  image_id_mapping.json
  pdo_change_label.json
```

## Demo Data

`demo_data/` contains two simulated samples in both processed and raw-R2 style:

```text
demo_data/
  On-chip_Data/
    Chip-R1_mask/
      Chip-R1_DEMO-001/
        actin.png ck.png cd8.png cd68.png dapi.png
      Chip-R1_DEMO-002/
        actin.png ck.png cd8.png cd68.png dapi.png
    data_split.json
    image_id_mapping.json
    pdo_change_label.json
  On-chip_Data_R2/
    Chip WSI_..._Round 1/
      NYU001/BV421-CD68_488-a-SMA_PE-CD8_647-CK19_1_NYU001_d1/
        Demo RGB 405.tif Demo RGB 561.tif Demo RGB 640.tif Demo RGB BF.tif
      NYU002/BV421-CD68_488-a-SMA_PE-CD8_647-CK19_2_NYU002_d1/
        Demo RGB 405.tif Demo RGB 561.tif Demo RGB 640.tif Demo RGB BF.tif
```

Preview R2 mask generation without writing outputs:

```bash
python -m preprocess.onchip_wsi.generate_r2_masks \
  --base-dir preprocess/onchip_wsi/demo_data/On-chip_Data_R2 \
  --dry-run
```

The processed `On-chip_Data/` demo can also be used directly by
`onchip_cd8_distribution`, `pdo_size_change_prediction`, and
`onchip_distribution_analysis`.

## Convert R1/R2/R3 Masks

Dry-run style inspection is not available for this script, so use a separate
output copy if you want to test without touching existing masks.

```bash
python -m preprocess.onchip_wsi.convert_masks \
  --base-dir ../data/On-chip_Data \
  --rounds R1,R2,R3 \
  --skip-rename
```

Overwrite existing mask PNGs when intentionally regenerating:

```bash
python -m preprocess.onchip_wsi.convert_masks \
  --base-dir ../data/On-chip_Data \
  --rounds R1,R2,R3 \
  --overwrite
```

## Generate R2 Masks

```bash
python -m preprocess.onchip_wsi.generate_r2_masks \
  --base-dir ../data/On-chip_Data_R2 \
  --overwrite
```

Preview actions without writing:

```bash
python -m preprocess.onchip_wsi.generate_r2_masks \
  --base-dir ../data/On-chip_Data_R2 \
  --dry-run
```

## Generate R4 Drug-Evaluation Masks

```bash
python -m preprocess.onchip_wsi.generate_r4_masks \
  --base-dir ../data/On-chip_Data \
  --overwrite
```

## Build PDO Size-Change Labels

```bash
python -m preprocess.onchip_wsi.build_pdo_change_label \
  --base-dir ../data/On-chip_Data \
  --output ../data/On-chip_Data/pdo_change_label.json
```

## Outputs

- Mask PNG folders.
- `image_id_mapping.json`.
- `data_split.json`.
- `pdo_change_label.json`.

These outputs feed `onchip_cd8_distribution`, `pdo_size_change_prediction`, and
`onchip_distribution_analysis`.

## Smoke Test

```bash
python -m preprocess.onchip_wsi.convert_masks --help
python -m preprocess.onchip_wsi.generate_r2_masks --help
python -m preprocess.onchip_wsi.build_pdo_change_label --help
```
