# TCGA WSI Preprocess

## Purpose

This preprocessing group prepares TCGA WSI data for mask-based modeling. It
normalizes downloaded WSI file locations, generates train/validation split
files, and provides optional GigaTIME processing utilities for mask and
annotation generation.

## Inputs

Typical raw WSI layout:

```text
../data/TCGA_Data/
  TCGA_WSI/
    nested_download_folder/
      TCGA-...svs
```

Model-ready mask layout:

```text
../data/TCGA_Data/TCGA_WSI_masks/
  TCGA-...-DX1/
    cd4.png
    cd68.png
    ck.png
    actin.png
    pd-1.png
    tissue.png
    cd8.png
```

## Demo Data

`demo_data/` contains two synthetic TCGA-like cases:

```text
demo_data/
  raw_wsi/
    download_1/TCGA-DEMO-0001-01Z-00-DX1.svs
    download_2/TCGA-DEMO-0002-01Z-00-DX1.svs
  masks/
    TCGA-DEMO-0001-01Z-00-DX1/
      cd4.png cd68.png ck.png actin.png pd-1.png tissue.png cd8.png
    TCGA-DEMO-0002-01Z-00-DX1/
      cd4.png cd68.png ck.png actin.png pd-1.png tissue.png cd8.png
  data_split.json
```

The `.svs` files are placeholders for testing path handling; the PNG masks are
small synthetic masks that can be split directly:

```bash
python -m preprocess.tcga_wsi.split_case_ids \
  --masks-dir preprocess/tcga_wsi/demo_data/masks \
  --output-json /tmp/ml4cart_demo_tcga_split.json \
  --val-ratio 0.5 \
  --seed 1
```

Preview WSI flattening on a copy of the placeholders:

```bash
cp -r preprocess/tcga_wsi/demo_data/raw_wsi /tmp/ml4cart_demo_tcga_raw_wsi
python -m preprocess.tcga_wsi.flatten_wsi \
  --root-dir /tmp/ml4cart_demo_tcga_raw_wsi \
  --dry-run
```

## Step 1: Flatten WSI Files

Use this when TCGA `.svs` slides are nested inside download subfolders:

```bash
python -m preprocess.tcga_wsi.flatten_wsi \
  --root-dir ../data/TCGA_Data/TCGA_WSI \
  --dry-run
```

If the dry run looks correct:

```bash
python -m preprocess.tcga_wsi.flatten_wsi \
  --root-dir ../data/TCGA_Data/TCGA_WSI \
  --overwrite
```

## Step 2: Generate Split JSON

Generate a train/validation split from available mask folders:

```bash
python -m preprocess.tcga_wsi.split_case_ids \
  --masks-dir ../data/TCGA_Data/TCGA_WSI_masks \
  --output-json ../data/TCGA_Data/data_split.json \
  --val-ratio 0.1 \
  --seed 1
```

## Step 3: Optional WSI Processing

GigaTIME processing:

```bash
python -m preprocess.tcga_wsi.run_gigatime_processing \
  --input-dir ../data/TCGA_Data/TCGA_WSI \
  --output-dir ../data/TCGA_Data/TCGA_gigatime
```

Extract GigaTIME masks:

```bash
python -m preprocess.tcga_wsi.extract_gigatime_masks \
  --input-dir ../data/TCGA_Data/TCGA_gigatime \
  --output-dir ../data/TCGA_Data/TCGA_WSI_masks
```

Generate annotation CSVs:

```bash
python -m preprocess.tcga_wsi.generate_annotations --help
```

## Outputs

- Flattened `.svs` files under `TCGA_WSI/`.
- `data_split.json`.
- Optional GigaTIME outputs.
- Mask PNG folders used by `tcga_cd8_distribution`.

## Smoke Test

```bash
python -m preprocess.tcga_wsi.split_case_ids --help
```
