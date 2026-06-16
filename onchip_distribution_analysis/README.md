# On-Chip Distribution Analysis

## Purpose

This folder analyzes local relationships among actin/CAF signal, tumor mask
signal, CD8 signal, and model attribution. It is separate from training and is
used to quantify whether CD8 density or CD8-restricted GradCAM attribution is
higher in actin-dense microregions than in actin-sparse microregions.

## Inputs

Default cohorts:

- `r1`: `../data/On-chip_Data`
- `r2`: `../data/On-chip_Data_R2`

Expected masks per sample:

```text
sample/
  actin.png
  cd8.png
  ck.png
```

For attribution analyses, trained PDO size-change checkpoints and compatible
mask channels are also required.

## Demo Data

`demo_data/` contains two synthetic on-chip mask samples:

```text
demo_data/
  On-chip_Data/
    Chip-R1_mask/
      Chip-R1_DEMO-001/
        actin.png cd8.png ck.png
      Chip-R1_DEMO-002/
        actin.png cd8.png ck.png
    data_split.json
```

Run the raw 16-tile analysis on the demo masks from `ml4cart/`:

```bash
python -m onchip_distribution_analysis.analyze_raw_mask_16_tiles \
  --cohort r1 \
  --masks-dir onchip_distribution_analysis/demo_data/On-chip_Data \
  --split-json onchip_distribution_analysis/demo_data/On-chip_Data/data_split.json \
  --splits train val \
  --tile-size 16 \
  --output-dir /tmp/ml4cart_demo_onchip_raw16
```

## Raw 16-Tile Mask Analysis

Run R1 analysis across all splits:

```bash
python -m onchip_distribution_analysis.analyze_raw_mask_16_tiles \
  --cohort r1 \
  --splits train val test \
  --tile-size 16 \
  --caf-channel actin \
  --cart-channel cd8 \
  --tumor-channel ck \
  --output-dir onchip_distribution_analysis/results/r1_raw16
```

Run R2 analysis:

```bash
python -m onchip_distribution_analysis.analyze_raw_mask_16_tiles \
  --cohort r2 \
  --splits train val \
  --tile-size 16 \
  --output-dir onchip_distribution_analysis/results/r2_raw16
```

## Attribution Analyses

Local CAF-CD8 analysis:

```bash
python -m onchip_distribution_analysis.analyze_caf_cd8_local --help
```

Combined R1/R2 local analysis:

```bash
python -m onchip_distribution_analysis.analyze_caf_cd8_local_combined --help
```

CD8-restricted SegX-GradCAM analysis:

```bash
python -m onchip_distribution_analysis.analyze_caf_cd8_segxgradcam --help
```

Export report figures:

```bash
python -m onchip_distribution_analysis.export_16tile_bar_report --help
python -m onchip_distribution_analysis.export_figures_and_report --help
```

## Outputs

Raw 16-tile analysis writes:

- `raw16_sample_stats.csv`
- `raw16_cohort_stats.csv`

Attribution and export scripts write figure files, summary CSVs, and report
artifacts under the selected output directories.

## Smoke Test

```bash
python -m pytest tests/test_dynamics_and_analysis_smoke.py::test_onchip_raw_mask_analysis_with_tmp_cohort -q
```
