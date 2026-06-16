# preprocess

## Purpose

Preprocessing is split by data source so each data stream can be prepared
independently before running its model or analysis.

- `tcga_wsi/`: TCGA WSI flattening, split generation, GigaTIME processing, and
  mask extraction.
- `onchip_wsi/`: on-chip mask conversion, R2/R4 mask generation, and PDO
  size-change labels.
- `dynamics/`: time-lapse image sorting, TrackMate tracking, dataset creation,
  and metadata utilities.

## Data Roots

Defaults:

```bash
export ML4CART_DATA_ROOT=../data
export ML4CART_DYNAMICS_ROOT=../dynamics_data
```

Each script also accepts explicit path arguments.

## Demo Data

Each data-specific preprocess folder has its own two-sample `demo_data/`:

- `tcga_wsi/demo_data/`: two placeholder `.svs` paths plus two TCGA-like mask folders.
- `onchip_wsi/demo_data/`: two processed on-chip mask folders plus two raw R2-style channel exports.
- `dynamics/demo_data/`: two raw `.tif` images plus tiny generated `.npz` model inputs.

See the subfolder README files for demo commands that use only these local
synthetic files.

## Quick Commands

TCGA WSI:

```bash
python -m preprocess.tcga_wsi.flatten_wsi --help
python -m preprocess.tcga_wsi.split_case_ids --help
```

On-chip WSI:

```bash
python -m preprocess.onchip_wsi.convert_masks --help
python -m preprocess.onchip_wsi.build_pdo_change_label --help
```

Dynamics:

```bash
python -m preprocess.dynamics.sort_images_by_xy --help
python -m preprocess.dynamics.create_dataset --help
```

See each subfolder README for complete step-by-step commands.
