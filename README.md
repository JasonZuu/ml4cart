# ML4CAR-T

`ML4CAR-T` is the compact, runnable version of the project. It keeps the
original research workspace untouched and reorganizes the core experiments into
independent modules with shared utilities in `common/`.

## Purpose

The project supports three data streams and their downstream experiments:

- TCGA WSI preprocessing and a CD8 distribution prediction model.
- On-chip WSI preprocessing, CD8 distribution prediction, local distribution
  analysis, and PDO size-change prediction.
- Time-lapse dynamics preprocessing, dynamics response prediction, and dynamics
  feature analyses.

`wsi_survival_analysis` is intentionally not included in this compact package.

## Install

Use a Python environment with GPU-enabled PyTorch if training on CUDA. The
current local smoke tests were run with the `cart` conda environment.

```bash
cd /home/jasonz/Code/ml4cart-private/ml4cart
pip install -r requirements.txt
```

The single `requirements.txt` includes core ML dependencies plus WSI and
TrackMate-related dependencies. Some WSI tools may still need system libraries
such as OpenSlide to be available on the machine.

## Data Layout

Real data is kept outside this package. Defaults resolve relative to the parent
repository:

```bash
export ML4CART_DATA_ROOT=../data
export ML4CART_DYNAMICS_ROOT=../dynamics_data
```

The default paths are:

- `../data/TCGA_Data/TCGA_WSI_masks`
- `../data/TCGA_Data/data_split.json`
- `../data/On-chip_Data/data_split.json`
- `../data/On-chip_Data/pdo_change_label.json`
- `../dynamics_data/generated/trajectory_dataset_100.npz`
- `../dynamics_data/generated/track_dataset.npz`
- `../dynamics_data/data_split.json`

Every training script also accepts explicit input paths, so these environment
variables are conveniences rather than hard requirements.

## Demo Data

Every experiment/preprocess folder contains a tiny `demo_data/` directory with
two synthetic samples. These files are meant only to demonstrate the software
interfaces and directory conventions; they are not biologically meaningful.

Regenerate all demo files at any time:

```bash
python scripts/create_demo_data.py
```

The folder READMEs describe each demo layout and include copy-paste commands.
The most useful quick checks are:

```bash
python -m tcga_cd8_distribution.train \
  --masks-dir tcga_cd8_distribution/demo_data/masks \
  --split-json tcga_cd8_distribution/demo_data/data_split.json \
  --output-dir /tmp/ml4cart_demo_tcga_cd8 \
  --image-size 32 --base-channels 2 --epochs 1 --batch-size 2 --device cpu

python -m onchip_cd8_distribution.train \
  --preset actin_ck \
  --masks-dir onchip_cd8_distribution/demo_data/On-chip_Data \
  --split-json onchip_cd8_distribution/demo_data/On-chip_Data/data_split.json \
  --output-dir /tmp/ml4cart_demo_onchip_cd8 \
  --image-size 32 --base-channels 2 --epochs 1 --batch-size 2 --device cpu

python -m pdo_size_change_prediction.train \
  --version actin_ck \
  --masks-dir pdo_size_change_prediction/demo_data/On-chip_Data \
  --split-json pdo_size_change_prediction/demo_data/On-chip_Data/data_split.json \
  --label-json pdo_size_change_prediction/demo_data/On-chip_Data/pdo_change_label.json \
  --output-dir /tmp/ml4cart_demo_pdo \
  --image-size 64 --hidden-dim 8 --epochs 1 --batch-size 2 --device cpu

python -m dynamics_model.train \
  --seq-path dynamics_model/demo_data/generated/trajectory_dataset_100.npz \
  --track-path dynamics_model/demo_data/generated/track_dataset.npz \
  --split-json dynamics_model/demo_data/data_split.json \
  --output-dir /tmp/ml4cart_demo_dynamics \
  --training-method ce --use-class-weight false --max-epochs 1 \
  --batch-size 2 --hidden-size 4 --fusion-size 4 --use-wandb false
```

## Main Workflows

### TCGA WSI CD8 distribution

```bash
python -m tcga_cd8_distribution.train \
  --masks-dir ../data/TCGA_Data/TCGA_WSI_masks \
  --split-json ../data/TCGA_Data/data_split.json \
  --output-dir tcga_cd8_distribution/results \
  --epochs 60 \
  --batch-size 4 \
  --device cuda
```

### On-chip WSI CD8 distribution

```bash
python -m onchip_cd8_distribution.train \
  --preset actin_ck \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --output-dir onchip_cd8_distribution/results/actin_ck \
  --epochs 60 \
  --batch-size 4 \
  --device cuda
```

### On-chip PDO size-change prediction

```bash
python -m pdo_size_change_prediction.train \
  --version cd8_actin_ck \
  --masks-dir ../data/On-chip_Data \
  --split-json ../data/On-chip_Data/data_split.json \
  --label-json ../data/On-chip_Data/pdo_change_label.json \
  --output-dir pdo_size_change_prediction/results \
  --epochs 100 \
  --batch-size 8 \
  --device cuda
```

### Dynamics response model

```bash
python -m dynamics_model.train \
  --seq-path ../dynamics_data/generated/trajectory_dataset_100.npz \
  --track-path ../dynamics_data/generated/track_dataset.npz \
  --split-json ../dynamics_data/data_split.json \
  --output-dir dynamics_model/results \
  --training-method focal \
  --use-wandb false
```

### Preprocessing and analyses

```bash
python -m preprocess.tcga_wsi.split_case_ids --help
python -m preprocess.onchip_wsi.convert_masks --help
python -m preprocess.dynamics.create_dataset --help
python -m dynamics_analysis.cluster --help
python -m onchip_distribution_analysis.analyze_raw_mask_16_tiles --help
```

Each experiment folder has its own README with purpose, inputs, detailed run
commands, outputs, and smoke-test notes.

## Tests

The test suite uses synthetic fixtures and does not require real data:

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ml4cart-mpl-cache \
  /home/jasonz/miniconda3/bin/conda run -n cart python -m pytest -q -p no:cacheprovider
```

The smoke suite covers CLI loading, TCGA CD8 training, on-chip CD8 training,
PDO size-change training, dynamics training, dynamics clustering, and on-chip
raw mask analysis.
