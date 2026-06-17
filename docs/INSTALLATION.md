# Installation Guide

## Version

This guide applies to `ML4CAR-T` version `0.1.0`.

## Tested Environment

- Operating system: Linux workstation.
- Programming language: Python `>=3.10`.
- Validated environment: conda environment `cart`.
- GPU: optional for full training; CPU is sufficient for smoke tests and all
  included demo-data runs.

## Install With Conda

```bash
cd /home/jasonz/Code/ml4cart-private/ml4cart
conda create -n cart python=3.10
conda activate cart
pip install -r requirements.txt
```

If the `cart` environment already exists, activate it and reinstall only when
dependencies have changed:

```bash
conda activate cart
pip install -r requirements.txt
```

## External System Requirements

Most model demos and tests require only the Python dependencies in
`requirements.txt`. The following optional workflows need additional external
software or system libraries:

- TCGA WSI GigaTIME processing: OpenSlide shared library plus
  `openslide-python`.
- Dynamics TrackMate preprocessing: Fiji/ImageJ, Java, and the TrackMate
  runtime configured in `dynamics_model/config.py`.
- Full-size model training: CUDA-capable GPU is recommended, but not required
  for the included demo runs.

## Typical Install Time

On a current Linux workstation with an existing conda installation, the Python
dependency install typically takes 5-15 minutes. Installing GPU-enabled PyTorch,
OpenSlide system packages, or Fiji/ImageJ may take longer and depends on local
network and system package managers.

## Verify Installation

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/ml4cart-mpl-cache \
  python -m pytest -q -p no:cacheprovider
```

Expected result:

```text
7 passed
```
