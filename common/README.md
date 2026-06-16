# common

Shared code used by the independent experiment folders.

- `paths.py`: external data-root defaults and output helpers.
- `seed.py`: reproducibility helper.
- `masks.py`: mask path resolution and grayscale loading.
- `plotting.py`: headless matplotlib setup.
- `cd8_model.py`, `cd8_training.py`: U-Net and shared CD8 segmentation loop.
- `pdochange_data.py`, `pdochange_model.py`, `pdochange_training.py`: PDO size-change dataset, ResNet classifier, and training utilities.

Import from `common.*`; do not import code from sibling experiment packages here.
