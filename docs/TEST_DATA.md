# Test And Demo Data

All included `demo_data/` folders are synthetic and contain two samples. They
are designed to demonstrate input formats, command-line interfaces, and output
generation. They are not intended for biological interpretation.

## Included Demo Data

- `preprocess/tcga_wsi/demo_data/`: two placeholder `.svs` files and two
  TCGA-like mask folders.
- `preprocess/onchip_wsi/demo_data/`: two processed on-chip mask folders and
  two R2-style raw channel-export folders.
- `preprocess/dynamics/demo_data/`: two raw `.tif` images and tiny generated
  dynamics `.npz` files.
- `tcga_cd8_distribution/demo_data/`: two TCGA-like mask samples and split JSON.
- `onchip_cd8_distribution/demo_data/`: two on-chip mask samples and split JSON.
- `pdo_size_change_prediction/demo_data/`: two on-chip mask samples, split JSON,
  and PDO size-change labels.
- `dynamics_model/demo_data/`: two synthetic tracks in `.npz` files, split JSON,
  labels, and metadata.
- `dynamics_analysis/demo_data/`: two image-level dynamics feature rows.
- `onchip_distribution_analysis/demo_data/`: two on-chip mask samples for local
  distribution analysis.

## Reproducibility

The demo data can be regenerated deterministically:

```bash
python scripts/create_demo_data.py
```

Training commands in `docs/DEMO.md` use explicit `--seed 1` when the entrypoint
supports it. Outputs are written to `/tmp/ml4cart_demo_*` by default so repeated
runs do not modify the repository.

## External Dependencies

The included demo-data runs do not require real TCGA WSI, real on-chip WSI, or
real dynamics data. Heavy external dependencies are only needed for full
preprocessing on real data:

- OpenSlide shared library for TCGA WSI GigaTIME processing.
- Fiji/ImageJ and Java for TrackMate-based dynamics tracking.
- CUDA GPU is recommended for full training but not required for demo runs.
