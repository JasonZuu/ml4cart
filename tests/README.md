# tests

Synthetic smoke tests for the compact project. They do not read real mounted
data and are intended to verify package imports, CLI help, tiny one-epoch
training runs, and minimal analysis outputs.

Run from `ml4cart/`:

```bash
/home/jasonz/miniconda3/bin/conda run -n cart python -m pytest -q
```
