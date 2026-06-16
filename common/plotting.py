"""Plotting setup helpers."""

from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_cache(cache_dir: str | Path = ".matplotlib-cache") -> None:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(path.resolve()))


def use_headless_matplotlib(cache_dir: str | Path = ".matplotlib-cache") -> None:
    configure_matplotlib_cache(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
