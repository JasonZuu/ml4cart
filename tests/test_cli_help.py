from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_help(module: str, *extra_args: str) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
    result = subprocess.run(
        [sys.executable, "-B", "-m", module, *extra_args, "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_public_cli_help():
    run_help("preprocess.tcga_wsi.split_case_ids")
    run_help("preprocess.onchip_wsi.convert_masks")
    run_help("preprocess.dynamics.create_dataset")
    run_help("tcga_cd8_distribution.train")
    run_help("onchip_cd8_distribution.train", "--preset", "actin_ck")
    run_help("pdo_size_change_prediction.train", "--version", "cd8_actin_ck")
    run_help("dynamics_model.train")
    run_help("dynamics_analysis.cluster")
