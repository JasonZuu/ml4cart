"""
This script flattens nested TCGA WSI downloads into a single folder.

Why: some pipelines expect all .svs slides to live directly under data/TCGA_WSI
and to be keyed by the first token of the filename (A.svs from A.B.svs).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def first_token_svs_name(filename: str) -> str:
    """Return the normalized target WSI filename `A.svs` from an input like `A.B.svs`."""
    if not filename:
        raise ValueError("filename must be non-empty.")
    parts = filename.split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected a filename with an extension, got: {filename}")
    token = parts[0].strip()
    if not token:
        raise ValueError(f"First token is empty for filename: {filename}")
    return f"{token}.svs"


def iter_svs_files(root_dir: Path) -> list[Path]:
    """Collect all `.svs` files under `root_dir` recursively (including nested subfolders)."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    files: list[Path] = []
    for p in root_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".svs":
            files.append(p)
    files.sort()
    return files


def move_and_rename_svs(
    *,
    src: Path,
    root_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    """Move `src` into `root_dir` and rename it to the normalized `A.svs` format."""
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    target_name = first_token_svs_name(src.name)
    dst = root_dir / target_name

    # If already in the right place with the right name, do nothing.
    if src.resolve() == dst.resolve():
        return dst

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {dst} (use --overwrite to replace)")
        if not dry_run:
            dst.unlink()

    if not dry_run:
        root_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    return dst


def remove_empty_dirs(root_dir: Path) -> int:
    """Remove empty subdirectories under `root_dir` (keeps `root_dir` itself)."""
    if not root_dir.exists():
        return 0

    removed = 0
    # Bottom-up deletion: deepest directories first.
    for d in sorted((p for p in root_dir.rglob("*") if p.is_dir()), reverse=True):
        if d == root_dir:
            continue
        if any(d.iterdir()):
            continue
        d.rmdir()
        removed += 1
    return removed


def parse_args() -> argparse.Namespace:
    """Parse CLI args to control where files are flattened and how collisions are handled."""
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=Path, default=Path("data/TCGA_WSI"))
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing A.svs targets.")
    p.add_argument("--dry-run", action="store_true", help="Print actions without moving files.")
    p.add_argument(
        "--keep-empty-dirs",
        action="store_true",
        help="Do not delete empty subfolders after moving.",
    )
    return p.parse_args()


def main() -> int:
    """Flatten nested TCGA WSI folders into a single directory with normalized slide names."""
    args = parse_args()
    root_dir: Path = args.root_dir

    svs_files = iter_svs_files(root_dir)

    moved = 0
    for src in svs_files:
        dst = move_and_rename_svs(
            src=src,
            root_dir=root_dir,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        action = "WOULD MOVE" if args.dry_run else "MOVED"
        print(f"{action}: {src} -> {dst}")
        if src.resolve() != dst.resolve():
            moved += 1

    if not args.keep_empty_dirs and not args.dry_run:
        removed_dirs = remove_empty_dirs(root_dir)
        print(f"Removed empty dirs: {removed_dirs}")

    print(f"Total .svs found: {len(svs_files)} | moved/renamed: {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())