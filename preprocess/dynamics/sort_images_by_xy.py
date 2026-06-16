import os
import shutil
import re
import argparse

# Match 'XY' followed by digits anywhere in the name (case-insensitive)
_GLOBAL_XY_PATTERN = re.compile(r"(?i)xy\d+")

def parse_xy_from_filename(fname):
    """Return the XY token (e.g., 'XY1') from a filename.

    Primary rule: split by underscores, take the last segment, strip extension,
    and ensure it matches 'XY' + digits. Fallback: regex search anywhere.
    """
    base = os.path.basename(fname)
    name, _ext = os.path.splitext(base)

    # Primary rule: last underscore-separated segment
    parts = name.split("_")
    if parts:
        last = parts[-1]
        if _GLOBAL_XY_PATTERN.fullmatch(last):
            return last.upper()

    # Fallback: search anywhere
    m = _GLOBAL_XY_PATTERN.search(base)
    if m:
        return m.group(0).upper()

    return None

def sort_images_in_folder(images_folder):
    """Move images in a single folder into XY subfolders.

    Returns the number of files moved.
    """
    moved = 0
    for fname in sorted(os.listdir(images_folder)):
        fpath = os.path.join(images_folder, fname)

        if not os.path.isfile(fpath):
            continue  # skip directories

        # Only process common microscopy image types
        if not fname.lower().endswith((".tif", ".tiff")):
            continue

        xy = parse_xy_from_filename(fname)
        if not xy:
            print(f"Skipping {fname}: no XY# found")
            continue

        dest_folder = os.path.join(images_folder, xy)
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, fname)
        if os.path.exists(dest_path):
            print(f"Exists, skipping: {dest_path}")
            continue

        shutil.move(fpath, dest_path)
        moved += 1
        print(f"Moved {fname} → {dest_folder}")

    return moved

def sort_images_by_XY(base_folder):
    """Walk one level of subdirectories under base_folder and sort images."""
    total_moved = 0
    for entry in sorted(os.listdir(base_folder)):
        subdir_path = os.path.join(base_folder, entry)
        if not os.path.isdir(subdir_path):
            continue
        moved = sort_images_in_folder(subdir_path)
        total_moved += moved
        print(f"{entry}: moved {moved} file(s)")
    print(f"Total moved: {total_moved}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Put images into corresponding XY subfolders based on filename.")
    parser.add_argument("--base_folder", 
                        default="dynamics_data/20260114_8 patients_early CAR T",
                        help="Path to base folder (e.g., data/William_20250429_CART_8 patients_day 6_for AI)")
    args = parser.parse_args()

    sort_images_by_XY(args.base_folder)