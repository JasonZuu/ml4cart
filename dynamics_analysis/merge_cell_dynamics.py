"""Merge cell-level dynamics features from three test patients into one CSV
with human-readable PREFIX labels (NYUxxx_drug format).
"""
import pandas as pd
from pathlib import Path

# Mapping from case_name to human-readable format
CASE_MAPPING = {
    "Drug_1.1": ("NYU285", "IgG"),
    "Drug_1.2": ("NYU285", "iAREG"),
    "Drug_1.3": ("NYU285", "FAP"),
    "Drug_2.1": ("NYU318", "IgG"),
    "Drug_2.2": ("NYU318", "iAREG"),
    "Drug_2.3": ("NYU318", "FAP"),
    "Drug_3.1": ("NYU774", "IgG"),
    "Drug_3.2": ("NYU774", "iAREG"),
    "Drug_3.3": ("NYU774", "FAP"),
}

def load_and_merge():
    files = [
        "dynamics_analysis/cluster/test-NYU285_results/cell_level/cell_features_raw.csv",
        "dynamics_analysis/cluster/test-NYU318_results/cell_level/cell_features_raw.csv",
        "dynamics_analysis/cluster/test-NYU774_results/cell_level/cell_features_raw.csv",
    ]

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(merged)} cells from {len(files)} files.")
    return merged


def reformat_prefix(df):
    """Replace PREFIX with human-readable format: NYUxxx_drug_XYn"""
    def make_new_prefix(row):
        case = row["case_name"]
        old_prefix = row["PREFIX"]

        # Extract XY position from old PREFIX (e.g., "Drug_1.1_XY1" -> "XY1")
        xy_part = old_prefix.split("_")[-1]  # last component after splitting by "_"

        # Map case to (patient, drug)
        if case not in CASE_MAPPING:
            return old_prefix  # fallback

        patient, drug = CASE_MAPPING[case]
        return f"{patient}_{drug}_{xy_part}"

    df["PREFIX"] = df.apply(make_new_prefix, axis=1)
    return df


def main():
    merged = load_and_merge()
    merged = reformat_prefix(merged)

    # Sort by patient, drug, XY, TRACK_ID for readability
    merged = merged.sort_values(["case_name", "PREFIX", "TRACK_ID"]).reset_index(drop=True)

    out_path = Path("dynamics_analysis/cluster/test_all_patients_cell_level_dynamics.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved merged cell-level dynamics to {out_path}")
    print(f"  Total cells: {len(merged)}")
    print(f"  Patients: {merged['case_name'].nunique()} cases")
    print(f"  Columns: {len(merged.columns)} ({', '.join(merged.columns[:5])}...)")

    # Print sample of new PREFIX format
    print("\nSample of new PREFIX format:")
    for prefix in merged["PREFIX"].unique()[:6]:
        print(f"  {prefix}")


if __name__ == "__main__":
    main()
