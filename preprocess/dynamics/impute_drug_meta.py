"""
K-means-based meta feature imputation for Drug cases in data_split.json.

Clusters all non-Drug cases by their size_change_by_case value, computes
per-cluster mean for each meta feature, then assigns each Drug case to its
nearest cluster based on its own size_change_by_case value.

Usage (from repo root):
    python dynamics/preprocess/impute_drug_meta.py --n_clusters 3
"""

import argparse
import json
import numpy as np
from sklearn.cluster import KMeans

JSON_PATH = "/mnt/e/CART_Data/data_split.json"

META_FEATURES = ["pdo_size", "antigen", "COL-I", "COL-III", "COL-IV", "HA", "PD-1", "LAG-3"]

DRUG_CASES = [
    "Drug_1.1", "Drug_1.2", "Drug_1.3",
    "Drug_2.1", "Drug_2.2", "Drug_2.3",
    "Drug_3.1", "Drug_3.2", "Drug_3.3",
]


def main(n_clusters: int = 3):
    with open(JSON_PATH) as f:
        d = json.load(f)

    meta = d["meta"]
    size_map = meta["size_change_by_case"]

    # Collect non-Drug cases that have size_change AND all 8 meta features
    base_cases = [c for c in size_map if not c.startswith("Drug_")]
    valid_cases = [c for c in base_cases if all(c in meta[feat] for feat in META_FEATURES)]
    excluded = set(base_cases) - set(valid_cases)
    if excluded:
        print(f"  [INFO] Cases excluded (missing ≥1 meta feature): {sorted(excluded)}")

    print(f"\n  Using {len(valid_cases)} cases for clustering with k={n_clusters}")

    # (n, 1) array of size_change values for clustering
    sizes = np.array([[size_map[c]] for c in valid_cases], dtype=np.float64)
    # (n, 8) array of meta feature values
    feat_matrix = np.array(
        [[meta[feat][c] for feat in META_FEATURES] for c in valid_cases],
        dtype=np.float64,
    )

    # K-means on 1D size_change
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(sizes)

    # Sort clusters by centroid value (ascending) for readable output
    order = np.argsort(km.cluster_centers_[:, 0])
    centroid_sorted = km.cluster_centers_[order, 0]

    print("\n  Clusters (sorted by size_change centroid):")
    cluster_means: dict[int, np.ndarray] = {}
    for rank, orig_k in enumerate(order):
        mask = labels == orig_k
        means = feat_matrix[mask].mean(axis=0)
        cluster_means[orig_k] = means
        members = [valid_cases[i] for i in np.where(mask)[0]]
        feat_str = ", ".join(f"{feat}={means[i]:.1f}" for i, feat in enumerate(META_FEATURES))
        print(f"    Cluster {orig_k} (centroid={centroid_sorted[rank]:+.1f}, n={mask.sum()})")
        print(f"      cases : {members}")
        print(f"      means : {feat_str}")

    # Assign each Drug case to its nearest cluster and update meta
    drug_sizes = np.array([[size_map[c]] for c in DRUG_CASES], dtype=np.float64)
    drug_labels = km.predict(drug_sizes)

    print("\n  Drug case assignments:")
    for case, cluster_id in zip(DRUG_CASES, drug_labels):
        means = cluster_means[cluster_id]
        for feat_idx, feat in enumerate(META_FEATURES):
            meta[feat][case] = float(means[feat_idx])
        print(f"    {case}  size_change={size_map[case]:+.1f}  → cluster {cluster_id}"
              f"  (centroid={km.cluster_centers_[cluster_id, 0]:+.1f})")

    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=4)
    print(f"\n  Updated: {JSON_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-means meta imputation for Drug cases")
    parser.add_argument("--n_clusters", type=int, default=3,
                        help="Number of k-means clusters (default: 3)")
    args = parser.parse_args()
    main(n_clusters=args.n_clusters)
