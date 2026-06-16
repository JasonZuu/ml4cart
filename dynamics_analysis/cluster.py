import argparse
import json
import os
import re
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Metadata columns shared by both levels. Cell-level rows additionally carry a
# TRACK_ID column (added to this set at runtime via level_non_feature_columns).
NON_FEATURE_COLUMNS = {"PREFIX", "case_name", "LABEL"}
CELL_NON_FEATURE_COLUMNS = NON_FEATURE_COLUMNS | {"TRACK_ID"}

# Default output folders are chosen per level so image-level and cell-level
# results never overwrite each other. The split key (val, test_NYU285, ...)
# selects the parent <split>_results/ folder under dynamics_analysis/cluster/.
CLUSTER_OUT_ROOT = "dynamics_analysis/cluster"
DEFAULT_LEVEL_DIRS = {"image": "image_level", "cell": "cell_level"}

# Maps a split key in data_split.json to the dynamics CSV that already holds
# its per-XY image-level features. The val split lives in val_dynamics.csv;
# all three test splits share test_dynamics.csv and are separated by case_name
# at load time.
DEFAULT_SPLIT_INPUTS = {
    "val": "val_dynamics.csv",
    "test_NYU285": "test_dynamics.csv",
    "test_NYU318": "test_dynamics.csv",
    "test_NYU774": "test_dynamics.csv",
}
DEFAULT_DATA_SPLIT_JSON = "dynamics_data/data_split.json"

# Folder name used for results of a given split, e.g. test_NYU285 -> test-NYU285_results.
def split_results_dirname(split: str) -> str:
    return f"{split.replace('_', '-')}_results"

# Raw per-frame / per-track feature CSVs used to build cell-level features.
# These mirror the inputs of dynamics/extract_split_features.py.
DEFAULT_SPOT_CSV = "dynamics_data/generated/unscaled_spot_features.csv"
DEFAULT_TRACK_CSV = "dynamics_data/generated/unscaled_track_features.csv"

# Per-frame (spot) and per-track features aggregated to the cell level, plus the
# summary statistics computed for the per-frame features. These match
# dynamics/extract_split_features.py so cell-level columns line up with the
# *_<stat> columns already used by the image-level pipeline.
CELL_TS_FEATURES = [
    "PERIMETER", "CIRCULARITY", "ELLIPSE_ASPECTRATIO",
    "SOLIDITY", "SPEED", "MEAN_SQUARE_DISPLACEMENT",
]
CELL_TRACK_FEATURES = ["TRACK_DISPLACEMENT", "TRACK_STD_SPEED", "MEAN_DIRECTIONAL_CHANGE_RATE"]
CELL_TS_STATS = ["mean", "median", "std", "min", "max"]

HEATMAP_MEAN_FEATURES = [
    "PERIMETER_mean",
    "CIRCULARITY_mean",
    "ELLIPSE_ASPECTRATIO_mean",
    "SOLIDITY_mean",
    "SPEED_mean",
    "MEAN_SQUARE_DISPLACEMENT_mean",
    "TRACK_DISPLACEMENT_mean",
    "TRACK_STD_SPEED_mean",
    "MEAN_DIRECTIONAL_CHANGE_RATE_mean",
]

FEATURE_DISPLAY_NAMES = {
    "PERIMETER_mean": "Perimeter",
    "CIRCULARITY_mean": "Circularity",
    "ELLIPSE_ASPECTRATIO_mean": "Ellipse AR",
    "SOLIDITY_mean": "Solidity",
    "SPEED_mean": "Speed",
    "MEAN_SQUARE_DISPLACEMENT_mean": "MSD",
    "TRACK_DISPLACEMENT_mean": "Track displacement",
    "TRACK_STD_SPEED_mean": "Track speed SD",
    "MEAN_DIRECTIONAL_CHANGE_RATE_mean": "Direction change",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster a data split's dynamics features and create reference-style plots."
    )
    parser.add_argument(
        "--split",
        default="val",
        help=(
            "Split key in data_split.json to analyze (e.g. val, test_NYU285, "
            "test_NYU318, test_NYU774). Selects the default input CSV and the "
            "<split>_results/ output folder, and restricts rows to the cases "
            "listed for that split so data_split.json stays the source of truth."
        ),
    )
    parser.add_argument(
        "--data_split_json",
        default=DEFAULT_DATA_SPLIT_JSON,
        help="Split definition JSON used to resolve --split to a case list.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Image-level features CSV. Defaults to the CSV mapped to --split.",
    )
    parser.add_argument(
        "--level",
        choices=["image", "cell"],
        default="image",
        help=(
            "Clustering granularity. 'image' clusters one row per XY position "
            "(reads --input_csv directly). 'cell' clusters one row per tracked "
            "cell, built from --spot_csv/--track_csv and restricted to the "
            "images present in --input_csv."
        ),
    )
    parser.add_argument(
        "--spot_csv",
        default=DEFAULT_SPOT_CSV,
        help="Per-frame spot features CSV (cell-level only).",
    )
    parser.add_argument(
        "--track_csv",
        default=DEFAULT_TRACK_CSV,
        help="Per-track features CSV (cell-level only).",
    )
    parser.add_argument("--reducer", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--clusterer", choices=["kmeans", "dbscan"], default="kmeans")
    parser.add_argument("--kmeans_n_clusters", type=int, default=4)
    parser.add_argument("--dbscan_eps", type=float, default=0.5)
    parser.add_argument("--dbscan_min_samples", type=int, default=3)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output folder. Defaults to a level-specific folder under "
        "dynamics_analysis/cluster/ when omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.input_csv is None:
        if args.split not in DEFAULT_SPLIT_INPUTS:
            raise SystemExit(
                f"--split '{args.split}' has no default input CSV; pass --input_csv explicitly. "
                f"Known splits: {', '.join(sorted(DEFAULT_SPLIT_INPUTS))}."
            )
        args.input_csv = DEFAULT_SPLIT_INPUTS[args.split]
    if args.out_dir is None:
        args.out_dir = os.path.join(
            CLUSTER_OUT_ROOT, split_results_dirname(args.split), DEFAULT_LEVEL_DIRS[args.level]
        )
    return args


def load_split_cases(data_split_json: str, split: str) -> set[str]:
    """Return the set of case_name values that belong to `split`.

    data_split.json is the single source of truth for which cases sit in each
    split, so both image- and cell-level analyses restrict rows to this set.
    """
    with open(data_split_json) as fh:
        split_def = json.load(fh)
    if split not in split_def:
        available = [k for k in split_def if k != "meta"]
        raise ValueError(
            f"Split '{split}' not found in {data_split_json}. "
            f"Available: {', '.join(available)}."
        )
    cases = split_def[split]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Split '{split}' in {data_split_json} is empty or not a list of cases.")
    return set(cases)


def load_input_csv(input_csv: str, split_cases: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    missing = sorted(NON_FEATURE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")
    if split_cases is not None:
        before = len(df)
        df = df[df["case_name"].isin(split_cases)].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No rows in {input_csv} match the split's case_name set. "
                f"Check data_split.json vs. the CSV."
            )
        print(f"[INFO] Filtered {input_csv}: {before} -> {len(df)} rows (cases: {sorted(split_cases)})")
    return df


def level_non_feature_columns(level: str) -> set[str]:
    """Metadata columns to exclude from features for the given level."""
    return CELL_NON_FEATURE_COLUMNS if level == "cell" else NON_FEATURE_COLUMNS


def load_cell_features(args: argparse.Namespace, split_cases: set[str] | None = None) -> pd.DataFrame:
    """Build one row per tracked cell from the raw spot/track CSVs.

    Per-frame spot features are aggregated over the frames of each
    (PREFIX, TRACK_ID) using the same statistics as the image-level pipeline,
    and per-track features are joined in directly. Only cells belonging to the
    images present in --input_csv are kept, and case_name / LABEL are taken from
    --input_csv so the split definition stays the single source of truth.

    When split_cases is given, only cases in that set are kept before deriving
    the PREFIX filter, ensuring cell-level clustering uses the same split scope
    as image-level.
    """
    image_df = load_input_csv(args.input_csv, split_cases=split_cases)
    keep_prefixes = set(image_df["PREFIX"].unique())

    spots = pd.read_csv(args.spot_csv)
    tracks = pd.read_csv(args.track_csv)
    for name, frame, needed in (
        ("spot", spots, ["PREFIX", "TRACK_ID"] + CELL_TS_FEATURES),
        ("track", tracks, ["PREFIX", "TRACK_ID"] + CELL_TRACK_FEATURES),
    ):
        missing = [col for col in needed if col not in frame.columns]
        if missing:
            raise ValueError(f"{name} CSV is missing required column(s): {', '.join(missing)}")

    spots = spots[spots["PREFIX"].isin(keep_prefixes)]
    tracks = tracks[tracks["PREFIX"].isin(keep_prefixes)]

    agg_dict = {f: CELL_TS_STATS for f in CELL_TS_FEATURES}
    spot_cell = spots.groupby(["PREFIX", "TRACK_ID"])[CELL_TS_FEATURES].agg(agg_dict)
    spot_cell.columns = [f"{feat}_{stat}" for feat, stat in spot_cell.columns]
    spot_cell = spot_cell.reset_index()

    track_cell = tracks[["PREFIX", "TRACK_ID"] + CELL_TRACK_FEATURES].drop_duplicates(
        subset=["PREFIX", "TRACK_ID"]
    )
    # Track-level features carry no per-cell summary statistics; tag them with
    # _mean so the column names align with the image-level *_mean features that
    # downstream heatmaps and z-scores expect.
    track_cell = track_cell.rename(
        columns={feat: f"{feat}_mean" for feat in CELL_TRACK_FEATURES}
    )

    cell_df = spot_cell.merge(track_cell, on=["PREFIX", "TRACK_ID"], how="inner")

    # Attach case_name / LABEL from the split-defining image CSV.
    meta = image_df[["PREFIX", "case_name", "LABEL"]].drop_duplicates(subset=["PREFIX"])
    cell_df = cell_df.merge(meta, on="PREFIX", how="inner")

    ordered_cols = ["PREFIX", "TRACK_ID", "case_name", "LABEL"] + [
        col for col in cell_df.columns
        if col not in {"PREFIX", "TRACK_ID", "case_name", "LABEL"}
    ]
    return cell_df[ordered_cols].reset_index(drop=True)


def preprocess_features(
    df: pd.DataFrame,
    non_feature_columns: set[str] = NON_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    feature_df = df.drop(columns=list(non_feature_columns), errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number]).copy()
    if feature_df.empty:
        raise ValueError("No numeric dynamics feature columns found after excluding metadata.")

    missing_counts = {
        col: int(count)
        for col, count in feature_df.isna().sum().items()
        if int(count) > 0
    }

    # The *_std track columns can be NaN when an XY position has only one
    # tracked cell, because standard deviation is undefined for a singleton.
    medians = feature_df.median(numeric_only=True).fillna(0.0)
    feature_df = feature_df.fillna(medians)

    variances = feature_df.var(axis=0, ddof=0)
    zero_variance_cols = variances[np.isclose(variances, 0.0)].index.tolist()
    if zero_variance_cols:
        feature_df = feature_df.drop(columns=zero_variance_cols)
    if feature_df.empty:
        raise ValueError("All numeric dynamics features were zero-variance after imputation.")

    scaler = StandardScaler()
    z_values = scaler.fit_transform(feature_df.values)
    z_df = pd.DataFrame(z_values, columns=feature_df.columns, index=df.index)
    return z_df, zero_variance_cols, missing_counts


def reduce_to_embedding(z_df: pd.DataFrame, reducer: str, seed: int) -> pd.DataFrame:
    n_samples = len(z_df)
    if n_samples < 2:
        raise ValueError("At least two samples are required for dimensionality reduction.")

    if reducer == "tsne":
        perplexity = min(30, max(5, n_samples // 3 - 1))
        perplexity = min(perplexity, max(1, n_samples - 1))
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=seed,
        )
        coords = model.fit_transform(z_df.values)
    elif reducer == "umap":
        try:
            import umap
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "umap-learn is not installed. Install it with: pip install umap-learn"
            ) from exc
        model = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n_samples - 1),
            min_dist=0.1,
            random_state=seed,
        )
        coords = model.fit_transform(z_df.values)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")

    return pd.DataFrame(coords, columns=["reducer_x", "reducer_y"], index=z_df.index)


def cluster_embedding(
    embedding_df: pd.DataFrame,
    clusterer: str,
    kmeans_n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    seed: int,
) -> np.ndarray:
    embedding = embedding_df[["reducer_x", "reducer_y"]].values

    # Visualization-first choice: clustering is performed on the 2D embedding.
    # Clustering in the original feature space may yield different assignments.
    if clusterer == "kmeans":
        if kmeans_n_clusters < 1:
            raise ValueError("--kmeans_n_clusters must be >= 1.")
        if kmeans_n_clusters > len(embedding):
            raise ValueError("--kmeans_n_clusters cannot exceed the number of samples.")
        return KMeans(
            n_clusters=kmeans_n_clusters,
            n_init="auto",
            random_state=seed,
        ).fit_predict(embedding)

    if clusterer == "dbscan":
        if dbscan_min_samples < 1:
            raise ValueError("--dbscan_min_samples must be >= 1.")
        scaled_embedding = MinMaxScaler().fit_transform(embedding)
        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(
            scaled_embedding
        )
        if np.all(labels == -1):
            print("[WARN] DBSCAN assigned all samples to Noise; outputs will contain only a Noise cluster.")
        return labels

    raise ValueError(f"Unsupported clusterer: {clusterer}")


def format_cluster_labels(raw_labels: Iterable[int]) -> pd.Series:
    labels = ["Noise" if int(label) == -1 else f"Cluster {int(label)}" for label in raw_labels]
    return pd.Series(labels, name="cluster")


def cluster_sort_key(cluster: str) -> tuple[int, int | str]:
    if cluster == "Noise":
        return (1, 999999)
    match = re.search(r"-?\d+", str(cluster))
    if match:
        return (0, int(match.group(0)))
    return (0, str(cluster))


def get_cluster_order(cluster_labels: pd.Series) -> list[str]:
    return sorted(cluster_labels.unique().tolist(), key=cluster_sort_key)


def make_cluster_palette(cluster_order: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10" if len(cluster_order) <= 10 else "tab20")
    palette = {}
    for idx, cluster in enumerate(cluster_order):
        palette[cluster] = (0.55, 0.55, 0.55, 1.0) if cluster == "Noise" else cmap(idx % cmap.N)
    return palette


def plot_embedding_scatter(
    embedding_clusters_df: pd.DataFrame,
    cluster_order: list[str],
    palette: dict[str, tuple[float, float, float, float]],
    reducer: str,
    clusterer: str,
    kmeans_n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    out_path: str,
    level: str = "image",
) -> None:
    title_bits = [level.capitalize() + "-level", reducer.upper(), clusterer.upper()]
    if clusterer == "kmeans":
        title_bits.append(f"k={kmeans_n_clusters}")
    else:
        title_bits.append(f"eps={dbscan_eps:g}, min_samples={dbscan_min_samples}")

    # Cell-level embeddings hold hundreds of points, so use smaller, lighter
    # markers than the dozens of points in an image-level embedding.
    point_size = 22 if level == "cell" else 95
    point_alpha = 0.7 if level == "cell" else 0.88

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=embedding_clusters_df,
        x="reducer_x",
        y="reducer_y",
        hue="cluster",
        hue_order=cluster_order,
        style="case_name",
        palette=palette,
        s=point_size,
        edgecolor="black",
        linewidth=0.5,
        alpha=point_alpha,
        ax=ax,
    )
    ax.set_title(" / ".join(title_bits), fontsize=14)
    ax.set_xlabel(f"{reducer.upper()} 1")
    ax.set_ylabel(f"{reducer.upper()} 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_heatmap(
    cluster_feature_zscore: pd.DataFrame,
    cluster_order: list[str],
    out_path: str,
) -> None:
    heatmap_cols = [col for col in HEATMAP_MEAN_FEATURES if col in cluster_feature_zscore.columns]
    if not heatmap_cols:
        raise ValueError("None of the requested _mean heatmap features are available after preprocessing.")

    heatmap_df = cluster_feature_zscore.loc[cluster_order, heatmap_cols].rename(
        columns=FEATURE_DISPLAY_NAMES
    )
    max_abs = float(np.nanmax(np.abs(heatmap_df.values)))
    max_abs = max(max_abs, 1e-6)

    fig_width = max(8, 0.95 * len(heatmap_df.columns))
    fig_height = max(2.4, 0.65 * len(heatmap_df.index) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "z-score"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_case_cluster_composition(
    case_percentages: pd.DataFrame,
    cluster_order: list[str],
    palette: dict[str, tuple[float, float, float, float]],
    out_path: str,
) -> None:
    plot_df = case_percentages[cluster_order]
    fig_height = max(3.0, 0.45 * len(plot_df.index) + 1.3)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    left = np.zeros(len(plot_df), dtype=float)
    y_pos = np.arange(len(plot_df))
    for cluster in cluster_order:
        values = plot_df[cluster].values
        ax.barh(
            y_pos,
            values,
            left=left,
            height=0.68,
            color=palette[cluster],
            edgecolor="black",
            linewidth=0.45,
            label=cluster,
        )
        left += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cluster percentage (%)")
    ax.set_ylabel("")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=min(4, len(cluster_order)), frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_case_tables(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    cluster_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_order = pd.Index(pd.unique(df["case_name"]), name="case_name")
    counts = pd.crosstab(df["case_name"], cluster_labels)
    counts = counts.reindex(index=case_order, columns=cluster_order, fill_value=0)
    percentages = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    return counts, percentages


def compute_cluster_summary(
    df: pd.DataFrame,
    cluster_labels: pd.Series,
    cluster_order: list[str],
) -> pd.DataFrame:
    rows = []
    for cluster in cluster_order:
        mask = cluster_labels.values == cluster
        subset = df.loc[mask]
        cases = sorted(subset["case_name"].astype(str).unique().tolist())
        rows.append(
            {
                "cluster": cluster,
                "n_samples": int(mask.sum()),
                "n_cases_represented": len(cases),
                "cases_list": ";".join(cases),
                "mean_LABEL": float(pd.to_numeric(subset["LABEL"], errors="coerce").mean()),
                "median_LABEL": float(pd.to_numeric(subset["LABEL"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def compute_cluster_top_features(
    cluster_feature_zscore: pd.DataFrame,
    cluster_order: list[str],
) -> pd.DataFrame:
    rows = []
    for cluster in cluster_order:
        scores = cluster_feature_zscore.loc[cluster].dropna()
        positive = scores[scores > 0].sort_values(ascending=False).head(5)
        negative = scores[scores < 0].sort_values(ascending=True).head(5)
        selected = pd.concat([positive, negative])
        selected = selected.loc[selected.abs().sort_values(ascending=False).index]
        for feature, mean_zscore in selected.items():
            rows.append(
                {
                    "cluster": cluster,
                    "feature": feature,
                    "mean_zscore": float(mean_zscore),
                }
            )
    return pd.DataFrame(rows, columns=["cluster", "feature", "mean_zscore"])


def save_outputs(
    df: pd.DataFrame,
    z_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
    cluster_labels: pd.Series,
    args: argparse.Namespace,
) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    cluster_order = get_cluster_order(cluster_labels)
    palette = make_cluster_palette(cluster_order)

    # Cell-level rows are identified by (PREFIX, TRACK_ID); image-level by PREFIX.
    id_cols = ["PREFIX", "TRACK_ID", "case_name", "LABEL"] if args.level == "cell" \
        else ["PREFIX", "case_name", "LABEL"]
    embedding_clusters_df = df[id_cols].copy()
    embedding_clusters_df["reducer_x"] = embedding_df["reducer_x"].values
    embedding_clusters_df["reducer_y"] = embedding_df["reducer_y"].values
    embedding_clusters_df["cluster"] = cluster_labels.values

    z_with_cluster = z_df.copy()
    z_with_cluster["cluster"] = cluster_labels.values
    cluster_feature_zscore = z_with_cluster.groupby("cluster").mean().reindex(cluster_order)
    case_counts, case_percentages = compute_case_tables(df, cluster_labels, cluster_order)
    cluster_summary = compute_cluster_summary(df, cluster_labels, cluster_order)
    cluster_top_features = compute_cluster_top_features(cluster_feature_zscore, cluster_order)

    embedding_clusters_df.to_csv(os.path.join(args.out_dir, "embedding_clusters.csv"), index=False)
    cluster_feature_zscore.to_csv(
        os.path.join(args.out_dir, "cluster_feature_zscore.csv"),
        index_label="cluster",
    )
    case_counts.to_csv(os.path.join(args.out_dir, "case_cluster_counts.csv"), index_label="case_name")
    case_percentages.to_csv(
        os.path.join(args.out_dir, "case_cluster_percentages.csv"),
        index_label="case_name",
    )
    cluster_summary.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False)
    cluster_top_features.to_csv(os.path.join(args.out_dir, "cluster_top_features.csv"), index=False)

    # Save raw cell-level features with cluster assignment (added for collaborator request)
    raw_features_with_cluster = df.copy()
    raw_features_with_cluster["cluster"] = cluster_labels.values
    raw_features_with_cluster.to_csv(
        os.path.join(args.out_dir, "cell_features_raw.csv"), index=False
    )
    print(f"  Saved raw {args.level}-level features to {os.path.join(args.out_dir, 'cell_features_raw.csv')}")

    plot_embedding_scatter(
        embedding_clusters_df,
        cluster_order,
        palette,
        args.reducer,
        args.clusterer,
        args.kmeans_n_clusters,
        args.dbscan_eps,
        args.dbscan_min_samples,
        os.path.join(args.out_dir, "embedding_scatter.svg"),
        args.level,
    )
    plot_cluster_heatmap(
        cluster_feature_zscore,
        cluster_order,
        os.path.join(args.out_dir, "cluster_heatmap.svg"),
    )
    plot_case_cluster_composition(
        case_percentages,
        cluster_order,
        palette,
        os.path.join(args.out_dir, "case_cluster_composition.svg"),
    )


def run(args: argparse.Namespace) -> None:
    level_tag = f"[{args.level.upper()}-LEVEL]"
    split_cases = load_split_cases(args.data_split_json, args.split)

    if args.level == "cell":
        df = load_cell_features(args, split_cases=split_cases)
        unit, unit_plural = "cell", "cells"
    else:
        df = load_input_csv(args.input_csv, split_cases=split_cases)
        unit, unit_plural = "image", "images"

    z_df, zero_variance_cols, missing_counts = preprocess_features(
        df, level_non_feature_columns(args.level)
    )
    embedding_df = reduce_to_embedding(z_df, args.reducer, args.seed)
    raw_labels = cluster_embedding(
        embedding_df,
        args.clusterer,
        args.kmeans_n_clusters,
        args.dbscan_eps,
        args.dbscan_min_samples,
        args.seed,
    )
    cluster_labels = format_cluster_labels(raw_labels)
    save_outputs(df, z_df, embedding_df, cluster_labels, args)

    print(f"{level_tag} Split: {args.split} (from {args.data_split_json})")
    if args.level == "cell":
        print(
            f"{level_tag} Built {len(df)} {unit_plural} from "
            f"{df['PREFIX'].nunique()} images ({args.spot_csv}, {args.track_csv})"
        )
        print(f"{level_tag} Images restricted to split cases in {args.input_csv}")
    else:
        print(f"{level_tag} Loaded {len(df)} {unit_plural} from {args.input_csv}")
    print(f"{level_tag} Clustering unit: one row per {unit}")
    print(f"{level_tag} Used {z_df.shape[1]} z-scored features")
    if missing_counts:
        print(f"{level_tag} Median-imputed missing values in: {missing_counts}")
    if zero_variance_cols:
        print(f"{level_tag} Dropped zero-variance columns: {zero_variance_cols}")
    print(f"{level_tag} Saved plots and CSVs to {args.out_dir}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
