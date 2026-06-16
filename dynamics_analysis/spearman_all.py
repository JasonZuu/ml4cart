import argparse
import os
import json
import re
import unicodedata
import math
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch


LABEL_COLUMN_MAP = {
    "cart_response_number": "PDO size change (%)",
    "cart_infiltration": "CAR-T Infiltration (number/area)",
}

OUTPUT_BASENAME_MAP = {
    "cart_response_number": "tme_cart-response_spearman_bar-plot",
    "cart_infiltration": "tme_infiltration_spearman_bar-plot",
}


def _normalize_name(s: object) -> str:
    """Normalize names for matching: Unicode normalization, lowercase, Greek letter transliteration, hyphen unification, whitespace compression."""
    if s is None:
        return ""
    x = unicodedata.normalize("NFKC", str(s)).strip().lower()

    # Greek letters and common variants
    greek = {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    }
    for k, v in greek.items():
        x = x.replace(k, v)

    # Normalize hyphens
    x = x.replace("–", "-").replace("—", "-").replace("-", "-").replace("−", "-")

    # Canonicalize aliases (common naming variations)
    x = re.sub(r"\btgf[-\s]*b\b", "tgf-beta", x)          # TGF-b -> TGF-beta
    x = re.sub(r"\bmip-1\s+beta\b", "mip-1-beta", x)      # MIP-1 beta -> MIP-1-beta
    x = re.sub(r"\bck\s*beta\s*(8-1)\b", r"ck beta \1", x)  # CK beta 8-1 variants
    x = re.sub(r"\s*-\s*", "-", x)                        # remove extra spaces around hyphens
    x = re.sub(r"\s+", " ", x).strip()
    return x

def load_immune_data(filepath: str = "data/Multimodal data.xlsx",
                     sheet_name: str = "Immune",
                     drop_rows_with_any_nan: bool = True) -> Optional[pd.DataFrame]:
    """
    Load data from an Excel worksheet and return all available features.

    Assumptions:
    - First column is the factor name (index), remaining columns are samples.

    Pipeline:
    1) Coerce all cells to numeric (non-numeric -> NaN)
    2) Clean row names (Unicode/Greek/hyphen/whitespace)
    3) Merge duplicate names by mean
    4) Optionally drop rows containing any NaN

    Returns:
        pd.DataFrame or None
    """
    try:
        df_raw = pd.read_excel(filepath, sheet_name=sheet_name, index_col=0)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        print("Make sure 'pandas' and 'openpyxl' are installed.")
        return None

    # Coerce values to numeric and drop empty rows
    df_num = df_raw.apply(pd.to_numeric, errors='coerce')
    df_num = df_num.dropna(how='all')

    df_kept = df_num.copy()
    df_kept.index = pd.Index([_normalize_name(i) for i in df_num.index])

    if df_kept.index.duplicated().any():
        dup_cnt = int(df_kept.index.duplicated().sum())
        print(f"Note: Detected {dup_cnt} duplicated rows after normalization; merging by mean.")
        df_kept = df_kept.groupby(level=0).mean()

    # Optional: drop rows containing any NaN
    if drop_rows_with_any_nan:
        before = df_kept.shape[0]
        df_kept = df_kept.dropna()
        dropped = before - df_kept.shape[0]
        if dropped > 0:
            print(f"Warning: Dropped {dropped} mapped rows due to missing data (NaN).")

    print(f"Successfully loaded. Shape: {df_kept.shape}")
    return df_kept


def plot_common_signatures_volcano(log2fc_df, p_thresh=0.05, fc_thresh=1.0, fig_save_path: Optional[str] = None):
    """
    Plots a volcano plot to identify cytokines commonly dysregulated across all patients.
    
    Uses a one-sample t-test to check if the mean log2(Fold Change) for each
    analyte is significantly different from 0.

    Args:
    log2fc_df (pd.DataFrame): DataFrame with analytes as rows, patients as columns.
                               Values are log2(Fold Change).
    p_thresh (float): Significance p-value threshold (e.g., 0.05).
    fc_thresh (float): log2(Fold Change) magnitude threshold (e.g., 1.0 for 2-fold change).

    Returns:
    plt.Axes: The Matplotlib axes object.
    pd.DataFrame: DataFrame containing the statistical results.
    """
    
    # 1. Calculate statistics
    # Perform a one-sample t-test for each row (analyte)
    # axis=1 means calculate across columns (i.e., for each row)
    t_stats, p_values = stats.ttest_1samp(log2fc_df, popmean=0, axis=1)
    
    # 2. Organize results
    results_df = pd.DataFrame({
        'mean_log2fc': log2fc_df.mean(axis=1),
        'p_value': p_values,
        '-log10_p_value': -np.log10(p_values)
    })
    results_df.index = log2fc_df.index

    # 3. Classify (Upregulated, Downregulated, Not Significant)
    results_df['significance'] = 'Not Significant'
    
    # Condition 1: Significantly Upregulated
    condition_up = (results_df['mean_log2fc'] > fc_thresh) & (results_df['p_value'] < p_thresh)
    results_df.loc[condition_up, 'significance'] = 'Upregulated'
    
    # Condition 2: Significantly Downregulated
    condition_down = (results_df['mean_log2fc'] < -fc_thresh) & (results_df['p_value'] < p_thresh)
    results_df.loc[condition_down, 'significance'] = 'Downregulated'

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    
    palette = {
        'Upregulated': 'red',
        'Downregulated': 'blue',
        'Not Significant': 'grey'
    }
    
    ax = sns.scatterplot(
        data=results_df,
        x='mean_log2fc',
        y='-log10_p_value',
        hue='significance',
        palette=palette,
        s=50,          # size of points
        alpha=0.7,     # transparency
        edgecolor='k', # edge color of points
        linewidth=0.5
    )

    # 5. Add threshold lines
    ax.axhline(y=-np.log10(p_thresh), color='grey', linestyle='--', linewidth=1)
    ax.axvline(x=fc_thresh, color='grey', linestyle='--', linewidth=1)
    ax.axvline(x=-fc_thresh, color='grey', linestyle='--', linewidth=1)

    # 6. Add labels and title
    ax.set_title('Volcano Plot for Common Signatures', fontsize=16)
    ax.set_xlabel('Mean Log2(Fold Change) across 8 Patients', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    
    # (Optional) Add labels for significant points
    # for idx, row in results_df[results_df['significance'] != 'Not Significant'].iterrows():
    #     ax.text(row['mean_log2fc'], row['-log10_p_value'], idx, fontsize=8)

    plt.legend(title='Significance')
    plt.grid(alpha=0.3)
    if fig_save_path:
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    return ax, results_df

# -------------------------------------------------------------------

def plot_patient_pca(log2fc_df, annotate_points=True, fig_save_path: Optional[str] = None):
    """
    Performs PCA and plots PC1 vs PC2 to visualize patient clustering/subtypes.

    Args:
    log2fc_df (pd.DataFrame): DataFrame with analytes as rows, patients as columns.
    annotate_points (bool): Whether to label the points with patient IDs.

    Returns:
    plt.Axes: The Matplotlib axes object.
    pd.DataFrame: DataFrame containing the PCA coordinates for each patient.
    """
    
    # 1. Prepare data
    # PCA expects (n_samples, n_features)
    # Our patients are "samples" (n=8)
    # Our analytes are "features" (n=~80)
    # Therefore, we need to transpose the DataFrame
    data_transposed = log2fc_df.T
    
    # 2. Perform PCA
    pca = PCA(n_components=2)
    pc_components = pca.fit_transform(data_transposed)
    
    # Get explained variance ratio
    explained_var = pca.explained_variance_ratio_
    pc1_var = explained_var[0] * 100
    pc2_var = explained_var[1] * 100
    
    # 3. Organize results into a DataFrame
    pca_df = pd.DataFrame(
        data=pc_components,
        columns=['PC1', 'PC2'],
        index=data_transposed.index  # Patient IDs
    )

    # 4. Plotting
    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        s=150,           # size of points
        alpha=0.8,
        edgecolor='k',
        linewidth=1
    )
    
    # 5. Add labels and title
    ax.set_title('PCA of Patient Samples', fontsize=16)
    ax.set_xlabel(f'Principal Component 1 ({pc1_var:.1f}%)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pc2_var:.1f}%)', fontsize=12)
    
    # 6. (Optional) Annotate points
    if annotate_points:
        for patient_id in pca_df.index:
            ax.text(
                pca_df.loc[patient_id, 'PC1'] + 0.1,  # slight x-offset
                pca_df.loc[patient_id, 'PC2'] + 0.1,  # slight y-offset
                patient_id,
                fontsize=9,
                ha='left' # horizontal alignment
            )
            
    plt.grid(alpha=0.3)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    if fig_save_path:
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    return ax, pca_df

# -------------------------------------------------------------------

def plot_cytokine_correlation_clustermap(log2fc_df, method='spearman', fig_save_path: Optional[str] = None):
    """
    Calculates the correlation between cytokines and visualizes 
    "co-regulated" modules using a clustermap.

    Args:
    log2fc_df (pd.DataFrame): DataFrame with analytes as rows, patients as columns.
    method (str): Correlation method ('pearson' or 'spearman').
                   Spearman (rank-based) is often more robust to outliers.

    Returns:
    sns.matrix.ClusterGrid: The Seaborn ClusterGrid object.
    pd.DataFrame: The correlation matrix.
    """
    
    # 1. Calculate correlation
    # We want to calculate analyte-analyte correlation
    # (based on their profile across the 8 patients).
    # df.corr() calculates column-column correlation.
    # So we transpose the DataFrame to make analytes the columns.
    corr_matrix = log2fc_df.T.corr(method=method)
    
    # 2. Plotting (Clustermap)
    # Clustermap is powerful because it clusters both rows and columns,
    # automatically grouping related modules together.
    # For a large 80x80 matrix, showing labels is not practical.
    # The insight comes from the colored blocks formed by clustering.
    
    g = sns.clustermap(
        corr_matrix,
        cmap='vlag',      # A good blue-white-red diverging colormap
        center=0,         # Center the colormap at 0 (no correlation)
        figsize=(12, 12),
        xticklabels=False, # Hide x-axis labels (too many)
        yticklabels=False, # Hide y-axis labels (too many)
        cbar_pos=(0.02, 0.8, 0.05, 0.18), # Adjust colorbar position
        cbar_kws={'label': f'{method.capitalize()} Correlation'}
    )
    
    g.fig.suptitle('Cytokine Correlation Clustermap', fontsize=16, y=1.02)
    if fig_save_path:
        g.fig.tight_layout()
        g.fig.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    return g, corr_matrix


def plot_cytokine_correlation_graph(
    log2fc_df: pd.DataFrame,
    method: str = "spearman",
    top_p: float = 0.03,
    min_abs_r: float = 0.0,
    layout: str = "spring",
    label_strategy: str = "all",   # 'none' | 'all' | 'top_degree'
    label_top_k: int = 15,
    fig_save_path: Optional[str] = None,
    min_node_separation = 0.2
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Build a correlation graph and visualize it, keeping only the top_p strongest |r| edges.
    Rows=features, Cols=samples.
    Returns: (G, corr_matrix, kept_edges_df)
    """
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")

    # 1) feature-feature correlation
    corr_matrix = log2fc_df.T.corr(method=method)
    feats = corr_matrix.columns.tolist()

    # 2) flatten upper triangle and rank by |r|
    edges = []
    for i in range(len(feats)):
        for j in range(i+1, len(feats)):
            r = corr_matrix.iat[i, j]
            edges.append((feats[i], feats[j], r, abs(r)))
    edges_df = (pd.DataFrame(edges, columns=["source","target","r","abs_r"])
                .sort_values("abs_r", ascending=False))

    # 3) keep top_p edges (optionally enforce min_abs_r)
    m = len(edges_df)
    k = max(1, int(math.floor(m * top_p)))
    kept = edges_df.iloc[:k].copy()
    if min_abs_r > 0:
        kept = kept[kept["abs_r"] >= min_abs_r]

    # 4) build graph
    G = nx.Graph()
    G.add_nodes_from(feats)
    for _, row in kept.iterrows():
        G.add_edge(row["source"], row["target"], weight=abs(row["r"]), abs_weight=row["abs_r"])

    # 5) layout
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spring":
        n = max(1, G.number_of_nodes())
        k_base = 1.0 / math.sqrt(n)              # NetworkX 的默认标尺
        # 把 k 放大，让图更松；多迭代几步更稳定
        pos = nx.spring_layout(
            G,
            seed=0,
            weight="abs_weight",
            k=k_base * 2.2,                      # ← 拉开距离的关键：放大 2.2 倍，按需调大/调小
            iterations=400,
            threshold=1e-4
        )
        # 用斥力把仍然过近的点推开；min_node_separation 可调（0~1，相对坐标）
        pos = _repel_positions(pos, min_sep=min_node_separation, iters=80, step=0.5)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.kamada_kawai_layout(G)

    # 6) draw (单图，不设颜色，符合你那条奇怪的审美规矩)
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [150 + 150*(degrees[n]/max_deg if max_deg else 0) for n in G.nodes()]

    plt.figure(figsize=(9,9))
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

    labels = {}
    if label_strategy == "all":
        labels = {n: n for n in G.nodes()}
    elif label_strategy == "top_degree":
        top_nodes = sorted(degrees.items(), key=lambda kv: kv[1], reverse=True)[:label_top_k]
        labels = {n: n for n,_ in top_nodes}
    if labels:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # Legend for colors: edges represent correlation magnitude (grey), nodes sized by degree
    handles = [
        Patch(color='grey', label='Edge: |correlation| (width ∝ |r|)'),
        Patch(color='tab:blue', label='Node: degree-sized'),
    ]
    plt.legend(handles=handles, loc='best')

    plt.title(f"Cytokine correlation graph (method={method}, top_p={top_p:.2f})")
    plt.axis("off")
    if fig_save_path:
        plt.tight_layout()
        plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return G, corr_matrix, kept


def _repel_positions(pos, min_sep=0.035, iters=60, step=0.5):
    nodes = list(pos.keys())
    P = np.array([pos[n] for n in nodes], dtype=float)
    for _ in range(iters):
        moved = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                d = P[j] - P[i]
                dist = np.linalg.norm(d)
                if dist < 1e-9:
                    d = np.random.randn(2); dist = np.linalg.norm(d)
                if dist < min_sep:
                    move = (min_sep - dist) * (d / dist) * step
                    P[i] -= move; P[j] += move; moved = True
        if not moved:
            break
    for k, n in enumerate(nodes):
        pos[n] = (float(P[k,0]), float(P[k,1]))
    return pos


# ---------------------------------------------------------------------------
# Spearman correlation analysis against clinical labels
# ---------------------------------------------------------------------------

def _normalize_case_name(case_name: str) -> str:
    raw = str(case_name).strip()
    m = re.match(r"^CART\d+_(.+)$", raw)
    if m:
        return f"CART_{m.group(1)}"
    return raw


def load_label_map(cart_csv_path: str, label_type: str) -> dict:
    if label_type not in LABEL_COLUMN_MAP:
        raise ValueError(f"Unsupported label_type: {label_type}")
    label_col = LABEL_COLUMN_MAP[label_type]
    df = pd.read_csv(cart_csv_path)
    if "Case Name" not in df.columns:
        raise ValueError("CART csv must contain 'Case Name' column.")
    if label_col not in df.columns:
        raise ValueError(f"CART csv missing label column: {label_col}")
    df = df[["Case Name", label_col]].copy()
    df["Case Name"] = df["Case Name"].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    label_map = {}
    for case_name, val in zip(df["Case Name"], df[label_col]):
        label_map[_normalize_case_name(case_name)] = float(val)
    return label_map


def load_tme_data_from_json(json_path: str):
    """Load all TME features from tme_feature.json.

    Returns (X_df, case_series) where X_df has cases as rows and all TME
    features as columns.  Missing values are imputed with the column-wise median.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Load every key in the JSON as a feature
    all_cases: set = set()
    feat_maps: dict = {}
    for key, val in payload.items():
        if not isinstance(val, dict):
            continue
        val_map = {}
        for case_id, v in val.items():
            try:
                val_map[str(case_id)] = float(v)
            except Exception:
                continue
        if val_map:
            feat_maps[key] = val_map
            all_cases.update(val_map.keys())

    if not feat_maps:
        raise ValueError("No TME features found in JSON.")

    cases = sorted(all_cases)
    rows = []
    for case_id in cases:
        row = {}
        for feat, fmap in feat_maps.items():
            row[feat] = fmap.get(case_id, np.nan)
        rows.append(row)

    X_df = pd.DataFrame(rows, index=cases)

    # Impute NaN with column median
    for col in X_df.columns:
        med = float(np.nanmedian(X_df[col].values))
        if np.isnan(med):
            med = 0.0
        X_df[col] = np.where(np.isnan(X_df[col].values), med, X_df[col].values)

    case_series = pd.Series(cases, name="Case Name")
    return X_df, case_series


def compute_ranked_spearman(X_df: pd.DataFrame, y: pd.Series) -> pd.Series:
    yv = y.values
    vals = {}
    for feat in X_df.columns:
        r, _ = stats.spearmanr(X_df[feat].values, yv)
        vals[feat] = 0.0 if np.isnan(r) else float(r)
    return pd.Series(vals).sort_values(key=lambda v: np.abs(v), ascending=False)


def build_rank_color_map(ranked_spearman: pd.Series) -> dict:
    n = len(ranked_spearman)
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    return {feat: colors[i] for i, feat in enumerate(ranked_spearman.index)}


def plot_ranked_spearman_bar(ranked_spearman: pd.Series, fig_save_path: str, csv_save_path: str):
    fig_h = max(4.0, 0.35 * len(ranked_spearman))
    plt.figure(figsize=(10, fig_h))
    feature_color_map = build_rank_color_map(ranked_spearman)
    colors = [feature_color_map[f] for f in ranked_spearman.index]
    ax = plt.gca()
    ax.barh(ranked_spearman.index, ranked_spearman.values, color=colors, edgecolor="black")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Ranked Spearman correlation with label")
    ax.set_xlabel("Spearman correlation")
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
    plt.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    ranked_spearman.reset_index().rename(
        columns={"index": "feature", 0: "spearman"}
    ).to_csv(csv_save_path, index=False)


def main(args):
    X_df, case_series = load_tme_data_from_json(args.json_path)
    print(f"[INFO] Loaded TME data: {X_df.shape[1]} features, {X_df.shape[0]} cases")

    out_dir = args.output_dir
    for label_type in ["cart_response_number", "cart_infiltration"]:
        label_map = load_label_map(args.cart_csv_path, label_type)
        y = case_series.map(lambda c: label_map.get(_normalize_case_name(c)))
        valid_mask = y.notna()
        X_use = X_df.loc[valid_mask.values].reset_index(drop=True)
        y_use = y.loc[valid_mask].astype(float).reset_index(drop=True)
        if X_use.empty:
            raise ValueError(f"No matched samples for {label_type}.")
        if y_use.nunique() < 2:
            raise ValueError(f"Label {label_type} has <2 unique values.")
        ranked_spearman = compute_ranked_spearman(X_use, y_use)
        basename = OUTPUT_BASENAME_MAP[label_type]
        fig_path = os.path.join(out_dir, f"{basename}.svg")
        csv_path = os.path.join(out_dir, f"{basename}.csv")
        plot_ranked_spearman_bar(ranked_spearman, fig_path, csv_path)
        print(f"[INFO] Saved plot → {fig_path}")
        print(f"[INFO] Saved csv  → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="dynamics_data/tme_feature.json")
    parser.add_argument("--cart_csv_path", default="dynamics_data/CART_chip.csv")
    parser.add_argument("--output_dir", default="dynamics_spearman/plot")
    main(parser.parse_args())
