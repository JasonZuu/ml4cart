import os
import json
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import networkx as nx

TARGET_LABEL_NAMES = {
    "cart_response": "CART Cell response",
    "cart_infiltration": "CAR T cell infiltration",
}

TARGET_COLORS = {
    "cart_response": "#9d81b4",
    "cart_infiltration": "#e05c5c",
}

LABEL_COLUMN_MAP = {
    "cart_response": "PDO size change (%)",
    "cart_infiltration": "CAR-T Infiltration (number/area)",
}

FEATURE_DISPLAY_MAP = {
    "COL-I": "COL I",
    "COL-III": "COL III",
    "COL-IV": "COL IV",
    "HA": "HA",
    "PD-1": "PD-1",
    "LAG-3": "LAG-3",
    "antigen": "Target antigen\nexpression",
    "pdo_size": "PDO size",
}

FEATURE_COLOR_MAP = {
    "COL I": "#56a0fb",
    "COL III": "#56a0fb",
    "COL IV": "#56a0fb",
    "HA": "#56a0fb",
    "PD-1": "#ffa040",
    "LAG-3": "#ffa040",
    "Target antigen\nexpression": "#c06001",
    "PDO size": "#c06001",
}

DYNAMICS_SEQ_COLOR = "#4db84e"    # green — organoid dynamics features
DYNAMICS_TRACK_COLOR = "#2b7a78"  # teal — cell track motility features


def _fmt_seq_name(raw: str) -> str:
    """Format a seq feature name: 'MEAN_SQUARE_DISPLACEMENT' → 'Organoid displacement (MSD)'"""
    name = raw.replace("_", " ").lower()
    replacements = {
        "ellipse aspectratio": "aspect ratio",
        "mean square displacement": "displacement (MSD)",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return "Organoid " + name


def _fmt_track_name(raw: str) -> str:
    """Format a track feature name: 'TRACK_STD_SPEED' → 'Track speed variation'"""
    name = raw
    has_prefix = name.upper().startswith("TRACK_")
    if has_prefix:
        name = name[6:]
    name = name.replace("_", " ").lower()
    replacements = {
        "std speed": "speed variation",
        "mean directional change rate": "Directional change",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    if has_prefix:
        name = "Track " + name
    return name[0].upper() + name[1:] if name else name


def _normalize_case_name(case_name):
    raw = str(case_name).strip()
    m = re.match(r"^CART\d+_(.+)$", raw)
    if m:
        return f"CART_{m.group(1)}"
    return raw


def _load_cart_csv_label_map(cart_csv_path, label_col):
    df = pd.read_csv(cart_csv_path)
    df["Case Name"] = df["Case Name"].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    return {_normalize_case_name(c): float(v) for c, v in zip(df["Case Name"], df[label_col])}


def _load_dynamics_case_features(seq_path, track_path, split_annotation_path, target_split):
    """Load seq and track dynamics features, averaged per case.

    Returns a DataFrame indexed by case name (e.g. 'CART3_NCI2') with columns
    for each seq feature (prefixed 'SEQ_') and each track feature (prefixed 'TRACK_').
    """
    from dynamics_model.dataset.load_data import subset_split_by_case
    from dynamics_model.config import features as _seq_names, track_features as _track_names

    split = subset_split_by_case(seq_path, track_path, split_annotation_path)
    part = split[target_split]

    X_seq = np.asarray(part["X_seq"])       # [N, T, F_seq]
    X_track = np.asarray(part["X_track"])   # [N, F_track]
    case_names = list(part["case_names"])   # [N]

    # Average seq over the time dimension → [N, F_seq]
    X_seq_avg = X_seq.mean(axis=1)

    seq_cols = [_fmt_seq_name(n) for n in list(_seq_names)[: X_seq_avg.shape[1]]]
    track_cols = [_fmt_track_name(n) for n in list(_track_names)[: X_track.shape[1]]]

    df = pd.DataFrame(
        np.hstack([X_seq_avg, X_track]),
        columns=seq_cols + track_cols,
    )
    df["_case"] = case_names
    case_df = df.groupby("_case").mean()
    return case_df, set(seq_cols), set(track_cols)


def _load_metadata_feature_matrix(split_path, target_split, external_label_map=None):
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    splits = payload.get("splits", payload)
    target_ids = [str(x) for x in (splits.get(target_split, []) or [])]
    meta = payload.get("meta", {}) or {}
    if external_label_map is not None:
        label_map = external_label_map
    else:
        label_map = meta.get("size_change_by_case", {}) or {}
    if len(target_ids) == 0:
        raise ValueError(f"No case IDs found in split '{target_split}'.")
    if len(label_map) == 0:
        raise ValueError("Label map is missing or empty.")

    feature_maps = {}
    for k, v in meta.items():
        if k == "size_change_by_case" or not isinstance(v, dict):
            continue
        numeric_map = {}
        for case_id, val in v.items():
            try:
                numeric_map[str(case_id)] = float(val)
            except Exception:
                continue
        if len(numeric_map) > 0:
            feature_maps[k] = numeric_map
    if len(feature_maps) == 0:
        raise ValueError("No usable numeric metadata feature maps found in split json meta.")

    rows = []
    y = []
    loaded_case_ids = []
    for case_id in target_ids:
        lookup_key = _normalize_case_name(case_id) if external_label_map is not None else case_id
        if lookup_key not in label_map:
            continue
        try:
            label_val = float(label_map[lookup_key])
        except Exception:
            continue
        row = {}
        for feat_name, fmap in feature_maps.items():
            val = fmap.get(case_id, np.nan)
            try:
                row[feat_name] = float(val)
            except Exception:
                row[feat_name] = np.nan
        rows.append(row)
        y.append(label_val)
        loaded_case_ids.append(case_id)
    if len(rows) == 0:
        raise ValueError(f"No usable samples with labels from split '{target_split}'.")

    X_df = pd.DataFrame(rows)
    for col in X_df.columns:
        global_vals = np.array(list(feature_maps[col].values()), dtype=float)
        med = float(np.nanmedian(global_vals)) if global_vals.size > 0 else float("nan")
        col_vals = []
        for v in X_df[col].tolist():
            try:
                col_vals.append(float(v))
            except Exception:
                col_vals.append(np.nan)
        col_arr = np.array(col_vals, dtype=float)
        if np.isnan(med):
            med = float(np.nanmedian(col_arr)) if col_arr.size > 0 else float("nan")
        if np.isnan(med):
            med = 0.0
        X_df[col] = np.where(np.isnan(col_arr), med, col_arr)
    X_df = X_df.rename(columns={k: v for k, v in FEATURE_DISPLAY_MAP.items() if k in X_df.columns})
    return X_df, np.asarray(y, dtype=float), loaded_case_ids


def compute_ranked_spearman(X_df, y):
    vals = {}
    for feat in X_df.columns:
        r, _ = stats.spearmanr(X_df[feat].values, y)
        vals[feat] = 0.0 if np.isnan(r) else float(r)
    return pd.Series(vals).sort_values(key=lambda v: np.abs(v), ascending=False)


def build_rank_color_map(ranked_series, seq_feats=None, track_feats=None):
    seq_feats = set(seq_feats or [])
    track_feats = set(track_feats or [])
    out = {}
    fallback_features = []
    for feat in ranked_series.index:
        if feat in FEATURE_COLOR_MAP:
            out[feat] = FEATURE_COLOR_MAP[feat]
        elif feat in seq_feats:
            out[feat] = DYNAMICS_SEQ_COLOR
        elif feat in track_feats:
            out[feat] = DYNAMICS_TRACK_COLOR
        else:
            fallback_features.append(feat)
    n = len(fallback_features)
    cmap = plt.get_cmap("tab20") if n <= 20 else plt.get_cmap("hsv")
    for j, feat in enumerate(fallback_features):
        out[feat] = cmap(j if n <= 20 else j / max(1, n - 1))
    return out


def save_pairwise_spearman_csv(X_df, y, label_name, csv_save_path):
    all_df = X_df.copy()
    all_df[label_name] = y
    corr = all_df.corr(method="spearman").fillna(0.0)
    csv_dir = os.path.dirname(csv_save_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    corr.to_csv(csv_save_path)
    return corr


def plot_feature_correlation_network(
    X_df,
    y,
    ranked_spearman,
    label_name="CAR T Infiltration Effect",
    fl_min_abs=0.1,
    ff_min_abs=0.2,
    fig_save_path="plot/cart-response_graph.svg",
    seq_feats=None,
    track_feats=None,
):
    fl_corrs = ranked_spearman.to_dict()
    ff_corr = X_df.corr(method="spearman").fillna(0.0)
    G = nx.Graph()
    G.add_node(label_name, node_type="label")
    for feat in X_df.columns:
        G.add_node(feat, node_type="feature")
        w = abs(fl_corrs[feat])
        if w >= fl_min_abs:
            G.add_edge(label_name, feat, weight=w, edge_type="feature_label")
    feat_list = list(X_df.columns)
    for i in range(len(feat_list)):
        for j in range(i + 1, len(feat_list)):
            f1 = feat_list[i]
            f2 = feat_list[j]
            w = abs(float(ff_corr.loc[f1, f2]))
            if w >= ff_min_abs:
                G.add_edge(f1, f2, weight=w, edge_type="feature_feature")

    if G.number_of_edges() == 0:
        raise ValueError("No edges pass thresholds. Lower --fl_min_abs or --ff_min_abs.")

    feature_color_map = build_rank_color_map(ranked_spearman, seq_feats=seq_feats, track_feats=track_feats)
    pos_init = {label_name: np.array([0.0, 0.0], dtype=float)}
    n_all = max(1, G.number_of_nodes())
    k = 1.5 / np.sqrt(n_all)
    pos = nx.spring_layout(
        G,
        seed=0,
        weight="weight",
        k=k,
        iterations=600,
        pos=pos_init,
        fixed=[label_name],
    )
    pos[label_name] = np.array([0.0, 0.0], dtype=float)

    node_sizes = []
    node_colors = []
    for node in G.nodes():
        if node == label_name:
            node_sizes.append(3000)
            node_colors.append(TARGET_COLOR)
        else:
            node_sizes.append(3000)
            node_colors.append(feature_color_map[node])

    fl_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("edge_type") == "feature_label"]
    ff_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("edge_type") == "feature_feature"]
    fl_widths = [1.0 + 3.0 * abs(G.edges[e]["weight"]) for e in fl_edges]
    ff_widths = [0.5 + 2.0 * abs(G.edges[e]["weight"]) for e in ff_edges]

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=ff_edges, width=ff_widths, alpha=0.4, edge_color="grey", style="dashed", ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=fl_edges, width=fl_widths, alpha=0.8, edge_color="black", ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=20, ax=ax)

    # Legend
    legend_entries = [
        mpatches.Patch(facecolor=TARGET_COLOR, edgecolor="black", label=label_name),
        mpatches.Patch(facecolor="#56a0fb", label="ECM components"),
        mpatches.Patch(facecolor="#ffa040", label="Immune checkpoints"),
        mpatches.Patch(facecolor="#c06001", label="Antigen / tumor size"),
    ]
    if seq_feats:
        legend_entries.append(mpatches.Patch(facecolor=DYNAMICS_SEQ_COLOR, label="Organoid morphology"))
    if track_feats:
        legend_entries.append(mpatches.Patch(facecolor=DYNAMICS_TRACK_COLOR, label="Cell track motility"))
    ax.legend(handles=legend_entries, loc="upper right", fontsize=12, framealpha=0.85)

    coords = np.array([pos[n] for n in G.nodes()], dtype=float)
    if coords.size > 0:
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        x_span = float(x_max - x_min)
        y_span = float(y_max - y_min)
        x_pad = max(0.15, 0.12 * x_span)
        y_pad = max(0.15, 0.12 * y_span)
        ax.set_xlim(float(x_min - x_pad), float(x_max + x_pad))
        ax.set_ylim(float(y_min - y_pad), float(y_max + y_pad))
    ax.set_aspect("auto")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig_dir = os.path.dirname(fig_save_path)
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(fig_save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return G


def default_csv_path(fig_path):
    root, _ = os.path.splitext(fig_path)
    return f"{root}.csv"


def main(split_path, target_split, target, cart_csv_path, use_dynamics, seq_path, track_path,
         fig_path, csv_path, fl_min_abs, ff_min_abs):
    global TARGET_COLOR
    TARGET_COLOR = TARGET_COLORS[target]
    label_name = TARGET_LABEL_NAMES[target]

    external_label_map = None
    if target == "cart_infiltration":
        label_col = LABEL_COLUMN_MAP["cart_infiltration"]
        external_label_map = _load_cart_csv_label_map(cart_csv_path, label_col)

    X_df, y, case_ids = _load_metadata_feature_matrix(split_path, target_split, external_label_map)

    seq_feats: set = set()
    track_feats: set = set()
    if use_dynamics:
        dyn_df, seq_feats, track_feats = _load_dynamics_case_features(seq_path, track_path, split_path, target_split)
        # Align dynamics rows to the same case order as X_df
        dyn_rows = []
        for cid in case_ids:
            if cid in dyn_df.index:
                dyn_rows.append(dyn_df.loc[cid])
            else:
                dyn_rows.append(pd.Series(np.nan, index=dyn_df.columns))
        dyn_aligned = pd.DataFrame(dyn_rows).reset_index(drop=True)
        # Impute any NaN with column median
        for col in dyn_aligned.columns:
            med = float(np.nanmedian(dyn_aligned[col].values))
            dyn_aligned[col] = dyn_aligned[col].fillna(med if not np.isnan(med) else 0.0)
        X_df = pd.concat([X_df.reset_index(drop=True), dyn_aligned], axis=1)
        print(f"[INFO] Added {len(seq_feats) + len(track_feats)} dynamics features (seq + track).")

    ranked_spearman = compute_ranked_spearman(X_df, y)
    plot_feature_correlation_network(
        X_df=X_df,
        y=y,
        ranked_spearman=ranked_spearman,
        label_name=label_name,
        fl_min_abs=fl_min_abs,
        ff_min_abs=ff_min_abs,
        fig_save_path=fig_path,
        seq_feats=seq_feats,
        track_feats=track_feats,
    )
    save_pairwise_spearman_csv(X_df, y, label_name, csv_path)
    print(f"[INFO] Saved plot to {fig_path}")
    print(f"[INFO] Saved csv to {csv_path}")


if __name__ == "__main__":
    from dynamics_model.config import SEQ_DATASET_PATH, TRACK_DATASET_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", default="dynamics_data/data_split.json")
    parser.add_argument("--target_split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--target",
        default="cart_response",
        choices=["cart_response", "cart_infiltration"],
        help="Which clinical outcome to correlate against.",
    )
    parser.add_argument("--cart_csv_path", default="dynamics_data/CART_chip.csv",
                        help="Path to CART_chip.csv (required when --target cart_infiltration).")
    parser.add_argument("--use_dynamics", action=argparse.BooleanOptionalAction, default=False,
                        help="Include CAR-T cell dynamics (seq + track) features in the graph.")
    parser.add_argument("--seq_path", default=SEQ_DATASET_PATH,
                        help="Path to trajectory .npz dataset.")
    parser.add_argument("--track_path", default=TRACK_DATASET_PATH,
                        help="Path to track .npz dataset.")
    parser.add_argument("--fig_path", default=None,
                        help="Output SVG path. Defaults to dynamics_tme-plot/plot/<target>_graph.svg.")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--fl_min_abs", type=float, default=0.0)
    parser.add_argument("--ff_min_abs", type=float, default=0.0)
    args = parser.parse_args()
    fig_path = args.fig_path or f"dynamics_spearman/plot/{args.target.replace('_', '-')}{'_dyn' if args.use_dynamics else ''}_graph.svg"
    csv_path = args.csv_path or f"dynamics_spearman/plot/{args.target.replace('_', '-')}{'_dyn' if args.use_dynamics else ''}_graph.csv"
    main(
        split_path=args.split_path,
        target_split=args.target_split,
        target=args.target,
        cart_csv_path=args.cart_csv_path,
        use_dynamics=args.use_dynamics,
        seq_path=args.seq_path,
        track_path=args.track_path,
        fig_path=fig_path,
        csv_path=csv_path,
        fl_min_abs=args.fl_min_abs,
        ff_min_abs=args.ff_min_abs,
    )
