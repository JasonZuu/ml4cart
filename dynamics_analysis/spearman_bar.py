import os
import json
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from dynamics_model.config import SEQ_DATASET_PATH, TRACK_DATASET_PATH


LABEL_COLUMN_MAP = {
    "cart_response_number": "PDO size change (%)",
    "cart_infiltration": "CAR-T Infiltration (number/area)",
}


OUTPUT_BASENAME_MAP = {
    "cart_response_number": "cart-response_spearman_bar-plot",
    "cart_infiltration": "infiltration_spearman_bar-plot",
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


def load_sample_features_from_split(seq_path, track_path, split_path, splits):
    _ = seq_path, track_path
    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    split_map = payload.get("splits", payload)
    meta = payload.get("meta", {}) or {}
    label_map = meta.get("size_change_by_case", {}) or {}
    if len(label_map) == 0:
        raise ValueError("meta.size_change_by_case is missing or empty.")
    target_ids = []
    for split_name in splits:
        target_ids.extend([str(x) for x in (split_map.get(split_name, []) or [])])
    if len(target_ids) == 0:
        raise ValueError("No samples found in selected splits.")
    feature_maps = {}
    for key, val in meta.items():
        if key == "size_change_by_case" or not isinstance(val, dict):
            continue
        numeric_map = {}
        for case_id, v in val.items():
            try:
                numeric_map[str(case_id)] = float(v)
            except Exception:
                continue
        if len(numeric_map) > 0:
            feature_maps[key] = numeric_map
    if len(feature_maps) == 0:
        raise ValueError("No usable numeric metadata feature maps found in split json meta.")
    rows = []
    case_all = []
    y_all = []
    for case_id in target_ids:
        if case_id not in label_map:
            continue
        try:
            label_val = float(label_map[case_id])
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
        case_all.append(case_id)
        y_all.append(label_val)
    if len(rows) == 0:
        raise ValueError("No usable samples with labels from selected splits.")
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
    return X_df, pd.Series(case_all, name="Case Name"), pd.Series(y_all, name="y")


def compute_ranked_spearman(X_df, y):
    vals = {}
    yv = y.values
    for feat in X_df.columns:
        r, _ = stats.spearmanr(X_df[feat].values, yv)
        vals[feat] = 0.0 if np.isnan(r) else float(r)
    return pd.Series(vals).sort_values(key=lambda v: np.abs(v), ascending=False)


def build_rank_color_map(ranked_spearman):
    n = len(ranked_spearman)
    if n <= 20:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / max(1, n - 1)) for i in range(n)]
    return {feat: colors[i] for i, feat in enumerate(ranked_spearman.index)}


def plot_ranked_spearman_bar(ranked_spearman, fig_save_path, csv_save_path):
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
    ranked_spearman.reset_index().rename(columns={"index": "feature", 0: "spearman"}).to_csv(csv_save_path, index=False)


def _normalize_case_name(case_name):
    raw = str(case_name).strip()
    m = re.match(r"^CART\d+_(.+)$", raw)
    if m:
        return f"CART_{m.group(1)}"
    return raw


def load_label_map(cart_csv_path, label_type):
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


def build_output_paths(output_dir):
    return {
        "cart_response_number": {
            "fig": os.path.join(output_dir, f"{OUTPUT_BASENAME_MAP['cart_response_number']}.svg"),
            "csv": os.path.join(output_dir, f"{OUTPUT_BASENAME_MAP['cart_response_number']}.csv"),
        },
        "cart_infiltration": {
            "fig": os.path.join(output_dir, f"{OUTPUT_BASENAME_MAP['cart_infiltration']}.svg"),
            "csv": os.path.join(output_dir, f"{OUTPUT_BASENAME_MAP['cart_infiltration']}.csv"),
        },
    }


def main(args):
    X_df, case_series, _ = load_sample_features_from_split(
        seq_path=args.seq_path,
        track_path=args.track_path,
        split_path=args.split_path,
        splits=args.splits,
    )
    out = build_output_paths(args.output_dir)
    for label_type in ["cart_response_number", "cart_infiltration"]:
        label_map = load_label_map(args.cart_csv_path, label_type)
        y = case_series.map(lambda c: label_map.get(_normalize_case_name(c)))
        valid_mask = y.notna()
        X_use = X_df.loc[valid_mask].reset_index(drop=True)
        y_use = y.loc[valid_mask].astype(float).reset_index(drop=True)
        if X_use.empty:
            raise ValueError(f"No matched samples found for {label_type}.")
        if y_use.nunique() < 2:
            raise ValueError(f"Label {label_type} has <2 unique values; Spearman is not meaningful.")
        ranked_spearman = compute_ranked_spearman(X_use, y_use)
        plot_ranked_spearman_bar(ranked_spearman, out[label_type]["fig"], out[label_type]["csv"])
        print(f"[INFO] Saved plot to {out[label_type]['fig']}")
        print(f"[INFO] Saved csv to {out[label_type]['csv']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_path", default=SEQ_DATASET_PATH)
    parser.add_argument("--track_path", default=TRACK_DATASET_PATH)
    parser.add_argument("--split_path", default="dynamics_data/data_split.json")
    parser.add_argument("--cart_csv_path", default="dynamics_data/CART_chip.csv")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output_dir", default="dynamics_tme-plot/plot")
    main(parser.parse_args())
