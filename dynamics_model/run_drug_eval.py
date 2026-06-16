"""
Evaluate the trained dynamics model on Drug test cases and visualise drug impact per patient.

Run from repo root:
    PYTHONPATH=dynamics python dynamics/run_drug_eval.py

Outputs one figure per patient (NYU285 / NYU318 / NYU774) to:
    dynamics/results/dl/test_plots/drug_impact_<patient>.png/.svg
"""

import os
import sys
# Allow both `import Config` (dynamics/ on path) and `from dynamics_model.config import …` (repo root on path)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dyn_root  = os.path.join(_repo_root, "dynamics")
for _p in (_dyn_root, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split
from dynamics_model.model.CrossAttnFusionModel import CrossAttnFusionModel
from dynamics_model.config import (
    SEQ_DATASET_PATH, TRACK_DATASET_PATH,
    TEST_TRAIN_SPLIT_ANNOTATION_PATH,
    FEATURE_LEN, TRACK_LEN, DROPOUT,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH  = "dynamics/results/dl/best_model.pth"
OUTPUT_DIR  = "dynamics/results/dl/test_plots"

# Patient group → Drug case names (condition order: IgG ctrl, iAREG, FAP CAR T)
PATIENT_GROUPS = {
    "NYU285": ["Drug_1.1", "Drug_1.2", "Drug_1.3"],
    "NYU318": ["Drug_2.1", "Drug_2.2", "Drug_2.3"],
    "NYU774": ["Drug_3.1", "Drug_3.2", "Drug_3.3"],
}

CONDITION_LABELS = ["IgG ctrl", "iAREG", "FAP CAR T"]

# Matches existing codebase colour scheme
CLASS_COLORS = ["#90BFF9", "#FFC080", "#FFA0A0"]   # Progressive, Stable, Responsive
CLASS_NAMES  = ["Progressive", "Stable", "Responsive"]


# ---------------------------------------------------------------------------
# Step 1 – Fit scalers from training data
# ---------------------------------------------------------------------------
def get_training_scalers(seq_path, track_path, split_json):
    """Return scalers fitted on training data."""
    split = subset_split_by_case(seq_path, track_path, split_json)
    ds    = build_datasets_from_case_split(split)
    return ds["scalers"], ds


# ---------------------------------------------------------------------------
# Step 2 – Load Drug tracks from NPZ
# ---------------------------------------------------------------------------
def load_drug_tracks(seq_path, track_path):
    """Return dict: case_name → {'X_seq': ndarray(N,100,7), 'X_track': ndarray(N,3)}."""
    seq_data   = np.load(seq_path,   allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq_all      = seq_data["X"]           # (N_seq, 100, 7)
    track_ids_seq  = seq_data["track_ids"]
    X_track_all    = track_data["X"]         # (N_trk, 3)
    track_ids_trk  = track_data["track_ids"]

    # Build lookup: track_id tuple → row index in X_track_all
    trk_lookup = {}
    for i, tid in enumerate(track_ids_trk):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        trk_lookup[key] = i

    drug_tracks = {}
    for i, tid in enumerate(track_ids_seq):
        key         = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        sample_name = tid[0] if isinstance(tid, (list, tuple, np.ndarray)) else str(tid)
        case_name   = "_".join(sample_name.split("_")[:-1])   # strip XY## suffix

        if not case_name.startswith("Drug_"):
            continue
        if key not in trk_lookup:
            continue

        trk_idx = trk_lookup[key]
        if case_name not in drug_tracks:
            drug_tracks[case_name] = {"X_seq": [], "X_track": []}
        drug_tracks[case_name]["X_seq"].append(X_seq_all[i])
        drug_tracks[case_name]["X_track"].append(X_track_all[trk_idx])

    # Stack lists to arrays
    for case in drug_tracks:
        drug_tracks[case]["X_seq"]   = np.stack(drug_tracks[case]["X_seq"],   axis=0)
        drug_tracks[case]["X_track"] = np.stack(drug_tracks[case]["X_track"], axis=0)

    return drug_tracks


# ---------------------------------------------------------------------------
# Step 3 – Normalise Drug data using training scalers
# ---------------------------------------------------------------------------
def scale_drug_data(drug_tracks, scalers, meta, all_drug_cases):
    """
    Apply training scalers to Drug tracks.
    Returns dict: case_name → dict of torch tensors ready for the model.
    """
    seq_scaler     = scalers["seq"]
    track_scaler   = scalers["track"]
    pdo_scaler     = scalers["pdo_size"]
    antigen_scaler = scalers["antigen"]
    tme_scaler     = scalers["tme"]    # fitted on [lag3, pd1, col1, col3, col4, ha]

    pdo_map    = meta.get("pdo_size",  {})
    antigen_map= meta.get("antigen",   {})
    lag3_map   = meta.get("LAG-3",     {})
    pd1_map    = meta.get("PD-1",      {})
    col1_map   = meta.get("COL-I",     {})
    col3_map   = meta.get("COL-III",   {})
    col4_map   = meta.get("COL-IV",    {})
    ha_map     = meta.get("HA",        {})

    scaled = {}
    for case in all_drug_cases:
        if case not in drug_tracks:
            print(f"  [WARN] No tracks found for {case}")
            continue

        X_seq   = drug_tracks[case]["X_seq"]    # (N, 100, 7)
        X_track = drug_tracks[case]["X_track"]  # (N,  3)
        N       = len(X_seq)

        # Sequence scaling
        n, t, f = X_seq.shape
        X_seq_s = seq_scaler.transform(X_seq.reshape(-1, f)).reshape(n, t, f)

        # Track scaling
        X_track_s = track_scaler.transform(X_track)

        # Metadata: broadcast single case value to all N tracks
        def _broadcast(val):
            return np.full((N, 1), float(val), dtype=np.float32)

        pdo_val    = _broadcast(pdo_map.get(case, 0.0))
        antigen_val= _broadcast(antigen_map.get(case, 0.0))
        tme_row    = np.array([[
            lag3_map.get(case, 0.0),
            pd1_map.get(case, 0.0),
            col1_map.get(case, 0.0),
            col3_map.get(case, 0.0),
            col4_map.get(case, 0.0),
            ha_map.get(case, 0.0),
        ]], dtype=np.float32)
        tme_val = np.repeat(tme_row, N, axis=0)     # (N, 6)

        pdo_s     = pdo_scaler.transform(pdo_val)
        antigen_s = antigen_scaler.transform(antigen_val)
        tme_s     = tme_scaler.transform(tme_val)

        scaled[case] = {
            "x_seq":     torch.tensor(X_seq_s,        dtype=torch.float32),
            "x_track":   torch.tensor(X_track_s,      dtype=torch.float32),
            "x_pdo":     torch.tensor(pdo_s,          dtype=torch.float32),
            "x_antigen": torch.tensor(antigen_s,      dtype=torch.float32),
            "x_immune":  torch.tensor(tme_s[:, :2],   dtype=torch.float32),  # [lag3, pd1]
            "x_stroma":  torch.tensor(tme_s[:, 2:6],  dtype=torch.float32),  # [col1, col3, col4, ha]
            "n_tracks":  N,
        }
    return scaled


# ---------------------------------------------------------------------------
# Step 4 – Run inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, case_data, device, batch_size=512):
    """Return (preds ndarray, probs ndarray) for a single Drug case."""
    keys   = ["x_seq", "x_track", "x_pdo", "x_antigen", "x_stroma", "x_immune"]
    N      = case_data["n_tracks"]
    preds_all, probs_all = [], []

    for start in range(0, N, batch_size):
        end     = min(start + batch_size, N)
        tensors = {k: case_data[k][start:end].to(device) for k in keys}
        logits  = model(
            tensors["x_seq"], tensors["x_track"],
            tensors["x_pdo"], tensors["x_antigen"],
            x_stroma=tensors["x_stroma"],
            x_immune=tensors["x_immune"],
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        preds_all.append(preds)
        probs_all.append(probs)

    return np.concatenate(preds_all), np.concatenate(probs_all)


# ---------------------------------------------------------------------------
# Step 5 – Aggregate results per case
# ---------------------------------------------------------------------------
def aggregate_results(model, scaled_data, size_change_map, device):
    """Return dict: case_name → {proportions, combined_score, size_change, n_tracks}."""
    results = {}
    for case, data in scaled_data.items():
        preds, probs = run_inference(model, data, device)
        counts = np.bincount(preds, minlength=3)
        total  = counts.sum()
        props  = counts / total  # [Progressive, Stable, Responsive]
        score  = 0.0 * props[0] + 0.5 * props[1] + 1.0 * props[2]

        results[case] = {
            "proportions":    props,
            "combined_score": float(score),
            "size_change":    size_change_map.get(case, float("nan")),
            "n_tracks":       int(total),
            "probs":          probs,
        }
        print(f"  {case}  n={total:4d}  "
              f"Prog={props[0]:.2%}  Stab={props[1]:.2%}  Resp={props[2]:.2%}  "
              f"Score={score:.3f}  ΔPDOsize={size_change_map.get(case, float('nan')):+.1f}%")
    return results


# ---------------------------------------------------------------------------
# Step 6 – Plot one figure per patient
# ---------------------------------------------------------------------------
def plot_patient_figure(patient_id, case_keys, results, output_dir):
    """
    Single-panel figure for a single patient:
      Stacked bar chart of Progressive / Stable / Responsive proportions
      for each drug condition, with Combined Score annotated above.
    """
    present = [k for k in case_keys if k in results]
    if not present:
        print(f"  [WARN] No results for patient {patient_id}, skipping plot.")
        return

    labels      = CONDITION_LABELS[:len(present)]
    props       = np.array([results[k]["proportions"]  for k in present])  # (3, 3)
    scores      = [results[k]["combined_score"]        for k in present]
    pdo_changes = [results[k]["size_change"]           for k in present]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5.5))
    fig.suptitle(f"Patient {patient_id}  —  Drug Impact on CAR-T Cell Response",
                 fontsize=13, fontweight="bold")

    x     = np.arange(len(present))
    bar_w = 0.5

    # Stacked proportion bars — label each bar with its condition name
    bottoms = np.zeros(len(present))
    bar_handles = []
    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        vals = props[:, cls_idx] * 100
        bars = ax1.bar(x, vals, bar_w, bottom=bottoms,
                       color=color, edgecolor="black", linewidth=0.8)
        bar_handles.append(bars[0])
        for i, (v, bot) in enumerate(zip(vals, bottoms)):
            if v > 8:
                ax1.text(i, bot + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=8, color="#333333")
        bottoms += vals

    # Annotate Combined Score above each bar
    for i, score in enumerate(scores):
        ax1.text(i, 102, f"Score\n{score:.3f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#1A1A2E")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Proportion of Tracks (%)", fontsize=10)
    ax1.set_ylim(0, 118)
    ax1.set_title("T Cell Behavioural Distribution", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # PDO size change annotations below each x-tick label
    for i, pdo in enumerate(pdo_changes):
        if not np.isfinite(pdo):
            continue
        color = "#C0392B" if pdo < 0 else "#2980B9"
        ax1.annotate(f"ΔSize: {pdo:+.1f}%", xy=(i, 0), xytext=(0, -28),
                     xycoords=("data", "axes fraction"),
                     textcoords="offset points",
                     ha="center", va="top", fontsize=8.5,
                     color=color, fontweight="bold",
                     annotation_clip=False)

    # Legend outside on the right, class colours only, no frame
    ax1.legend(bar_handles, CLASS_NAMES,
               loc="upper left", bbox_to_anchor=(1.02, 1),
               borderaxespad=0, fontsize=11, frameon=False,
               handlelength=1.2, handletextpad=0.5)

    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "svg"):
        path = os.path.join(output_dir, f"drug_impact_{patient_id}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Fit scalers on training data ────────────────────────────────────────
    print("\n[1] Loading training data to fit scalers …")
    scalers, _ = get_training_scalers(
        SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH
    )
    print("    Scalers fitted.")

    # ── Load meta (PDO size change etc.) ────────────────────────────────────
    print("\n[2] Loading metadata from data_split.json …")
    with open(TEST_TRAIN_SPLIT_ANNOTATION_PATH) as f:
        cfg = json.load(f)
    meta             = cfg.get("meta", {})
    size_change_map  = meta.get("size_change_by_case", {})

    # ── Load Drug tracks ────────────────────────────────────────────────────
    print("\n[3] Loading Drug tracks from NPZ …")
    drug_tracks = load_drug_tracks(SEQ_DATASET_PATH, TRACK_DATASET_PATH)
    all_drug_cases = [c for group in PATIENT_GROUPS.values() for c in group]
    print(f"    Cases loaded: {sorted(drug_tracks.keys())}")

    # ── Normalise ────────────────────────────────────────────────────────────
    print("\n[4] Normalising Drug data with training scalers …")
    scaled_data = scale_drug_data(drug_tracks, scalers, meta, all_drug_cases)

    # ── Load model ───────────────────────────────────────────────────────────
    print("\n[5] Loading model …")
    model = CrossAttnFusionModel(
        seq_input_size=FEATURE_LEN,
        track_input_size=TRACK_LEN,
        hidden_size=32,
        fusion_size=32,
        dropout=DROPOUT,
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    Loaded from {MODEL_PATH}")

    # ── Inference ────────────────────────────────────────────────────────────
    print("\n[6] Running inference …")
    results = aggregate_results(model, scaled_data, size_change_map, device)

    # ── Save per-case CSV ────────────────────────────────────────────────────
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "drug_eval_results.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "case", "patient", "condition",
            "n_tracks", "Progressive", "Stable", "Responsive",
            "combined_score", "pdo_size_change"
        ])
        writer.writeheader()
        patient_map    = {v: k for k, grp in PATIENT_GROUPS.items() for v in grp}
        condition_map  = {
            "Drug_1.1": "IgG ctrl",  "Drug_1.2": "iAREG",  "Drug_1.3": "FAP CAR T",
            "Drug_2.1": "IgG ctrl",  "Drug_2.2": "iAREG",  "Drug_2.3": "FAP CAR T",
            "Drug_3.1": "IgG ctrl",  "Drug_3.2": "iAREG",  "Drug_3.3": "FAP CAR T",
        }
        for case in sorted(results):
            r = results[case]
            writer.writerow({
                "case":           case,
                "patient":        patient_map.get(case, ""),
                "condition":      condition_map.get(case, ""),
                "n_tracks":       r["n_tracks"],
                "Progressive":    f"{r['proportions'][0]:.4f}",
                "Stable":         f"{r['proportions'][1]:.4f}",
                "Responsive":     f"{r['proportions'][2]:.4f}",
                "combined_score": f"{r['combined_score']:.4f}",
                "pdo_size_change":f"{r['size_change']:.4f}" if np.isfinite(r["size_change"]) else "",
            })
    print(f"\n  Results CSV saved: {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n[7] Generating figures …")
    for patient_id, case_keys in PATIENT_GROUPS.items():
        print(f"\n  Patient {patient_id}:")
        plot_patient_figure(patient_id, case_keys, results, OUTPUT_DIR)

    print("\nDone. Plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
