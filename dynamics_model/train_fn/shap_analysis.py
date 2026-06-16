import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dynamics_model.model import SelfAttnFusionModel, CrossAttnFusionModel
from dynamics_model.config import DROPOUT, SEQ_LEN, FEATURE_LEN, TRACK_LEN, SEQ_DATASET_PATH, TRACK_DATASET_PATH, RESULTS_DIR, TEST_TRAIN_SPLIT_ANNOTATION_PATH, features, track_features
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split


def _cap(s):
    return str(s).upper()


def run_shap_analysis(model_name: str, out_dir: str, model_path: str, hidden_size: int = 64, fusion_size: int = 64, dropout: float = DROPOUT, use_track: bool = True, use_pdosize: bool = True, use_antigen: bool = True, use_tme: bool = True, output_index: int = 0, split_name: str = "test"):
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = subset_split_by_case(SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    data = build_datasets_from_case_split(split)
    x_seq = data[split_name]["x_seq"].to(device)
    x_track = data[split_name]["x_track"].to(device)
    x_pdo = data[split_name]["x_pdosize"].to(device)
    x_antigen = data[split_name]["x_antigen"].to(device)
    x_tme = data[split_name]["x_tme"].to(device)

    if model_name == "self_attn":
        model = SelfAttnFusionModel(seq_input_size=FEATURE_LEN, track_input_size=TRACK_LEN, hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout, use_track=use_track, use_pdosize=use_pdosize, use_antigen=use_antigen, use_tme=use_tme).to(device)
    elif model_name == "cross_attn":
        model = CrossAttnFusionModel(seq_input_size=FEATURE_LEN, track_input_size=TRACK_LEN, hidden_size=hidden_size, fusion_size=fusion_size, dropout=dropout, use_track=use_track, use_pdosize=use_pdosize, use_antigen=use_antigen, use_tme=use_tme).to(device)
    else:
        raise ValueError("model_name must be 'self_attn' or 'cross_attn'.")
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    if hasattr(model, "self_attn") and hasattr(model.self_attn, "dropout"):
        model.self_attn.dropout = 0.0
    if hasattr(model, "cross_attn") and hasattr(model.cross_attn, "dropout"):
        model.cross_attn.dropout = 0.0

    bsz = x_seq.shape[0]
    seq_dim = SEQ_LEN * FEATURE_LEN
    parts = []
    slices = {}
    off = 0
    x_seq_flat = x_seq.reshape(bsz, -1)
    parts.append(x_seq_flat)
    slices["seq"] = slice(off, off + seq_dim)
    off += seq_dim
    if use_track:
        parts.append(x_track)
        slices["track"] = slice(off, off + TRACK_LEN)
        off += TRACK_LEN
    if use_pdosize:
        parts.append(x_pdo)
        slices["pdo"] = slice(off, off + 1)
        off += 1
    if use_antigen:
        parts.append(x_antigen)
        slices["antigen"] = slice(off, off + 1)
        off += 1
    if use_tme:
        parts.append(x_tme)
        slices["tme"] = slice(off, off + 5)
        off += 5
    x_all = torch.cat(parts, dim=1)
    x_all.requires_grad_(True)

    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            b = x.shape[0]
            off = 0
            seq_flat = x[:, off : off + seq_dim]
            seq = seq_flat.view(b, SEQ_LEN, FEATURE_LEN)
            off += seq_dim
            if use_track:
                trk = x[:, off : off + TRACK_LEN]
                off += TRACK_LEN
            else:
                trk = torch.zeros(b, TRACK_LEN, dtype=seq.dtype, device=seq.device)
            if use_pdosize:
                pdo = x[:, off : off + 1]
                off += 1
            else:
                pdo = torch.zeros(b, 1, dtype=seq.dtype, device=seq.device)
            if use_antigen:
                antigen = x[:, off : off + 1]
                off += 1
            else:
                antigen = torch.zeros(b, 1, dtype=seq.dtype, device=seq.device)
            if use_tme:
                tme = x[:, off : off + 5]
            else:
                tme = torch.zeros(b, 5, dtype=seq.dtype, device=seq.device)
            if model_name in {"self_attn", "cross_attn"}:
                immune = tme[:, :2]
                stroma = torch.zeros(b, 4, dtype=seq.dtype, device=seq.device)
                width = min(4, max(0, tme.shape[1] - 2))
                if width > 0:
                    stroma[:, :width] = tme[:, 2 : 2 + width]
                return self.model(seq, trk, pdo, antigen, x_stroma=stroma, x_immune=immune)
            immune = tme[:, :2]
            ecm = tme[:, 2:3]
            cytokine = tme[:, 3:5]
            return self.model(seq, trk, pdo, antigen, immune, ecm, cytokine)

    wrapped = WrappedModel(model).to(device)
    background = x_all[: min(100, x_all.shape[0])]
    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(x_all[: background.shape[0]])
    shap_vals = shap_values[output_index] if isinstance(shap_values, (list, tuple)) else shap_values

    rows = []
    seq_slice = slices["seq"]
    shap_seq = shap_vals[:, seq_slice].reshape(-1, SEQ_LEN, FEATURE_LEN)
    for idx, name in enumerate(features):
        val = float(np.mean(shap_seq[:, :, idx]))
        rows.append({"feature": _cap(name), "importance": val})

    if use_track:
        for i, name in enumerate(track_features):
            idx = slices["track"].start + i
            val = float(np.mean(shap_vals[:, idx]))
            rows.append({"feature": _cap(name), "importance": val})
    if use_pdosize:
        idx = slices["pdo"].start
        val = float(np.mean(shap_vals[:, idx]))
        rows.append({"feature": "PDO_SIZE", "importance": val})
    if use_antigen:
        idx = slices["antigen"].start
        val = float(np.mean(shap_vals[:, idx]))
        rows.append({"feature": "TARGETED_ANTIGEN", "importance": val})
    if use_tme:
        idx0 = slices["tme"].start
        val0 = float(np.mean(shap_vals[:, idx0]))
        val1 = float(np.mean(shap_vals[:, idx0 + 1]))
        rows.append({"feature": "PD1", "importance": val0})
        rows.append({"feature": "LAG3", "importance": val1})
        idx = slices["tme"].start + 2
        val = float(np.mean(shap_vals[:, idx]))
        rows.append({"feature": "COL3", "importance": val})
        idx0 = slices["tme"].start + 3
        val0 = float(np.mean(shap_vals[:, idx0]))
        val1 = float(np.mean(shap_vals[:, idx0 + 1]))
        rows.append({"feature": "SDF1", "importance": val0})
        rows.append({"feature": "PIGF", "importance": val1})

    df = pd.DataFrame(rows)
    df = df.reindex(df["importance"].abs().sort_values(ascending=False).index)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "shap_feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.barplot(data=df, x="importance", y="feature", palette="viridis")
    plt.title("SHAP Importance by Feature", fontsize=13, pad=10)
    plt.xlabel("Mean SHAP (signed)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "shap_feature_importance.svg"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unified", "self_attn", "cross_attn"], default="cross_attn")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default=os.path.join(RESULTS_DIR, "shap_new"))
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--fusion_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--use_track", type=bool, default=True)
    parser.add_argument("--use_pdosize", type=bool, default=True)
    parser.add_argument("--use_antigen", type=bool, default=True)
    parser.add_argument("--use_tme", type=bool, default=True)
    parser.add_argument("--output_index", type=int, default=0)
    parser.add_argument("--split_name", type=str, default="test")
    args = parser.parse_args()
    run_shap_analysis(args.model, args.output_dir, args.model_path, hidden_size=args.hidden_size, fusion_size=args.fusion_size, dropout=args.dropout, use_track=args.use_track, use_pdosize=args.use_pdosize, use_antigen=args.use_antigen, use_tme=args.use_tme, output_index=args.output_index, split_name=args.split_name)
