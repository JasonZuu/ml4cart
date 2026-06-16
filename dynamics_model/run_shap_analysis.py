import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dynamics_model.config import (
    DROPOUT,
    SEQ_DATASET_PATH,
    TRACK_DATASET_PATH,
    TEST_TRAIN_SPLIT_ANNOTATION_PATH,
    FEATURE_LEN,
    TRACK_LEN,
    features,
    track_features,
)
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split
from dynamics_model.model import CrossAttnFusionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, default="dynamics/results/dl")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split_name", choices=["train", "val"], default="val")
    return parser.parse_args()


def _collect_inputs(split_part, device):
    x_seq = split_part["x_seq"].to(device)
    b = x_seq.shape[0]
    seq_len = x_seq.shape[1]
    feat_len = x_seq.shape[2]
    track_len = split_part["x_track"].shape[1]
    x_seq_flat = x_seq.reshape(b, -1)
    x_track = split_part["x_track"].to(device)
    x_pdo = split_part["x_pdosize"].to(device)
    x_antigen = split_part["x_antigen"].to(device)
    x_immune = split_part["x_immune"].to(device)
    x_stroma = split_part["x_stroma"].to(device)
    x_tme = torch.cat([x_immune, x_stroma], dim=1)
    x_all = torch.cat([x_seq_flat, x_track, x_pdo, x_antigen, x_tme], dim=1).detach()
    offsets = {
        "seq": (0, seq_len * feat_len),
        "track": (seq_len * feat_len, seq_len * feat_len + track_len),
        "pdo": (seq_len * feat_len + track_len, seq_len * feat_len + track_len + 1),
        "antigen": (seq_len * feat_len + track_len + 1, seq_len * feat_len + track_len + 2),
        "tme": (seq_len * feat_len + track_len + 2, seq_len * feat_len + track_len + 8),
    }
    return x_all, offsets, seq_len, feat_len, track_len


def _build_feature_names(seq_len, feat_len, track_len):
    seq_names = list(features[:feat_len]) if len(features) >= feat_len else [f"SEQ_{i}" for i in range(feat_len)]
    track_names = list(track_features[:track_len]) if len(track_features) >= track_len else [f"TRACK_{i}" for i in range(track_len)]
    flat_seq = [f"{name}_T{t}" for t in range(seq_len) for name in seq_names]
    return flat_seq + track_names + ["PDO_SIZE", "TARGETED_ANTIGEN", "LAG3", "PD1", "COL-I", "COL-III", "COL-IV", "HA"]


def _to_numpy_array(x):
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_per_class_shap(shap_values, num_classes_hint=3):
    if isinstance(shap_values, (list, tuple)):
        out = []
        for sv in shap_values:
            arr = _to_numpy_array(sv)
            if arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            out.append(arr)
        return out
    arr = _to_numpy_array(shap_values)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[0] == num_classes_hint:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.shape[-1] == num_classes_hint:
            return [arr[:, :, i] for i in range(arr.shape[-1])]
    arr = arr.reshape(arr.shape[0], -1)
    return [arr]


def _merge_time_features(feature_names, importance_abs):
    groups = {}
    order = []
    for i, name in enumerate(feature_names):
        m = re.match(r"^(.*)_T\d+$", str(name))
        key = m.group(1) if m else str(name)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(float(importance_abs[i]))
    rows = [{"feature": k, "importance": float(np.mean(v))} for k, v in [(k, groups[k]) for k in order]]
    df = pd.DataFrame(rows)
    df["_abs"] = df["importance"].abs()
    df = df.sort_values(by="_abs", ascending=False).drop(columns=["_abs"])
    return df.reset_index(drop=True)


def _save_df_and_plot(df, out_dir, stem, title):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    plt.figure(figsize=(10, 10))
    plt.style.use("seaborn-v0_8-whitegrid")
    top_df = df.head(30)
    sns.barplot(data=top_df, x="importance", y="feature", palette="viridis")
    plt.title(title, fontsize=13, pad=10)
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{stem}.svg"), dpi=300, bbox_inches="tight")
    plt.close()


def _run_shap_analysis(args, out_dir, split_name):
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = subset_split_by_case(SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    data = build_datasets_from_case_split(split)
    split_part = data[split_name]
    model_path = args.model_path or os.path.join(args.weights_dir, "best_model.pth")
    model = CrossAttnFusionModel(
        seq_input_size=FEATURE_LEN,
        track_input_size=TRACK_LEN,
        hidden_size=32,
        fusion_size=32,
        dropout=DROPOUT,
        use_track=True,
        use_pdosize=True,
        use_antigen=True,
        use_tme=True,
    ).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

    x_all, offsets, seq_len, feat_len, track_len = _collect_inputs(split_part, device=device)
    x_all.requires_grad_(True)

    class WrappedModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            b = x.shape[0]
            s0, s1 = offsets["seq"]
            seq_flat = x[:, s0:s1]
            seq = seq_flat.view(b, seq_len, feat_len)
            t0, t1 = offsets["track"]
            track = x[:, t0:t1]
            p0, p1 = offsets["pdo"]
            pdo = x[:, p0:p1]
            a0, a1 = offsets["antigen"]
            antigen = x[:, a0:a1]
            m0, m1 = offsets["tme"]
            tme = x[:, m0:m1]
            immune = tme[:, :2]
            stroma = tme[:, 2:6]
            return self.base_model(seq, track, x_pdosize=pdo, x_antigen=antigen, x_stroma=stroma, x_immune=immune)

    wrapped = WrappedModel(model).to(device)
    bg_n = min(100, x_all.shape[0])
    background = x_all[:bg_n]
    eval_n = min(200, x_all.shape[0])
    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(x_all[:eval_n])
    feature_names = _build_feature_names(seq_len, feat_len, track_len)
    per_class = _extract_per_class_shap(shap_values, num_classes_hint=3)
    per_class_abs_raw = []
    for class_idx, class_sv in enumerate(per_class):
        class_sv = _to_numpy_array(class_sv)
        if class_sv.ndim > 2:
            class_sv = class_sv.reshape(class_sv.shape[0], -1)
        if class_sv.shape[1] != len(feature_names):
            class_sv = class_sv[:, : len(feature_names)]
        class_abs = np.mean(np.abs(class_sv), axis=0)
        per_class_abs_raw.append(class_abs)
        class_df = _merge_time_features(feature_names, class_abs)
        _save_df_and_plot(
            class_df,
            out_dir,
            stem=f"shap_feature_importance_class_{class_idx}",
            title=f"SHAP Importance Class {class_idx} ({split_name})",
        )

    use_n = min(3, len(per_class_abs_raw))
    if use_n == 0:
        raise ValueError("No SHAP outputs found.")
    global_abs = np.mean(np.stack(per_class_abs_raw[:use_n], axis=0), axis=0)
    global_df = _merge_time_features(feature_names, global_abs)
    _save_df_and_plot(
        global_df,
        out_dir,
        stem="shap_feature_importance_global",
        title=f"SHAP Importance Global ({split_name})",
    )
    _save_df_and_plot(
        global_df,
        out_dir,
        stem="shap_feature_importance",
        title=f"SHAP Importance Global ({split_name})",
    )


if __name__ == "__main__":
    args = parse_args()
    base_out = args.output_dir or os.path.join(args.weights_dir, "plots")
    split_out = os.path.join(base_out, f"{args.split_name}_shap")
    _run_shap_analysis(args, split_out, args.split_name)
    print(f"SHAP analysis saved to {split_out}")
