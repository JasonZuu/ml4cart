import os
import argparse
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dynamics_model.config import SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH, FEATURE_LEN, TRACK_LEN, DROPOUT
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split
from dynamics_model.model import CrossAttnFusionModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, default="dynamics/results/dl")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split_name", choices=["train", "val"], default="val")
    parser.add_argument("--method", choices=["pca", "tsne"], default="tsne")
    parser.add_argument("--marker_size", type=float, default=36.0)
    return parser.parse_args()


def _concat_input_features(split_part):
    x_seq = split_part["x_seq"].cpu().numpy()
    x_seq_mean = np.mean(x_seq, axis=1)
    x_seq_max = np.max(x_seq, axis=1)
    x_seq_min = np.min(x_seq, axis=1)
    parts = [x_seq_mean, x_seq_max, x_seq_min]
    parts.append(split_part["x_track"].cpu().numpy())
    parts.append(split_part["x_pdosize"].cpu().numpy())
    parts.append(split_part["x_antigen"].cpu().numpy())
    parts.append(split_part["x_tme"].cpu().numpy())
    return np.concatenate(parts, axis=1)


def _compute_embeddings(model, split_part, device):
    with torch.no_grad():
        x_seq = split_part["x_seq"].to(device)
        x_track = split_part["x_track"].to(device)
        x_pdo = split_part["x_pdosize"].to(device)
        x_antigen = split_part["x_antigen"].to(device)
        x_stroma = split_part["x_stroma"].to(device)
        x_immune = split_part["x_immune"].to(device)
        embeds = model.get_embedding(
            x_seq,
            x_track,
            x_pdosize=x_pdo,
            x_antigen=x_antigen,
            x_stroma=x_stroma,
            x_immune=x_immune,
        )
    return embeds.detach().cpu().numpy()


def _plot_space(z_train, z_eval, y_train_np, y_eval_np, classes, set_colors, markers, marker_size, title, eval_name):
    train_handles = []
    val_handles = []
    for idx, c in enumerate(classes):
        tr_mask = (y_train_np == c)
        va_mask = (y_eval_np == c)
        marker = markers[idx % len(markers)]
        if np.any(tr_mask):
            plt.scatter(
                z_train[tr_mask, 0],
                z_train[tr_mask, 1],
                s=marker_size,
                alpha=0.85,
                c=set_colors["train"],
                marker=marker,
                edgecolors="white",
                linewidths=0.5,
            )
            train_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=set_colors["train"],
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=max(4.0, float(marker_size) ** 0.5),
                    label=f"class {c} | train",
                )
            )
        if np.any(va_mask):
            plt.scatter(
                z_eval[va_mask, 0],
                z_eval[va_mask, 1],
                s=marker_size,
                alpha=0.85,
                c=set_colors["val"],
                marker=marker,
                edgecolors="white",
                linewidths=0.5,
            )
            val_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=set_colors["val"],
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=max(4.0, float(marker_size) ** 0.5),
                    label=f"class {c} | {eval_name}",
                )
            )
    handles = train_handles + val_handles
    plt.legend(handles=handles, ncol=2, fontsize=9, frameon=True)
    plt.title(title, fontsize=12, pad=8)
    plt.tight_layout()


def _reduce_and_plot(X_train, X_eval, E_train, E_eval, y_train, y_eval, out_dir, method="pca", marker_size=36.0, eval_name="val"):
    os.makedirs(out_dir, exist_ok=True)
    if method == "pca":
        reducer_in = PCA(n_components=2, random_state=42)
        reducer_emb = PCA(n_components=2, random_state=42)
    else:
        reducer_in = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        reducer_emb = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    Xin = np.concatenate([X_train, X_eval], axis=0)
    Ein = np.concatenate([E_train, E_eval], axis=0)
    Z_in = reducer_in.fit_transform(Xin)
    Z_emb = reducer_emb.fit_transform(Ein)
    n_train = X_train.shape[0]
    Z_in_train, Z_in_eval = Z_in[:n_train], Z_in[n_train:]
    Z_emb_train, Z_emb_eval = Z_emb[:n_train], Z_emb[n_train:]
    y_train_np = y_train.cpu().numpy() if hasattr(y_train, "cpu") else np.asarray(y_train)
    y_eval_np = y_eval.cpu().numpy() if hasattr(y_eval, "cpu") else np.asarray(y_eval)
    classes = np.unique(np.concatenate([y_train_np, y_eval_np]))
    set_colors = {"train": "#2563eb", "val": "#f59e0b"}
    markers = ["o", "s", "^", "D", "P", "X", "*"]
    plt.figure(figsize=(7, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    _plot_space(
        Z_in_train,
        Z_in_eval,
        y_train_np,
        y_eval_np,
        classes,
        set_colors,
        markers,
        marker_size,
        f"Input Space (Train vs {eval_name.capitalize()})",
        eval_name,
    )
    plt.savefig(os.path.join(out_dir, f"input_{method}_{eval_name}.png"))
    plt.savefig(os.path.join(out_dir, f"input_{method}_{eval_name}.svg"))
    plt.close()
    plt.figure(figsize=(7, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    _plot_space(
        Z_emb_train,
        Z_emb_eval,
        y_train_np,
        y_eval_np,
        classes,
        set_colors,
        markers,
        marker_size,
        f"Embedding Space (Train vs {eval_name.capitalize()})",
        eval_name,
    )
    plt.savefig(os.path.join(out_dir, f"embedding_{method}_{eval_name}.png"))
    plt.savefig(os.path.join(out_dir, f"embedding_{method}_{eval_name}.svg"))
    plt.close()


def main(args):
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = subset_split_by_case(SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    data = build_datasets_from_case_split(split)
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
    model_path = args.model_path or os.path.join(args.weights_dir, "best_model.pth")
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        model.load_state_dict(sd)
    model.eval()
    base_out = args.output_dir or os.path.join(args.weights_dir, "plots")
    out_dir = os.path.join(base_out, f"train_{args.split_name}_space")
    X_train = _concat_input_features(data["train"])
    X_val = _concat_input_features(data[args.split_name])
    E_train = _compute_embeddings(model, data["train"], device)
    E_val = _compute_embeddings(model, data[args.split_name], device)
    y_train = data["train"]["y"]
    y_val = data[args.split_name]["y"]
    _reduce_and_plot(
        X_train,
        X_val,
        E_train,
        E_val,
        y_train,
        y_val,
        out_dir,
        method=args.method,
        marker_size=args.marker_size,
        eval_name=args.split_name,
    )
    print(f"Input/Embedding comparison saved to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
