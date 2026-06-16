import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, r2_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import torch
from torch.utils.data import DataLoader
matplotlib.use('Agg')


def compute_metrics(y_true, preds, probs):
    acc = np.mean(preds == y_true)
    f1 = f1_score(y_true, preds, average="macro")

    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    try:
        auc_value = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
    except:
        auc_value = -1
    return acc, f1, auc_value, y_true_bin


def plot_roc(y_true_bin, probs, result_path, n_classes=3):
    if y_true_bin.shape[1] == 1: # turn to one-hot for n_classes
        y_true_bin = label_binarize(y_true_bin, classes=np.arange(n_classes))
    rows = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
        for j in range(len(fpr)):
            rows.append({"class": i, "fpr": float(fpr[j]), "tpr": float(tpr[j]), "auc": float(roc_auc)})
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(result_path,"roc_curve.png"))
    plt.savefig(os.path.join(result_path,"roc_curve.svg"))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(result_path, "roc_curve.csv"), index=False)
    plt.close()


def plot_confusion_matrix(y_true, preds, classes, result_path):
    labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, preds, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.savefig(os.path.join(result_path, "confusion_matrix.svg"))
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    df_cm.to_csv(os.path.join(result_path, "confusion_matrix.csv"))
    plt.close()
    return cm


def fusion_weight_analysis(model, test_loader, device, result_path):
    model.eval()
    default_query_labels = ["seq", "track", "tumor", "stroma", "immune"]
    default_key_labels = ["seq", "track", "tumor", "stroma", "immune"]
    cross_query_labels = ["seq", "track"]
    cross_key_labels = ["tumor", "stroma", "immune"]
    attn_sum = None
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                batch_seq, batch_track, batch_y = batch
                x_pdo = torch.zeros(batch_seq.size(0), 1, dtype=batch_seq.dtype, device=device)
                x_antigen = torch.zeros(batch_seq.size(0), 1, dtype=batch_seq.dtype, device=device)
                x_immune = torch.zeros(batch_seq.size(0), 2, dtype=batch_seq.dtype, device=device)
                x_stroma = torch.zeros(batch_seq.size(0), 4, dtype=batch_seq.dtype, device=device)
            elif len(batch) == 5:
                batch_seq, batch_track, batch_y, x_pdo, x_antigen = batch
                x_immune = torch.zeros(batch_seq.size(0), 2, dtype=batch_seq.dtype, device=device)
                x_stroma = torch.zeros(batch_seq.size(0), 4, dtype=batch_seq.dtype, device=device)
            elif len(batch) >= 7:
                batch_seq, batch_track, batch_y, x_pdo, x_antigen, x_stroma, x_immune = batch[:7]
            else:
                continue
            batch_seq = batch_seq.to(device)
            batch_track = batch_track.to(device)
            x_pdo = x_pdo.to(device)
            x_antigen = x_antigen.to(device)
            x_stroma = x_stroma.to(device)
            x_immune = x_immune.to(device)
            if not hasattr(model, "get_attn_weights"):
                continue
            attn = model.get_attn_weights(
                batch_seq,
                batch_track,
                x_pdo,
                x_antigen,
                x_stroma=x_stroma,
                x_immune=x_immune,
            )
            attn = attn.mean(dim=0).detach().cpu().numpy()
            if attn_sum is None:
                attn_sum = attn
            else:
                attn_sum += attn
            count += 1
    if attn_sum is None:
        return None
    attn_map = attn_sum / max(count, 1)
    q_len, k_len = attn_map.shape[0], attn_map.shape[1]
    if q_len == 2 and k_len == 3:
        query_labels = cross_query_labels
        key_labels = cross_key_labels
        title = "Cross-Attention Map"
    else:
        if q_len <= len(default_query_labels):
            query_labels = default_query_labels[:q_len]
        else:
            query_labels = default_query_labels + [f"query_{i}" for i in range(len(default_query_labels), q_len)]
        if k_len <= len(default_key_labels):
            key_labels = default_key_labels[:k_len]
        else:
            key_labels = default_key_labels + [f"key_{i}" for i in range(len(default_key_labels), k_len)]
        title = "Attention Map"
    os.makedirs(result_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_map, xticklabels=key_labels, yticklabels=query_labels, cmap="viridis", annot=True, fmt=".2f")
    plt.title(title)
    plt.savefig(os.path.join(result_path, "attention_map.png"))
    plt.savefig(os.path.join(result_path, "attention_map.svg"))
    pd.DataFrame(attn_map, index=pd.Index(query_labels), columns=pd.Index(key_labels)).to_csv(os.path.join(result_path, "attention_map.csv"))
    plt.close()


def compute_case_proportions(model, dataset, device, batch_size, result_path, case_names):
    assert len(dataset) == len(case_names), "Dataset size must match case names size"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prefix_dict = {}
    
    idx_base = 0
    for batch in loader:
        if len(batch) >= 7:
            batch_seq, batch_track, _, x_pdo, x_antigen, x_stroma, x_immune = batch[:7]
        else:
            batch_seq, batch_track = batch[:2]
            x_pdo = torch.zeros(batch_seq.size(0), 1, dtype=batch_seq.dtype, device=device)
            x_antigen = torch.zeros(batch_seq.size(0), 1, dtype=batch_seq.dtype, device=device)
            x_stroma = torch.zeros(batch_seq.size(0), 4, dtype=batch_seq.dtype, device=device)
            x_immune = torch.zeros(batch_seq.size(0), 2, dtype=batch_seq.dtype, device=device)
        batch_seq = batch_seq.to(device)
        batch_track = batch_track.to(device)
        x_pdo = x_pdo.to(device)
        x_antigen = x_antigen.to(device)
        x_stroma = x_stroma.to(device)
        x_immune = x_immune.to(device)
        logits = model(
            batch_seq,
            batch_track,
            x_pdo,
            x_antigen,
            x_stroma=x_stroma,
            x_immune=x_immune,
        )
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        n = len(pred)
        for i in range(n):
            case_id = case_names[idx_base + i] if case_names and (idx_base + i) < len(case_names) else "All"
            if case_id not in prefix_dict:
                prefix_dict[case_id] = [0, 0, 0]
            prefix_dict[case_id][int(pred[i])] += 1
        idx_base += n

    df = pd.DataFrame(prefix_dict).transpose()
    df.columns = ['Progressive', 'Stable', 'Responsive']
    df = df.div(df.sum(axis=1), axis=0).sort_index()

    ax = df.plot(kind='barh', stacked=True,
                 title='T cell Proportions by Case',
                 color=['#90BFF9', '#FFC080', '#FFA0A0'])
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    plt.savefig(os.path.join(result_path, "proportions_by_case.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(result_path, "proportions_by_case.svg"), dpi=300, bbox_inches="tight")
    df.to_csv(os.path.join(result_path, "proportions_by_case.csv"), index=True)
    plt.close()
    return df

def correlate_with_size_change(df, annotations_path, result_path):
    size_dict = {}
    if str(annotations_path).lower().endswith(".json"):
        with open(annotations_path, "r") as f:
            cfg = json.load(f)
        meta = cfg.get("meta", {})
        size_dict = meta.get("size_change_by_case", {}) or {}
    else:
        size_df = pd.read_excel(annotations_path)
        size_dict = size_df.set_index("Case Name")["Size Change"].to_dict()

    x, y, cases = [], [], []
    for case_name in df.index:
        if case_name not in size_dict:
            continue
        y.append(df.loc[case_name, "Combined Score"])
        x.append(size_dict[case_name])
        cases.append(case_name)

    x, y = np.array(x), np.array(y)
    if x.size == 0 or y.size == 0:
        return None

    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    r2 = r2_score(y, y_pred)

    plt.scatter(x, y)
    plt.plot(x, y_pred, color="red", label=f"Best fit (R²={r2:.2f})")
    plt.xlabel("Change in PDO size")
    plt.ylabel("Score")
    plt.title("Score by Change in PDO size")
    plt.legend()
    plt.savefig(os.path.join(result_path, "Score by Change in PDO size.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(result_path, "Score by Change in PDO size.svg"), dpi=300, bbox_inches="tight")
    pd.DataFrame({"case_id": cases, "size_change": x, "score": y, "predicted_score": y_pred}).to_csv(
        os.path.join(result_path, "Score by Change in PDO size.csv"),
        index=False,
    )
    plt.close()
    return r2

def plot_loss_curve(train_losses, val_losses, test_losses, results_path):
    print("[STEP 2] Drawing Loss Graph...")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    #plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{results_path}/loss_curve.png")
    plt.savefig(f"{results_path}/loss_curve.svg")
    plt.close()
    print("[STEP 2] Finished Drawing Loss Graph...")

def plot_accuracies(train_accs, val_accs, test_accs, results_path):
    print("[STEP 2] Drawing Validation Accuracy Graph...")
    plt.plot(np.array(train_accs) * 100)
    plt.plot(np.array(val_accs) * 100)
    plt.plot(np.array(test_accs) * 100)
    plt.title("Validation Accuracy (%)")
    plt.savefig(f"{results_path}/val_accuracy.png")
    plt.savefig(f"{results_path}/val_accuracy.svg")
    plt.close()
    print("[STEP 2] Finished Drawing Validation Accuracy Graph...")
