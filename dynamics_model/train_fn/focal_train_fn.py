import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score as _f1
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize

from dynamics_model.config import MAX_EPOCHS, EARLY_STOP_PATIENCE
from dynamics_model.utils.results_utils import compute_metrics, plot_roc, plot_confusion_matrix, fusion_weight_analysis, compute_case_proportions, correlate_with_size_change
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split
from dynamics_model.utils.loss_fn import FocalLoss
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sanitize_r2(value):
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


def _collect_logits(model, loader, device):
    logits_list = []
    y_list = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) >= 7:
                batch_seq, batch_track, batch_y, b_pdo, b_antigen, b_stroma, b_immune = batch[:7]
            elif len(batch) == 6:
                batch_seq, batch_track, batch_y, b_pdo, b_antigen, b_tme = batch
                b_immune = b_tme[:, :2]
                b_stroma = b_tme[:, 2:6]
            elif len(batch) == 5:
                batch_seq, batch_track, batch_y, b_pdo, b_antigen = batch
                b_stroma = torch.zeros(batch_seq.size(0), 4, dtype=batch_seq.dtype)
                b_immune = torch.zeros(batch_seq.size(0), 2, dtype=batch_seq.dtype)
            elif len(batch) == 3:
                batch_seq, batch_track, batch_y = batch
                b_pdo = torch.zeros(batch_seq.size(0), 1, dtype=batch_seq.dtype)
                b_antigen = torch.zeros(batch_seq.size(0), 1, dtype=batch_seq.dtype)
                b_stroma = torch.zeros(batch_seq.size(0), 4, dtype=batch_seq.dtype)
                b_immune = torch.zeros(batch_seq.size(0), 2, dtype=batch_seq.dtype)
            else:
                continue
            batch_seq = batch_seq.to(device)
            batch_track = batch_track.to(device)
            batch_y = batch_y.to(device)
            b_pdo = b_pdo.to(device)
            b_antigen = b_antigen.to(device)
            b_stroma = b_stroma.to(device)
            b_immune = b_immune.to(device)
            logits = model(batch_seq, batch_track, b_pdo, b_antigen, b_stroma, b_immune)
            logits_list.append(logits.detach().cpu())
            y_list.append(batch_y.detach().cpu())
    if not logits_list:
        return None, None
    logits_val = torch.cat(logits_list, dim=0)
    y_val = torch.cat(y_list, dim=0).numpy()
    return logits_val, y_val


def optimize_decision_weights_for_f1(probs, y_true, w_min=0.1, w_max=10.0, n_steps=50):
    def objective(x):
        weights = np.clip(np.array(x, dtype=np.float32), w_min, w_max)
        preds = np.argmax(probs * weights, axis=1)
        f1 = float(_f1(y_true, preds, average="macro"))
        return -f1
    x0 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": max(50, n_steps * 4), "xatol": 1e-3, "fatol": 1e-4})
    weights = np.clip(np.array(res.x, dtype=np.float32), w_min, w_max)
    preds_opt = np.argmax(probs * weights, axis=1)
    opt_f1 = float(_f1(y_true, preds_opt, average="macro"))
    return {
        "w0": float(weights[0]),
        "w1": float(weights[1]),
        "w2": float(weights[2]),
        "opt_f1": opt_f1,
        "weights": weights.tolist(),
    }


def focal_train_fn(
    seq_path,
    track_path,
    result_path,
    test_train_split_annotation_path,
    model,
    learning_rate=1e-3,
    batch_size=256,
    max_epochs=MAX_EPOCHS,
    weight_decay=1e-3,
    label_smoothing=0.0,
    scheduler_factor=0.5,
    scheduler_patience=10,
    early_stop_patience=EARLY_STOP_PATIENCE,
    focal_gamma=2.0,
    weighted_sampling: bool = False,
    use_class_weight: bool = True,
    wandb_run=None,
):
    split = subset_split_by_case(seq_path, track_path, test_train_split_annotation_path)
    data = build_datasets_from_case_split(split)
    le = data["label_encoder"]
    x_seq_train = data["train"]["x_seq"]
    x_seq_val = data["val"]["x_seq"]
    x_track_train = data["train"]["x_track"]
    x_track_val = data["val"]["x_track"]
    x_pdosize_train = data["train"]["x_pdosize"]
    x_pdosize_val = data["val"]["x_pdosize"]
    x_antigen_train = data["train"]["x_antigen"]
    x_antigen_val = data["val"]["x_antigen"]
    y_train = data["train"]["y"]
    y_val = data["val"]["y"]
    train_dataset = data["train"]["dataset"]
    val_dataset = data["val"]["dataset"]

    sampler = None
    if weighted_sampling:
        train_labels = y_train
        classes_unique = np.unique(train_labels)
        class_counts = np.array([(train_labels == c).sum() for c in classes_unique], dtype=np.float64)
        class_weights_sampling = 1.0 / np.maximum(class_counts, 1.0)
        sample_weights = np.array([class_weights_sampling[np.where(classes_unique == y)[0][0]] for y in train_labels], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False if sampler is not None else True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    alpha = None
    if use_class_weight:
        classes = np.unique(y_train)
        class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        alpha = torch.tensor(class_w, dtype=torch.float32, device=device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha, gamma=focal_gamma, label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)

    early_stop = 0
    best_model = None
    best_val_loss = float("inf")

    train_losses, val_losses = [] , []
    train_accs, val_accs = [] , []
    epoch_metrics = []

    epoch_iter = tqdm(range(max_epochs), desc="epochs")
    for epoch in epoch_iter:
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for b_seq, b_track, b_y, b_pdo, b_antigen, b_stroma, b_immune in train_loader:
            b_seq = b_seq.to(device)
            b_track = b_track.to(device)
            b_y = b_y.to(device)
            b_pdo = b_pdo.to(device)
            b_antigen = b_antigen.to(device)
            b_stroma = b_stroma.to(device)
            b_immune = b_immune.to(device)
            optimizer.zero_grad()
            logits = model(b_seq, b_track, b_pdo, b_antigen, b_stroma, b_immune)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == b_y).sum().item()
            train_total += b_y.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_y_all = []
        val_probs_all = []
        with torch.no_grad():
            for b_seq, b_track, b_y, b_pdo, b_antigen, b_stroma, b_immune in val_loader:
                b_seq = b_seq.to(device)
                b_track = b_track.to(device)
                b_y = b_y.to(device)
                b_pdo = b_pdo.to(device)
                b_antigen = b_antigen.to(device)
                b_stroma = b_stroma.to(device)
                b_immune = b_immune.to(device)
                logits = model(b_seq, b_track, b_pdo, b_antigen, b_stroma, b_immune)
                loss = criterion(logits, b_y)
                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == b_y).sum().item()
                val_total += b_y.size(0)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                val_preds_all.append(preds.cpu().numpy())
                val_y_all.append(b_y.cpu().numpy())
                val_probs_all.append(probs)

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss = val_loss_sum / max(len(val_loader), 1)

        train_acc = train_correct / max(train_total, 1)

        val_acc = val_correct / max(val_total, 1)

        val_f1_epoch = None
        val_auc_epoch = None
        if val_preds_all and val_y_all:
            val_preds_epoch = np.concatenate(val_preds_all)
            val_y_epoch = np.concatenate(val_y_all)
            val_probs_epoch = np.concatenate(val_probs_all) if val_probs_all else None
            if val_y_epoch.size > 0:
                val_f1_epoch = _f1(val_y_epoch, val_preds_epoch, average="macro")
                if val_probs_epoch is not None:
                    _, _, val_auc_epoch, _ = compute_metrics(val_y_epoch, val_preds_epoch, val_probs_epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
        epoch_iter.set_postfix({"train_acc": f"{train_acc:.3f}", "val_acc": f"{val_acc:.3f}", "val_loss": f"{val_loss:.4f}"})
        epoch_row = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/acc": train_acc,
            "val/acc": val_acc,
            "val/f1": val_f1_epoch,
            "val/auc": val_auc_epoch,
        }
        epoch_metrics.append(epoch_row)
        if wandb_run is not None:
            wandb_run.log(epoch_row, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= early_stop_patience:
                break

    if best_model is not None:
        model.load_state_dict(best_model)
    train_results_path = os.path.join(result_path, f"plots/train results")
    val_results_path = os.path.join(result_path, f"plots/val results")
    os.makedirs(train_results_path, exist_ok=True)
    os.makedirs(val_results_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(result_path, "best_model.pth"))
    if epoch_metrics:
        with open(os.path.join(result_path, "epoch_metrics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(epoch_metrics)

    preds_train, probs_train = run_inference(
        model,
        x_seq_train,
        x_track_train,
        device,
        X_pdo=x_pdosize_train,
        X_antigen=x_antigen_train,
        X_stroma=data["train"]["x_stroma"],
        X_immune=data["train"]["x_immune"],
    )
    best_train_acc, train_f1, train_auc, y_train_bin = compute_metrics(y_train, preds_train, probs_train)
    plot_roc(y_train_bin, probs_train, train_results_path, n_classes=3)
    cm_train = plot_confusion_matrix(y_train, preds_train, le.classes_, train_results_path)

    fusion_weight_analysis(model, train_loader, device, train_results_path)

    df_train = compute_case_proportions(model, train_dataset, device, batch_size, train_results_path, case_names=split["train"].get("case_names", []))
    df_train["Combined Score"] = (df_train["Progressive"]*0.0 + df_train["Stable"]*0.5 + df_train["Responsive"]*1.0)
    r2_train = correlate_with_size_change(df_train, test_train_split_annotation_path, train_results_path)
    logits_val, y_val_np = _collect_logits(model, val_loader, device)
    if logits_val is None:
        raise ValueError("No validation samples found for tuning.")
    T_opt = 1.0
    probs_val = F.softmax(logits_val, dim=1).cpu().numpy()
    tune = optimize_decision_weights_for_f1(probs_val, y_val_np)
    w0, w1, w2 = tune["w0"], tune["w1"], tune["w2"]
    probs_val_weighted = probs_val * np.array([w0, w1, w2], dtype=np.float32)
    preds_val = np.argmax(probs_val_weighted, axis=1)
    best_val_acc, val_f1, val_auc, y_val_bin = compute_metrics(y_val_np, preds_val, probs_val)
    plot_roc(y_val_bin, probs_val, val_results_path, n_classes=3)
    cm_val = plot_confusion_matrix(y_val, preds_val, le.classes_, val_results_path)

    fusion_weight_analysis(model, val_loader, device, val_results_path)
    df_val = compute_case_proportions(model, val_dataset, device, batch_size, val_results_path, case_names=split["val"].get("case_names", []))
    df_val["Combined Score"] = (
        df_val["Progressive"] * 0.0
        + df_val["Stable"] * 0.5
        + df_val["Responsive"] * 1.0
    )
    r2_val = correlate_with_size_change(df_val, test_train_split_annotation_path, val_results_path)
    with open(os.path.join(val_results_path, "best_val_thresholds.json"), "w") as f:
        json.dump(
            {
                "temperature": float(T_opt),
                "weights": [float(w0), float(w1), float(w2)],
                "best_val_f1": float(val_f1),
                "best_val_acc": float(best_val_acc),
                "val_auc": float(val_auc),
            },
            f,
            indent=2,
        )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracy": train_accs,
        "val_accuracy": val_accs,
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "cm_train": cm_train.tolist(),
        "cm_val": cm_val.tolist(),
        "r2_train": r2_train,
        "r2_val": r2_val,
    }


def run_inference(model, X_seq, X_track, device, X_pdo=None, X_antigen=None, X_tme=None, X_stroma=None, X_immune=None):
    model.eval()
    with torch.no_grad():
        if X_pdo is None:
            X_pdo = torch.zeros(X_seq.size(0), 1, dtype=X_seq.dtype)
        if X_antigen is None:
            X_antigen = torch.zeros(X_seq.size(0), 1, dtype=X_seq.dtype)
        if X_tme is None:
            if X_stroma is not None and X_immune is not None:
                X_tme = torch.cat([X_immune, X_stroma], dim=1)
            else:
                X_tme = torch.zeros(X_seq.size(0), 6, dtype=X_seq.dtype)
        if X_stroma is None:
            X_stroma = X_tme[:, 2:6]
        if X_immune is None:
            X_immune = X_tme[:, :2]
        logits = model(
            X_seq.to(device),
            X_track.to(device),
            X_pdo.to(device),
            X_antigen.to(device),
            X_stroma.to(device),
            X_immune.to(device),
        )
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs
