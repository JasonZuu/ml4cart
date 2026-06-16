import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from sklearn.metrics import f1_score as _f1

from dynamics_model.config import MAX_EPOCHS, EARLY_STOP_PATIENCE
from dynamics_model.utils.results_utils import compute_metrics, plot_roc, plot_confusion_matrix, fusion_weight_analysis, compute_case_proportions, correlate_with_size_change
from dynamics_model.train_fn.focal_train_fn import run_inference
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ce_train_fn(
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
    weighted_sampling: bool = True,
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
        classes_unique = np.unique(y_train)
        class_counts = np.array([(y_train == c).sum() for c in classes_unique], dtype=np.float64)
        class_weights_sampling = 1.0 / np.maximum(class_counts, 1.0)
        sample_weights = np.array([class_weights_sampling[np.where(classes_unique == y)[0][0]] for y in y_train], dtype=np.float64)
        sampler = WeightedRandomSampler(sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False if sampler is not None else True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # classes = np.unique(y_train)
    # class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    # weight = torch.tensor(class_w, dtype=torch.float32, device=device)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if use_class_weight:
        classes = np.unique(y_train)
        class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight = torch.tensor(class_w, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=scheduler_patience)

    early_stop = 0
    best_model = None
    best_val_f1 = -1.0
    best_val_loss = float("inf")

    train_losses, val_losses = [] , []
    train_accs, val_accs = [] , []

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
                val_preds_all.append(preds.cpu().numpy())
                val_y_all.append(b_y.cpu().numpy())

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss = val_loss_sum / max(len(val_loader), 1)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        val_f1_epoch = None
        if val_preds_all and val_y_all:
            val_preds_epoch = np.concatenate(val_preds_all)
            val_y_epoch = np.concatenate(val_y_all)
            if val_y_epoch.size > 0:
                val_f1_epoch = _f1(val_y_epoch, val_preds_epoch, average="macro")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
        epoch_iter.set_postfix({"train_acc": f"{train_acc:.3f}", "val_acc": f"{val_acc:.3f}", "val_loss": f"{val_loss:.4f}"})
        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
            if val_f1_epoch is not None:
                log_payload["val_f1"] = val_f1_epoch
            wandb_run.log(log_payload, step=epoch)

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
    os.makedirs(train_results_path, exist_ok=True)
    val_results_path = os.path.join(result_path, f"plots/val results")
    os.makedirs(val_results_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(result_path, "best_model.pth"))

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
    df_train["Combined Score"] = (df_train["Progressive"]*0 + df_train["Stable"]*0.5 + df_train["Responsive"]*1.0)
    r2_train = correlate_with_size_change(df_train, test_train_split_annotation_path, train_results_path)
    df_val = compute_case_proportions(model, val_dataset, device, batch_size, val_results_path, case_names=split["val"].get("case_names", []))
    df_val["Combined Score"] = (df_val["Progressive"]*0 + df_val["Stable"]*0.5 + df_val["Responsive"]*1.0)
    r2_val = correlate_with_size_change(df_val, test_train_split_annotation_path, val_results_path)

    preds_val, probs_val = run_inference(
        model,
        x_seq_val,
        x_track_val,
        device,
        X_pdo=x_pdosize_val,
        X_antigen=x_antigen_val,
        X_stroma=data["val"]["x_stroma"],
        X_immune=data["val"]["x_immune"],
    )
    best_val_acc, val_f1, val_auc, y_val_bin = compute_metrics(y_val, preds_val, probs_val)
    plot_roc(y_val_bin, probs_val, val_results_path, n_classes=3)
    cm_val = plot_confusion_matrix(y_val, preds_val, le.classes_, val_results_path)

    fusion_weight_analysis(model, val_loader, device, val_results_path)

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
