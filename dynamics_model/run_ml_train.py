import os
import json
import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from dynamics_model.config import SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH, RESULTS_DIR
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split
from dynamics_model.utils.results_utils import compute_metrics, plot_roc, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["svm", "xgb", "logits_regression", "rf"], default="logits_regression")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--split_json", type=str, default=TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    return parser.parse_args()


def _concat_struct_features(split_part):
    x_seq_full = split_part["x_seq"].cpu().numpy()
    x_seq_mean = np.mean(x_seq_full, axis=1)
    x_seq_max = np.max(x_seq_full, axis=1)
    x_seq = np.concatenate([x_seq_mean, x_seq_max], axis=1)
    x_track = split_part["x_track"].cpu().numpy()
    x_pdo = split_part["x_pdosize"].cpu().numpy()
    x_antigen = split_part["x_antigen"].cpu().numpy()
    x_stroma = split_part["x_stroma"].cpu().numpy()
    x_immune = split_part["x_immune"].cpu().numpy()
    X = np.concatenate([x_seq, x_track, x_pdo, x_antigen, x_stroma, x_immune], axis=1)
    y = np.array(split_part["y"])
    return X, y


def _grid_params(method):
    if method == "svm":
        return [
            {"C": c, "gamma": g}
            for c in [0.1, 1.0, 10.0]
            for g in ["scale", "auto"]
        ]
    if method == "logits_regression":
        return [
            {"C": c}
            for c in [0.1, 1.0, 10.0]
        ]
    if method == "xgb":
        return [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n in [100, 300, 500]
            for d in [3, 5]
            for lr in [0.1, 0.05, 0.01]
        ]
    if method == "rf":
        return [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
            for n in [200, 500, 800]
            for d in [None, 10, 20]
            for leaf in [1, 3, 5]
        ]
    raise ValueError(f"Unknown ML method: {method}")


def _build_model(method, params):
    if method == "svm":
        return SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    if method == "logits_regression":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=1,
        )
    if method == "xgb":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            objective="multi:softprob",
            num_class=3,
            random_state=1,
            n_jobs=4,
        )
    if method == "rf":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=1,
            n_jobs=4,
        )
    raise ValueError(f"Unknown ML method: {method}")


if __name__ == "__main__":
    args = parse_args()
    log_dir = os.path.join(args.output_dir, args.method)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    train_results_path = os.path.join(log_dir, "plots/train results")
    val_results_path = os.path.join(log_dir, "plots/val results")
    Path(train_results_path).mkdir(parents=True, exist_ok=True)
    Path(val_results_path).mkdir(parents=True, exist_ok=True)

    split = subset_split_by_case(SEQ_DATASET_PATH, TRACK_DATASET_PATH, args.split_json)
    data = build_datasets_from_case_split(split)

    X_train, y_train = _concat_struct_features(data["train"])
    X_val, y_val = _concat_struct_features(data["val"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    best = {
        "val_auc": float("-inf"),
        "params": None,
        "model": None,
        "val_acc": None,
        "val_f1": None,
    }

    for params in _grid_params(args.method):
        clf = _build_model(args.method, params)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc, val_f1, val_auc, _ = compute_metrics(y_val, val_preds, val_probs)
        if val_auc > best["val_auc"]:
            best.update(
                {
                    "val_auc": float(val_auc),
                    "params": params,
                    "model": clf,
                    "val_acc": float(val_acc),
                    "val_f1": float(val_f1),
                }
            )

    if best["model"] is None:
        raise ValueError("Grid search failed to produce a valid model.")

    train_probs = best["model"].predict_proba(X_train)
    train_preds = np.argmax(train_probs, axis=1)
    train_acc, train_f1, train_auc, y_train_bin = compute_metrics(y_train, train_preds, train_probs)
    plot_roc(y_train_bin, train_probs, train_results_path, n_classes=3)
    plot_confusion_matrix(y_train, train_preds, classes=np.unique(y_train), result_path=train_results_path)

    val_probs = best["model"].predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc, val_f1, val_auc, y_val_bin = compute_metrics(y_val, val_preds, val_probs)
    plot_roc(y_val_bin, val_probs, val_results_path, n_classes=3)
    plot_confusion_matrix(y_val, val_preds, classes=np.unique(y_train), result_path=val_results_path)

    with open(os.path.join(log_dir, "best_params.json"), "w") as f:
        json.dump(best["params"], f, indent=2)

    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "train_acc": float(train_acc),
                "train_f1": float(train_f1),
                "train_auc": float(train_auc),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
                "val_auc": float(val_auc),
            },
            f,
            indent=2,
        )

    joblib.dump(
        {
            "method": args.method,
            "params": best["params"],
            "scaler": scaler,
            "model": best["model"],
        },
        os.path.join(log_dir, "ml_model.joblib"),
    )
