import os
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from dynamics_model.config import SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH
from dynamics_model.dataset.load_data import subset_split_by_case, build_datasets_from_case_split
from dynamics_model.utils.results_utils import compute_metrics, plot_roc, plot_confusion_matrix


def _concat_struct_features(split_part):
    x_seq_full = split_part["x_seq"].cpu().numpy()
    x_seq_mean = np.mean(x_seq_full, axis=1)
    # x_seq_min = np.min(x_seq_full, axis=1)
    x_seq_max = np.max(x_seq_full, axis=1)
    x_seq = np.concatenate([x_seq_mean,x_seq_max], axis=1)
    x_track = split_part["x_track"].cpu().numpy()
    x_pdo = split_part["x_pdosize"].cpu().numpy()
    x_antigen = split_part["x_antigen"].cpu().numpy()
    if "x_tme" in split_part:
        x_tme = split_part["x_tme"].cpu().numpy()
    else:
        x_immune = split_part["x_immune"].cpu().numpy()
        x_ecm = split_part["x_ecm"].cpu().numpy()
        x_cytokine = split_part["x_cytokine"].cpu().numpy()
        x_tme = np.concatenate([x_immune, x_ecm, x_cytokine], axis=1)
    X = np.concatenate([x_seq, x_track, x_pdo, x_antigen, x_tme], axis=1)
    y = split_part["y"]
    return X, y


def ml_train_fn(method: str = "svm", result_path: str = "results/ml"):
    split = subset_split_by_case(SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    data = build_datasets_from_case_split(split)

    X_train, y_train = _concat_struct_features(data["train"])
    X_val, y_val = _concat_struct_features(data["val"])
    X_test, y_test = _concat_struct_features(data["test"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if method == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=42)
    elif method == "xgb":
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            random_state=1,
            n_jobs=4,
        )
    else:
        raise ValueError(f"Unknown ML method: {method}")

    clf.fit(X_train, y_train)

    os.makedirs(result_path, exist_ok=True)
    train_results_path = os.path.join(result_path, "plots/train results")
    val_results_path = os.path.join(result_path, "plots/val results")
    test_results_path = os.path.join(result_path, "plots/test results")
    os.makedirs(train_results_path, exist_ok=True)
    os.makedirs(val_results_path, exist_ok=True)
    os.makedirs(test_results_path, exist_ok=True)

    train_probs = clf.predict_proba(X_train)
    train_preds = np.argmax(train_probs, axis=1)
    best_train_acc, train_f1, train_auc, y_train_bin = compute_metrics(y_train, train_preds, train_probs)
    plot_roc(y_train_bin, train_probs, train_results_path, n_classes=3)
    plot_confusion_matrix(y_train, train_preds, classes=np.unique(y_train), result_path=train_results_path)

    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    best_val_acc, val_f1, val_auc, y_val_bin = compute_metrics(y_val, val_preds, val_probs)
    plot_roc(y_val_bin, val_probs, val_results_path, n_classes=3)
    plot_confusion_matrix(y_val, val_preds, classes=np.unique(y_train), result_path=val_results_path)

    test_probs = clf.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    best_test_acc, test_f1, test_auc, y_test_bin = compute_metrics(y_test, test_preds, test_probs)
    plot_roc(y_test_bin, test_probs, test_results_path, n_classes=3)
    plot_confusion_matrix(y_test, test_preds, classes=np.unique(y_train), result_path=test_results_path)

    print("Validation report:\n" + classification_report(y_val, val_preds, digits=4))
    print("Test report:\n" + classification_report(y_test, test_preds, digits=4))

    return {
        "best_train_acc": best_train_acc,
        "train_f1": train_f1,
        "train_auc": train_auc,
        "best_val_acc": best_val_acc,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "best_test_acc": best_test_acc,
        "test_f1": test_f1,
        "test_auc": test_auc,
    }
