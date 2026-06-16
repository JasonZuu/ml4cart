import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dynamics_model.config import DATASET_CONFIGS


_PDO_DEVICE_PATIENT_MAP = {
    "Device1": "NYU318-device1", "Device2": "NYU318-device2",
    "Device3": "NCI2-device1",   "Device4": "NCI2-device2",
    "Device5": "NCI6-device1",   "Device6": "NCI6-device2",
    "Device7": "NCI9-device1",   "Device8": "NCI9-device2",
}


def _load_annotation_mapping(path, folder):
    if folder == "CART1":
        df = pd.read_excel(path, sheet_name="Summary")
        id_series = df.iloc[:, 1].astype(str).str.strip()
        label_series = df["Score"].astype(float)
    elif folder == "CART2":
        df = pd.read_excel(path, sheet_name=0)
        id_series = df["Meso IL18 CAR T cells"].astype(str).str.strip()
        label_series = df["Score"].astype(float)
    elif folder == "CART3":
        df = pd.read_excel(path, sheet_name="Summary")
        id_series = df.iloc[:, 1].astype(str).str.strip()
        label_series = df["Score"].astype(float)
    elif folder == "PDO":
        df = pd.read_excel(path, sheet_name="Statistics")
        raw_ids = df["Name"].astype(str).str.replace(" ", "")
        id_series = raw_ids.map(_PDO_DEVICE_PATIENT_MAP)
        label_series = df["Score"].astype(float)
    elif folder == "Stroma1":
        df = pd.read_excel(path, sheet_name=0)
        raw_ids = df["Name"].astype(str).str.replace(" ", "")
        id_series = raw_ids.str.replace("_Stroma_", "_stroma")
        label_series = df["Score"].astype(float)
    elif folder == "Stroma2":
        df = pd.read_excel(path, sheet_name=0)
        df = df.dropna(subset=["Name"])
        raw_ids = df["Name"].astype(str).str.replace(" ", "")
        id_series = raw_ids.str.replace("_Stroma_", "_stroma")
        label_series = df["Score"].astype(float)
    else:
        df = pd.read_excel(path)
        if "Case Name" not in df.columns or "Score" not in df.columns:
            return {}
        id_series = df["Case Name"].astype(str).str.strip()
        label_series = df["Score"].astype(float)
    mapping = {}
    for device, label in zip(id_series, label_series):
        if pd.isna(device) or pd.isna(label):
            continue
        mapping[f"{folder}_{str(device).strip()}"] = float(label)
    return mapping


def _load_case_label_map_from_annotations():
    case_label_map = {}
    for cfg in DATASET_CONFIGS.values():
        annotation_path = cfg.get("annotation_path")
        data_folder = cfg.get("data_folder", "")
        if not annotation_path or not os.path.exists(annotation_path):
            continue
        folder = os.path.basename(os.path.normpath(data_folder))
        mapping = _load_annotation_mapping(annotation_path, folder)
        overlap = case_label_map.keys() & mapping.keys()
        if overlap:
            raise ValueError(f"Overlapping case labels found: {sorted(list(overlap))[:5]}")
        case_label_map.update(mapping)
    return case_label_map


def subset_split_by_case(seq_path, track_path, test_train_split_annotation_path):
    annotations_df = None
    train_cases, val_cases, test_cases = [], [], []
    pdo_map, antigen_map = {}, {}
    col1_map, col3_map, col4_map, ha_map, pd1_map, lag3_map = {}, {}, {}, {}, {}, {}
    case_label_map = _load_case_label_map_from_annotations()

    if str(test_train_split_annotation_path).lower().endswith(".json"):
        with open(test_train_split_annotation_path, "r") as f:
            cfg = json.load(f)
        splits = cfg.get("splits", cfg)
        train_cases = splits.get("train", []) or []
        val_cases = splits.get("val", []) or []
        test_cases = splits.get("test", []) or []
        meta = cfg.get("meta", {})
        json_label_map = (
            meta.get("label_by_case")
            or meta.get("labels_by_case")
            or meta.get("response_label_by_case")
            or cfg.get("label_by_case")
            or {}
        )
        case_label_map.update({str(k): v for k, v in json_label_map.items()})
        pdo_map = meta.get("pdo_size", {})
        antigen_map = meta.get("antigen", {})
        col1_map = meta.get("COL-I", {})
        col3_map = meta.get("COL-III", {})
        col4_map = meta.get("COL-IV", {})
        ha_map = meta.get("HA", {})
        pd1_map = meta.get("PD-1", {})
        lag3_map = meta.get("LAG-3", {})
    else:
        annotations_df = pd.read_excel(test_train_split_annotation_path)
        train_cases = annotations_df.loc[annotations_df["Test Set"] == 0, "Case Name"].tolist()
        test_cases = annotations_df.loc[annotations_df["Test Set"] == 1, "Case Name"].tolist()
        val_cases = annotations_df.loc[annotations_df["Test Set"] == 2, "Case Name"].tolist()

        pdo_map = pd.Series(annotations_df["PDO Size Day5"].values, index=annotations_df["Case Name"]).to_dict()
        antigen_map = pd.Series(annotations_df["Targeted antigen expression (%)"].values, index=annotations_df["Case Name"]).to_dict()
        
        def _norm(s):
            return str(s).strip().lower().replace(" ", "").replace("-", "").replace("%", "")
        
        def _find(name):
            target = _norm(name)
            for c in annotations_df.columns:
                if _norm(c) == target:
                    return c
            return None
        
        _col1 = _find("COL-I")
        _col3 = _find("COL-III")
        _col4 = _find("COL-IV")
        _ha = _find("HA")
        _pd1 = _find("PD-1")
        _lag3 = _find("LAG-3")
        col1_map = pd.Series(annotations_df[_col1].values, index=annotations_df["Case Name"]).to_dict() if _col1 else {}
        col3_map = pd.Series(annotations_df[_col3].values, index=annotations_df["Case Name"]).to_dict() if _col3 else {}
        col4_map = pd.Series(annotations_df[_col4].values, index=annotations_df["Case Name"]).to_dict() if _col4 else {}
        ha_map = pd.Series(annotations_df[_ha].values, index=annotations_df["Case Name"]).to_dict() if _ha else {}
        pd1_map = pd.Series(annotations_df[_pd1].values, index=annotations_df["Case Name"]).to_dict() if _pd1 else {}
        lag3_map = pd.Series(annotations_df[_lag3].values, index=annotations_df["Case Name"]).to_dict() if _lag3 else {}

    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, track_ids_seq = seq_data["X"], seq_data["track_ids"]
    X_track, track_ids_track = track_data["X"], track_data["track_ids"]

    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {
        tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
        for i, tid in enumerate(track_ids_track)
    }

    X_seq_train, X_track_train, y_train = [], [], []
    X_seq_test, X_track_test, y_test = [], [], []
    X_seq_val, X_track_val, y_val = [], [], []
    case_names_train, case_names_test, case_names_val = [], [], []
    missing_label_cases = set()

    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            case_name = "_".join(tid[0].split("_")[:-1])
            label = case_label_map.get(case_name, np.nan)
            if case_name in test_cases:
                if pd.isna(label):
                    missing_label_cases.add(case_name)
                    continue
                X_seq_test.append(X_seq[i])
                X_track_test.append(X_track[idx])
                y_test.append(label)
                case_names_test.append(case_name)
            elif case_name in val_cases:
                if pd.isna(label):
                    missing_label_cases.add(case_name)
                    continue
                X_seq_val.append(X_seq[i])
                X_track_val.append(X_track[idx])
                y_val.append(label)
                case_names_val.append(case_name)
            elif case_name in train_cases:
                if pd.isna(label):
                    missing_label_cases.add(case_name)
                    continue
                X_seq_train.append(X_seq[i])
                X_track_train.append(X_track[idx])
                y_train.append(label)
                case_names_train.append(case_name)

    if missing_label_cases:
        sample_cases = sorted(list(missing_label_cases))[:10]
        raise ValueError(f"Missing labels in annotation files for cases: {sample_cases} (total {len(missing_label_cases)})")

    return {
        "train": {
            "X_seq": np.array(X_seq_train),
            "X_track": np.array(X_track_train),
            "y": np.array(y_train),
            "case_names": case_names_train,
        },
        "val": {
            "X_seq": np.array(X_seq_val),
            "X_track": np.array(X_track_val),
            "y": np.array(y_val),
            "case_names": case_names_val,
        },
        "meta": {
            "pdo_size": pdo_map,
            "antigen": antigen_map,
            "COL-I": col1_map,
            "COL-III": col3_map,
            "COL-IV": col4_map,
            "HA": ha_map,
            "PD-1": pd1_map,
            "LAG-3": lag3_map,
        },
    }


def normalize_dataset(X_seq, X_track, seq_scaler=None, track_scaler=None):
    def transform_seq(X_seq, scaler):
        if X_seq is None:
            return None
        if getattr(X_seq, "size", 0) == 0 or getattr(X_seq, "ndim", 0) < 3:
            return X_seq
        n_samples, n_timesteps, n_features_seq = X_seq.shape
        X_flat = X_seq.reshape(-1, n_features_seq)
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(n_samples, n_timesteps, n_features_seq)

    def transform_track(X_track, scaler):
        if X_track is None:
            return None
        if getattr(X_track, "size", 0) == 0:
            return X_track
        return scaler.transform(X_track)

    if not seq_scaler:
        seq_scaler = StandardScaler()
        n_samples, n_timesteps, n_features_seq = X_seq.shape
        X_seq_train_flat = X_seq.reshape(-1, n_features_seq)
        seq_scaler.fit(X_seq_train_flat)

    if not track_scaler:
        track_scaler = StandardScaler()
        track_scaler.fit(X_track)

    X_seq_scaled = transform_seq(X_seq, seq_scaler)
    X_track_scaled = transform_track(X_track, track_scaler)
    return X_seq_scaled, X_track_scaled, seq_scaler, track_scaler


def _fit_label_encoder(train_labels, val_labels):
    le = LabelEncoder()
    train_arr = np.asarray(train_labels)
    val_arr = np.asarray(val_labels)
    observed = np.concatenate([arr.reshape(-1) for arr in (train_arr, val_arr) if arr.size > 0])
    if observed.size:
        numeric = observed.astype(float)
        fixed_three_class = (
            np.all(np.isfinite(numeric))
            and np.allclose(numeric, np.round(numeric))
            and set(numeric.astype(int).tolist()).issubset({0, 1, 2})
        )
        if fixed_three_class:
            le.fit(np.array([0, 1, 2]))
        else:
            le.fit(observed)
    else:
        le.fit(train_arr)
    return le


def build_datasets_from_case_split(split, batch_size=None):
    le = _fit_label_encoder(split["train"]["y"], split["val"]["y"])
    y_train = le.transform(split["train"]["y"]) if split["train"]["y"].size > 0 else split["train"]["y"]
    y_val = le.transform(split["val"]["y"]) if split["val"]["y"].size > 0 else split["val"]["y"]

    X_seq_train, X_track_train = split["train"]["X_seq"], split["train"]["X_track"]
    X_seq_val, X_track_val = split["val"]["X_seq"], split["val"]["X_track"]

    X_seq_train, X_track_train, seq_scaler, track_scaler = normalize_dataset(X_seq_train, X_track_train)
    X_seq_val, X_track_val, _, _ = normalize_dataset(X_seq_val, X_track_val, seq_scaler, track_scaler)

    x_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    x_seq_val = torch.tensor(X_seq_val, dtype=torch.float32)

    x_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    x_track_val = torch.tensor(X_track_val, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    meta = split.get("meta", {})
    pdo_map = meta.get("pdo_size", {})
    antigen_map = meta.get("antigen", {})
    col1_map = meta.get("COL-I", {})
    col3_map = meta.get("COL-III", {})
    col4_map = meta.get("COL-IV", {})
    ha_map = meta.get("HA", {})
    pd1_map = meta.get("PD-1", {})
    lag3_map = meta.get("LAG-3", {})
    cn_train = split["train"].get("case_names", [])
    cn_val = split["val"].get("case_names", [])

    pdo_train = np.array([pdo_map.get(c, np.nan) for c in cn_train], dtype=np.float32).reshape(-1,1)
    antigen_train = np.array([antigen_map.get(c, np.nan) for c in cn_train], dtype=np.float32).reshape(-1,1)
    pdo_med = float(np.nanmedian(pdo_train)) if pdo_train.size else 0.0
    antigen_med = float(np.nanmedian(antigen_train)) if antigen_train.size else 0.0
    pdo_train = np.nan_to_num(pdo_train, nan=pdo_med)
    antigen_train = np.nan_to_num(antigen_train, nan=antigen_med)

    tme_train = np.array(
        [[lag3_map.get(c, np.nan), pd1_map.get(c, np.nan), col1_map.get(c, np.nan), col3_map.get(c, np.nan), col4_map.get(c, np.nan), ha_map.get(c, np.nan)] for c in cn_train],
        dtype=np.float32,
    )
    tme_med = np.nanmedian(tme_train, axis=0) if tme_train.size else np.zeros(6, dtype=np.float32)
    tme_train = np.where(np.isnan(tme_train), tme_med, tme_train)

    pdo_scaler = StandardScaler(); pdo_scaler.fit(pdo_train); pdo_train = pdo_scaler.transform(pdo_train)
    antigen_scaler = StandardScaler(); antigen_scaler.fit(antigen_train); antigen_train = antigen_scaler.transform(antigen_train)
    tme_scaler = StandardScaler(); tme_scaler.fit(tme_train); tme_train = tme_scaler.transform(tme_train)

    def _scale_meta(cases, scaler, mapping, med):
        arr = np.array([mapping.get(c, np.nan) for c in cases], dtype=np.float32).reshape(-1,1)
        arr = np.nan_to_num(arr, nan=med)
        if arr.size == 0:
            return arr
        return scaler.transform(arr)
    def _scale_multi(cases, scaler, maps, med):
        arr = np.array([[m.get(c, np.nan) for m in maps] for c in cases], dtype=np.float32)
        if arr.size == 0:
            return arr
        arr = np.where(np.isnan(arr), med, arr)
        return scaler.transform(arr)

    pdo_val = _scale_meta(cn_val, pdo_scaler, pdo_map, pdo_med)
    antigen_val = _scale_meta(cn_val, antigen_scaler, antigen_map, antigen_med)

    tme_maps = [lag3_map, pd1_map, col1_map, col3_map, col4_map, ha_map]
    tme_val = _scale_multi(cn_val, tme_scaler, tme_maps, tme_med)

    x_pdosize_train = torch.tensor(pdo_train, dtype=torch.float32)
    x_pdosize_val = torch.tensor(pdo_val, dtype=torch.float32)
    x_antigen_train = torch.tensor(antigen_train, dtype=torch.float32)
    x_antigen_val = torch.tensor(antigen_val, dtype=torch.float32)

    x_tme_train = torch.tensor(tme_train, dtype=torch.float32)
    x_tme_val = torch.tensor(tme_val, dtype=torch.float32)
    x_immune_train = x_tme_train[:, :2]
    x_immune_val = x_tme_val[:, :2]
    x_stroma_train = x_tme_train[:, 2:6]
    x_stroma_val = x_tme_val[:, 2:6]

    train_dataset = TensorDataset(x_seq_train, x_track_train, y_train_tensor, x_pdosize_train, x_antigen_train, x_stroma_train, x_immune_train)
    val_dataset = TensorDataset(x_seq_val, x_track_val, y_val_tensor, x_pdosize_val, x_antigen_val, x_stroma_val, x_immune_val)

    return {
        "label_encoder": le,
        "scalers": {"seq": seq_scaler, "track": track_scaler, "pdo_size": pdo_scaler, "antigen": antigen_scaler, "tme": tme_scaler},
        "train": {
            "x_seq": x_seq_train,
            "x_track": x_track_train,
            "x_pdosize": x_pdosize_train,
            "x_antigen": x_antigen_train,
            "x_stroma": x_stroma_train,
            "x_immune": x_immune_train,
            "x_tme": x_tme_train,
            "y": y_train,
            "dataset": train_dataset,
        },
        "val": {
            "x_seq": x_seq_val,
            "x_track": x_track_val,
            "x_pdosize": x_pdosize_val,
            "x_antigen": x_antigen_val,
            "x_stroma": x_stroma_val,
            "x_immune": x_immune_val,
            "x_tme": x_tme_val,
            "y": y_val,
            "dataset": val_dataset,
        }
    }
