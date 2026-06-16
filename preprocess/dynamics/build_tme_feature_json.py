import argparse
import json
from pathlib import Path

import pandas as pd


PATIENT_ORDER = [
    "NCI2",
    "NCI6",
    "NCI8",
    "NCI9",
    "NYU318",
    "NYU352",
    "NYU358",
    "NYU360",
]


def _to_cart_id(patient_id):
    pid = str(patient_id).strip()
    if pid.startswith("CART_"):
        return pid
    return f"CART_{pid}"


def _to_number(v):
    if pd.isna(v):
        return None
    try:
        n = float(v)
        if n.is_integer():
            return int(n)
        return n
    except Exception:
        return str(v).strip()


def _empty_patient_map():
    return {_to_cart_id(pid): None for pid in PATIENT_ORDER}


def _normalize_feature_name(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def _find_patient_header_row(df, patient_ids):
    patient_set = set(patient_ids)
    for r in range(df.shape[0]):
        row_vals = []
        for c in range(df.shape[1]):
            v = df.iat[r, c]
            if pd.isna(v):
                continue
            row_vals.append(str(v).strip())
        matched = [v for v in row_vals if v in patient_set]
        if len(matched) >= 4:
            return r
    return None


def _add_feature(result, feature_name, sheet_name, patient_values):
    if feature_name not in result:
        result[feature_name] = patient_values
        return
    existing = result[feature_name]
    has_conflict = False
    for pid in existing:
        ev = existing[pid]
        nv = patient_values[pid]
        if ev is not None and nv is not None and ev != nv:
            has_conflict = True
            break
    if has_conflict:
        result[f"{sheet_name}::{feature_name}"] = patient_values
        return
    merged = {}
    for pid in existing:
        merged[pid] = existing[pid] if existing[pid] is not None else patient_values[pid]
    result[feature_name] = merged


def _extract_sheet_features(df, sheet_name):
    header_row = _find_patient_header_row(df, PATIENT_ORDER)
    if header_row is None:
        return {}
    patient_col_to_id = {}
    for c in range(df.shape[1]):
        v = df.iat[header_row, c]
        s = "" if pd.isna(v) else str(v).strip()
        if s in PATIENT_ORDER:
            patient_col_to_id[c] = s
    if len(patient_col_to_id) == 0:
        return {}
    out = {}
    seen_name_count = {}
    for r in range(header_row + 1, df.shape[0]):
        feature = _normalize_feature_name(df.iat[r, 0])
        if feature == "":
            continue
        row_map = _empty_patient_map()
        has_value = False
        for c, pid in patient_col_to_id.items():
            if c >= df.shape[1]:
                continue
            val = _to_number(df.iat[r, c])
            row_map[_to_cart_id(pid)] = val
            if val is not None:
                has_value = True
        if not has_value:
            continue
        seen_name_count[feature] = seen_name_count.get(feature, 0) + 1
        unique_feature = feature
        if seen_name_count[feature] > 1:
            unique_feature = f"{feature}_{seen_name_count[feature]}"
        out[unique_feature] = row_map
    return out


def build_tme_json(xlsx_path, output_path):
    xl = pd.ExcelFile(xlsx_path)
    result = {}
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
        sheet_features = _extract_sheet_features(df, sheet_name)
        for feature_name, patient_values in sheet_features.items():
            _add_feature(result, feature_name, sheet_name, patient_values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx_path",
        default="data/Multimodal Data/Multimodal data.xlsx",
    )
    parser.add_argument("--output_path", default="data/tme_feature.json")
    args = parser.parse_args()
    build_tme_json(Path(args.xlsx_path), Path(args.output_path))
    print(f"[INFO] Saved to {args.output_path}")


if __name__ == "__main__":
    main()
