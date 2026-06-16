import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELS_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
WORKBOOK_NS = {
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build pdo_change_label.json from On-chip PDO size xlsx files."
    )
    parser.add_argument(
        "--base-dir",
        default="data/On-chip_Data",
        help="On-chip base directory containing Chip-Rx_WSI and image_id_mapping.json",
    )
    parser.add_argument(
        "--image-id-mapping",
        default=None,
        help="Optional explicit path to image_id_mapping.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path for pdo_change_label.json",
    )
    return parser.parse_args(argv)


def _col_letters_to_index(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx


def _cell_ref_to_col_index(cell_ref: str) -> int | None:
    m = re.match(r"^([A-Z]+)[0-9]+$", cell_ref or "")
    if not m:
        return None
    return _col_letters_to_index(m.group(1))


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    shared = []
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return shared
    root = ET.fromstring(zf.read(name))
    for si in root:
        txt = "".join(t.text or "" for t in si.iter(f"{XML_NS}t"))
        shared.append(txt)
    return shared


def _read_sheet_rows(zf: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[dict[int, str]]:
    root = ET.fromstring(zf.read(sheet_path))
    sheet_data = root.find(f"{XML_NS}sheetData")
    rows: list[dict[int, str]] = []
    if sheet_data is None:
        return rows
    for row in sheet_data.findall(f"{XML_NS}row"):
        row_map: dict[int, str] = {}
        for cell in row.findall(f"{XML_NS}c"):
            cell_ref = cell.attrib.get("r")
            col_idx = _cell_ref_to_col_index(cell_ref) if cell_ref else None
            if col_idx is None:
                continue
            t = cell.attrib.get("t")
            v = cell.find(f"{XML_NS}v")
            if v is None or v.text is None:
                continue
            value = v.text
            if t == "s":
                if value.isdigit():
                    s_idx = int(value)
                    if 0 <= s_idx < len(shared_strings):
                        value = shared_strings[s_idx]
            row_map[col_idx] = str(value).strip()
        rows.append(row_map)
    return rows


def _normalize_patient(value: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"^\d+\.\s*", "", text)
    m = re.search(r"([A-Za-z]+)\s*-?\s*(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    prefix = m.group(1).lower()
    number = m.group(2).replace(".", "")
    return f"{prefix}{number}"


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_patient_to_change_from_xlsx(xlsx_path: Path) -> dict[str, float]:
    with zipfile.ZipFile(xlsx_path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in rels.findall("r:Relationship", RELS_NS)
        }
        shared_strings = _read_shared_strings(zf)
        patient_to_change: dict[str, float] = {}
        sheets_node = workbook.find("x:sheets", WORKBOOK_NS)
        if sheets_node is None:
            return patient_to_change
        for sheet in sheets_node:
            rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            if rid is None or rid not in rid_to_target:
                continue
            sheet_path = rid_to_target[rid]
            if not sheet_path.startswith("xl/"):
                sheet_path = f"xl/{sheet_path}"
            rows = _read_sheet_rows(zf, sheet_path, shared_strings)
            header_row: dict[int, str] | None = None
            value_row: dict[int, str] | None = None
            for row in rows:
                candidates = 0
                for v in row.values():
                    if _normalize_patient(v) is not None:
                        candidates += 1
                if candidates >= 3:
                    header_row = row
                    break
            for row in rows:
                has_label = False
                numeric_cells = 0
                for v in row.values():
                    t = str(v).strip().lower()
                    if t.startswith("pdo size change"):
                        has_label = True
                    if _parse_float(str(v)) is not None:
                        numeric_cells += 1
                if has_label and numeric_cells >= 3:
                    value_row = row
                    break
            if header_row is None or value_row is None:
                continue
            col_to_patient = {}
            for col, value in header_row.items():
                patient = _normalize_patient(value)
                if patient is not None:
                    col_to_patient[col] = patient
            for col, patient in col_to_patient.items():
                change = _parse_float(value_row.get(col))
                if change is not None:
                    patient_to_change[patient] = change
        return patient_to_change


def _extract_r4_labels_from_xlsx(xlsx_path: Path) -> dict[tuple[str, str], float]:
    """Parse R4 Excel 2-column layout: col A = '{patient}_{drug}', col B = pdo_change.

    Returns {("nyu{patient_num}", drug_lower): pdo_change}.
    """
    KNOWN_DRUGS = {"igg", "iareg", "fap"}
    label_pat = re.compile(r"^(\d+)_(.+)$")
    result: dict[tuple[str, str], float] = {}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rid_to_target = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in rels.findall("r:Relationship", RELS_NS)
        }
        sheets_node = workbook.find("x:sheets", WORKBOOK_NS)
        if sheets_node is None:
            return result
        for sheet in sheets_node:
            rid = sheet.attrib.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            if rid not in rid_to_target:
                continue
            sheet_path = rid_to_target[rid]
            if not sheet_path.startswith("xl/"):
                sheet_path = f"xl/{sheet_path}"
            rows = _read_sheet_rows(zf, sheet_path, shared_strings)
            for row in rows:
                label = row.get(1)   # column A (index 1)
                value_str = row.get(2)  # column B (index 2)
                if not label or value_str is None:
                    continue
                m = label_pat.match(label.strip())
                if not m:
                    continue
                patient_num = m.group(1)
                drug_lower = m.group(2).lower()
                if drug_lower not in KNOWN_DRUGS:
                    continue
                change = _parse_float(value_str)
                if change is None:
                    continue
                result[(f"nyu{patient_num}", drug_lower)] = change
    return result


def _extract_r4_patient_drug_key(image_id: str) -> tuple[str, str] | None:
    """For 'chip-r4_nyu285_fap-d1' → ('nyu285', 'fap')."""
    m = re.match(r"^chip-r4_(nyu\d+)_([a-z]+)-d\d+$", image_id.lower())
    if not m:
        return None
    return m.group(1), m.group(2)


def _extract_round_key(image_id: str) -> str | None:
    m = re.match(r"^(chip-r(\d+))_", image_id.lower())
    if not m:
        return None
    return f"R{m.group(2)}"


def _extract_patient_key(image_id: str) -> str | None:
    low = image_id.lower()
    m = re.match(r"^chip-r\d+_([a-z0-9]+?)-d\d+$", low)
    if m:
        return m.group(1)
    m = re.match(r"^chip-r\d+_([a-z0-9]+)", low)
    if m:
        return m.group(1)
    return None


def _collect_round_xlsx(base_dir: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for mask_dir in sorted(base_dir.glob("Chip-R*_mask")):
        m = re.match(r"^Chip-(R\d+)_mask$", mask_dir.name)
        if not m:
            continue
        round_key = m.group(1).upper()
        candidates = [
            p for p in mask_dir.glob("*.xlsx")
            if "pdo" in p.name.lower() and "size" in p.name.lower()
        ]
        if not candidates:
            continue
        out[round_key] = sorted(candidates, key=lambda p: p.name.lower())
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.base_dir)
    mapping_path = Path(args.image_id_mapping) if args.image_id_mapping else base_dir / "image_id_mapping.json"
    output_path = Path(args.output) if args.output else base_dir / "pdo_change_label.json"

    if not mapping_path.exists():
        raise FileNotFoundError(f"image_id_mapping file not found: {mapping_path}")

    round_to_xlsx = _collect_round_xlsx(base_dir)
    if not round_to_xlsx:
        raise FileNotFoundError(f"No PDO size xlsx found under: {base_dir}")

    round_to_patient_change: dict[str, dict[str, float]] = {}
    r4_labels: dict[tuple[str, str], float] = {}
    for round_key, xlsx_paths in round_to_xlsx.items():
        if round_key == "R4":
            for xlsx_path in xlsx_paths:
                r4_labels.update(_extract_r4_labels_from_xlsx(xlsx_path))
        else:
            patient_to_change: dict[str, float] = {}
            for xlsx_path in xlsx_paths:
                patient_to_change.update(_extract_patient_to_change_from_xlsx(xlsx_path))
            if patient_to_change:
                round_to_patient_change[round_key] = patient_to_change

    with mapping_path.open("r", encoding="utf-8") as f:
        image_id_mapping = json.load(f)

    pdo_change_label: dict[str, float] = {}
    for image_id in image_id_mapping:
        if image_id.lower().startswith("chip-r4_"):
            keys = _extract_r4_patient_drug_key(image_id)
            if keys is None:
                continue
            change = r4_labels.get(keys)
            if change is not None:
                pdo_change_label[image_id] = change
            continue
        round_key = _extract_round_key(image_id)
        patient_key = _extract_patient_key(image_id)
        if round_key is None or patient_key is None:
            continue
        patient_to_change = round_to_patient_change.get(round_key, {})
        if patient_key not in patient_to_change:
            continue
        pdo_change_label[image_id] = patient_to_change[patient_key]

    output_path.write_text(json.dumps(pdo_change_label, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved {len(pdo_change_label)} labels -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
