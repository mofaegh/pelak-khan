#!/usr/bin/env python
"""
Pelak-Khan dataset deep audit.

Audits:
- hezarai_persian_plate_ocr: CSV labels, alphabet, lengths, missing files
- imlp: VOC/XML plate bounding boxes and image/XML pairing
- ir_lpr: per-character XML boxes, class distribution, reconstructed labels
- iranis: character-class folder distribution
- ir_lpr_corners: 4-corner CSV validation
- sivd: image grouping to distinguish useful dataset areas from repository/code assets

Optional expensive checks:
    --verify-images       Open every image with Pillow and verify it.
    --hash-duplicates     SHA-256 every image and report exact duplicates.

Outputs are written to:
    artifacts/dataset_audit/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_DATASETS = [
    "hezarai_persian_plate_ocr",
    "imlp",
    "iranis",
    "ir_lpr",
    "ir_lpr_corners",
    "sivd",
]

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    ASCII_DIGITS + ASCII_DIGITS,
)

CHAR_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
    }
)

INVISIBLE_CHARS = {
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\ufeff",  # BOM
}


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_plate_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")
    text = text.translate(DIGIT_TRANSLATION).translate(CHAR_TRANSLATION)
    text = "".join(ch for ch in text if not ch.isspace())
    return text


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
        return
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def file_inventory(root: Path) -> dict[str, Any]:
    file_count = 0
    total_bytes = 0
    extensions: Counter[str] = Counter()
    image_count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        file_count += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
        ext = path.suffix.lower() or "<no_extension>"
        extensions[ext] += 1
        if ext in IMAGE_EXTS:
            image_count += 1

    return {
        "files": file_count,
        "images": image_count,
        "bytes": total_bytes,
        "gb": round(total_bytes / (1024**3), 3),
        "extensions": dict(extensions.most_common()),
    }


def build_image_name_index(root: Path) -> tuple[dict[str, Path], set[str]]:
    index: dict[str, Path] = {}
    ambiguous: set[str] = set()

    for image_path in iter_images(root):
        key = image_path.name.lower()
        if key in index:
            ambiguous.add(key)
        else:
            index[key] = image_path

    return index, ambiguous


def verify_images(root: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    images = list(iter_images(root))

    for path in tqdm(images, desc=f"Verify {root.name}", unit="img"):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as exc:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "corrupt_image",
                    "path": safe_rel(path, root),
                    "details": repr(exc),
                }
            )
    return problems


def hash_duplicates(root: Path) -> list[dict[str, Any]]:
    hashes: dict[str, list[Path]] = defaultdict(list)
    images = list(iter_images(root))

    for path in tqdm(images, desc=f"Hash {root.name}", unit="img"):
        h = hashlib.sha256()
        try:
            with path.open("rb") as fh:
                while True:
                    chunk = fh.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            hashes[h.hexdigest()].append(path)
        except OSError:
            continue

    rows: list[dict[str, Any]] = []
    for digest, paths in hashes.items():
        if len(paths) <= 1:
            continue
        group_id = digest[:16]
        for path in paths:
            rows.append(
                {
                    "dataset": root.name,
                    "duplicate_group": group_id,
                    "sha256": digest,
                    "path": safe_rel(path, root),
                    "group_size": len(paths),
                }
            )
    return rows


def parse_bbox(obj: ET.Element) -> Optional[tuple[float, float, float, float]]:
    box = obj.find("bndbox")
    if box is None:
        return None
    try:
        xmin = float(box.findtext("xmin", "nan"))
        ymin = float(box.findtext("ymin", "nan"))
        xmax = float(box.findtext("xmax", "nan"))
        ymax = float(box.findtext("ymax", "nan"))
        return xmin, ymin, xmax, ymax
    except (TypeError, ValueError):
        return None


def find_nearby_image(
    xml_path: Path,
    filename: str,
    image_index: dict[str, Path],
) -> Optional[Path]:
    if not filename:
        return None

    candidates = [
        xml_path.parent / filename,
        xml_path.parent.parent / "images" / filename,
        xml_path.parent.parent / "image" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return image_index.get(filename.lower())


def audit_hezarai(root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    char_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()

    csv_files = sorted(root.rglob("labels.csv"))
    seen_labels: Counter[str] = Counter()
    seen_filenames: Counter[str] = Counter()

    for csv_path in csv_files:
        split = csv_path.parent.name
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        except Exception as exc:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "csv_parse_error",
                    "path": safe_rel(csv_path, root),
                    "details": repr(exc),
                }
            )
            continue

        columns = {str(c).strip().lower(): c for c in df.columns}
        filename_col = columns.get("filename")
        label_col = columns.get("label")

        if filename_col is None or label_col is None:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "unexpected_csv_columns",
                    "path": safe_rel(csv_path, root),
                    "details": f"columns={list(df.columns)}",
                }
            )
            continue

        for _, rec in df.iterrows():
            filename = str(rec[filename_col]).strip()
            raw_label = "" if pd.isna(rec[label_col]) else str(rec[label_col]).strip()
            normalized = normalize_plate_text(raw_label)

            image_candidates = [
                csv_path.parent / "images" / filename,
                csv_path.parent / filename,
            ]
            image_path = next((p for p in image_candidates if p.exists()), None)

            seen_labels[normalized] += 1
            seen_filenames[f"{split}/{filename}".lower()] += 1
            length_counter[len(normalized)] += 1
            char_counter.update(normalized)

            row = {
                "split": split,
                "filename": filename,
                "raw_label": raw_label,
                "normalized_label": normalized,
                "length": len(normalized),
                "image_exists": image_path is not None,
                "image_path": safe_rel(image_path, root) if image_path else "",
            }
            rows.append(row)

            if not normalized:
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "empty_ocr_label",
                        "path": f"{split}/{filename}",
                        "details": f"raw={raw_label!r}",
                    }
                )

            if image_path is None:
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "missing_image_for_csv_row",
                        "path": f"{split}/{filename}",
                        "details": safe_rel(csv_path, root),
                    }
                )

    duplicate_labels = sum(1 for _, count in seen_labels.items() if count > 1)
    duplicate_filenames = sum(1 for _, count in seen_filenames.items() if count > 1)

    char_rows = [
        {"character": char, "count": count}
        for char, count in sorted(char_counter.items(), key=lambda x: (-x[1], x[0]))
    ]
    length_rows = [
        {"length": length, "count": count}
        for length, count in sorted(length_counter.items())
    ]

    write_csv(rows, out / "hezarai_labels.csv")
    write_csv(char_rows, out / "hezarai_characters.csv")
    write_csv(length_rows, out / "hezarai_label_lengths.csv")

    summary = {
        "csv_files": len(csv_files),
        "rows": len(rows),
        "unique_normalized_labels": len(seen_labels),
        "duplicate_label_values": duplicate_labels,
        "duplicate_filename_keys": duplicate_filenames,
        "alphabet_size": len(char_counter),
        "alphabet": "".join(sorted(char_counter.keys())),
        "length_distribution": dict(sorted(length_counter.items())),
    }
    return summary, problems


def audit_imlp(root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()

    image_index, ambiguous = build_image_name_index(root)
    xml_files = sorted(root.rglob("*.xml"))
    xml_with_plate = 0
    total_boxes = 0
    valid_boxes = 0

    for xml_path in tqdm(xml_files, desc="Audit IMLP XML", unit="xml"):
        try:
            tree = ET.parse(xml_path)
            ann = tree.getroot()
        except Exception as exc:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "xml_parse_error",
                    "path": safe_rel(xml_path, root),
                    "details": repr(exc),
                }
            )
            continue

        filename = (ann.findtext("filename") or "").strip()
        image_path = find_nearby_image(xml_path, filename, image_index)

        try:
            width = float(ann.findtext("size/width", "nan"))
            height = float(ann.findtext("size/height", "nan"))
        except ValueError:
            width = height = math.nan

        objects = ann.findall("object")
        plate_found = False

        for obj in objects:
            class_name = normalize_plate_text(obj.findtext("name", ""))
            class_counter[class_name] += 1
            bbox = parse_bbox(obj)
            total_boxes += 1

            if class_name.lower() == "plate":
                plate_found = True

            if bbox is None:
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "missing_or_invalid_bbox",
                        "path": safe_rel(xml_path, root),
                        "details": f"class={class_name!r}",
                    }
                )
                continue

            xmin, ymin, xmax, ymax = bbox
            valid = (
                xmax > xmin
                and ymax > ymin
                and (math.isnan(width) or (0 <= xmin < xmax <= width))
                and (math.isnan(height) or (0 <= ymin < ymax <= height))
            )
            if valid:
                valid_boxes += 1
            else:
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "bbox_out_of_bounds_or_degenerate",
                        "path": safe_rel(xml_path, root),
                        "details": f"class={class_name}, bbox={bbox}, size=({width},{height})",
                    }
                )

            rows.append(
                {
                    "xml": safe_rel(xml_path, root),
                    "filename": filename,
                    "image_exists": image_path is not None,
                    "image_path": safe_rel(image_path, root) if image_path else "",
                    "class": class_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "width": width,
                    "height": height,
                    "bbox_valid": valid,
                }
            )

        if plate_found:
            xml_with_plate += 1

        if filename and image_path is None:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "missing_image_for_xml",
                    "path": safe_rel(xml_path, root),
                    "details": filename,
                }
            )

    write_csv(rows, out / "imlp_boxes.csv")
    write_csv(
        [{"class": k, "count": v} for k, v in class_counter.most_common()],
        out / "imlp_classes.csv",
    )

    summary = {
        "xml_files": len(xml_files),
        "xml_with_plate": xml_with_plate,
        "total_objects": total_boxes,
        "valid_boxes": valid_boxes,
        "classes": dict(class_counter),
        "ambiguous_image_basenames": len(ambiguous),
    }
    return summary, problems


def audit_ir_lpr(root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()

    image_index, ambiguous = build_image_name_index(root)
    xml_files = sorted(root.rglob("*.xml"))
    valid_xml = 0
    total_objects = 0
    invalid_boxes = 0

    for xml_path in tqdm(xml_files, desc="Audit IR-LPR XML", unit="xml"):
        try:
            ann = ET.parse(xml_path).getroot()
        except Exception as exc:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "xml_parse_error",
                    "path": safe_rel(xml_path, root),
                    "details": repr(exc),
                }
            )
            continue

        valid_xml += 1
        filename = (ann.findtext("filename") or "").strip()
        image_path = find_nearby_image(xml_path, filename, image_index)

        objects_data: list[dict[str, Any]] = []
        xml_order_chars: list[str] = []

        for obj in ann.findall("object"):
            label_raw = (obj.findtext("name") or "").strip()
            label = normalize_plate_text(label_raw)
            class_counter[label] += 1
            xml_order_chars.append(label)
            total_objects += 1

            bbox = parse_bbox(obj)
            if bbox is None:
                invalid_boxes += 1
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "missing_or_invalid_character_bbox",
                        "path": safe_rel(xml_path, root),
                        "details": f"class={label_raw!r}",
                    }
                )
                continue

            xmin, ymin, xmax, ymax = bbox
            if xmax <= xmin or ymax <= ymin:
                invalid_boxes += 1
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "degenerate_character_bbox",
                        "path": safe_rel(xml_path, root),
                        "details": f"class={label}, bbox={bbox}",
                    }
                )
                continue

            objects_data.append(
                {
                    "label": label,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "cx": (xmin + xmax) / 2.0,
                    "cy": (ymin + ymax) / 2.0,
                    "h": ymax - ymin,
                }
            )

        x_sorted = sorted(objects_data, key=lambda item: item["cx"])
        x_order_label = "".join(item["label"] for item in x_sorted)
        xml_order_label = "".join(xml_order_chars)

        y_centers = [item["cy"] for item in objects_data]
        heights = [item["h"] for item in objects_data]
        median_h = statistics.median(heights) if heights else 0.0
        y_spread = (max(y_centers) - min(y_centers)) if len(y_centers) >= 2 else 0.0
        y_spread_ratio = y_spread / median_h if median_h > 0 else 0.0
        possible_multirow = y_spread_ratio > 0.75

        length_counter[len(x_order_label)] += 1

        rows.append(
            {
                "xml": safe_rel(xml_path, root),
                "filename": filename,
                "image_exists": image_path is not None,
                "image_path": safe_rel(image_path, root) if image_path else "",
                "num_objects": len(objects_data),
                "xml_order_label": xml_order_label,
                "x_order_label": x_order_label,
                "orders_match": xml_order_label == x_order_label,
                "y_spread_ratio": round(y_spread_ratio, 4),
                "possible_multirow": possible_multirow,
            }
        )

        if filename and image_path is None:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "missing_image_for_xml",
                    "path": safe_rel(xml_path, root),
                    "details": filename,
                }
            )

        if not objects_data:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "xml_without_valid_character_boxes",
                    "path": safe_rel(xml_path, root),
                    "details": "",
                }
            )

    class_rows = [
        {"character": char, "count": count}
        for char, count in sorted(class_counter.items(), key=lambda x: (-x[1], x[0]))
    ]
    write_csv(rows, out / "ir_lpr_labels.csv")
    write_csv(class_rows, out / "ir_lpr_classes.csv")
    write_csv(
        [{"length": length, "count": count} for length, count in sorted(length_counter.items())],
        out / "ir_lpr_label_lengths.csv",
    )

    summary = {
        "xml_files": len(xml_files),
        "valid_xml": valid_xml,
        "total_character_objects": total_objects,
        "invalid_character_boxes": invalid_boxes,
        "character_classes": len(class_counter),
        "class_distribution": dict(class_counter),
        "reconstructed_length_distribution": dict(sorted(length_counter.items())),
        "ambiguous_image_basenames": len(ambiguous),
    }
    return summary, problems


def audit_iranis(root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    unclassified = 0

    anchor_name = "Iranis Dataset Files"

    for image_path in tqdm(list(iter_images(root)), desc="Audit Iranis", unit="img"):
        parts = image_path.parts
        class_name: Optional[str] = None

        if anchor_name in parts:
            idx = parts.index(anchor_name)
            if idx + 1 < len(parts):
                class_name = parts[idx + 1]

        if class_name:
            class_counter[class_name] += 1
        else:
            unclassified += 1

    class_rows = [
        {"class": cls, "count": count}
        for cls, count in sorted(class_counter.items(), key=lambda x: x[0])
    ]
    write_csv(class_rows, out / "iranis_classes.csv")

    if unclassified:
        problems.append(
            {
                "dataset": root.name,
                "problem": "images_outside_class_tree",
                "path": "",
                "details": f"count={unclassified}",
            }
        )

    summary = {
        "classes": len(class_counter),
        "class_distribution": dict(class_counter),
        "unclassified_images": unclassified,
    }
    return summary, problems


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) != 4:
        return 0.0
    area = 0.0
    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def audit_corners(root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    csv_candidates = sorted(root.rglob("annotations.csv"))
    valid_rows = 0

    for csv_path in csv_candidates:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "csv_parse_error",
                    "path": safe_rel(csv_path, root),
                    "details": repr(exc),
                }
            )
            continue

        expected = ["image", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        missing_cols = [c for c in expected if c not in df.columns]
        if missing_cols:
            problems.append(
                {
                    "dataset": root.name,
                    "problem": "unexpected_corner_columns",
                    "path": safe_rel(csv_path, root),
                    "details": f"missing={missing_cols}",
                }
            )
            continue

        image_dir = csv_path.parent / "images"

        for _, rec in df.iterrows():
            image_name = str(rec["image"]).strip()
            try:
                coords = [float(rec[c]) for c in expected[1:]]
                points = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]
                numeric = all(math.isfinite(v) for v in coords)
            except Exception:
                coords = []
                points = []
                numeric = False

            in_unit_range = numeric and all(0.0 <= v <= 1.0 for v in coords)
            area = polygon_area(points) if points else 0.0
            nondegenerate = area > 1e-6
            image_path = image_dir / image_name
            exists = image_path.exists()

            is_valid = numeric and in_unit_range and nondegenerate and exists
            if is_valid:
                valid_rows += 1

            rows.append(
                {
                    "csv": safe_rel(csv_path, root),
                    "image": image_name,
                    "image_exists": exists,
                    "numeric": numeric,
                    "coordinates_in_0_1": in_unit_range,
                    "polygon_area": area,
                    "nondegenerate": nondegenerate,
                    "valid": is_valid,
                    **({expected[i + 1]: coords[i] for i in range(8)} if coords else {}),
                }
            )

            if not is_valid:
                problems.append(
                    {
                        "dataset": root.name,
                        "problem": "invalid_corner_annotation",
                        "path": image_name,
                        "details": (
                            f"exists={exists}, numeric={numeric}, "
                            f"in_range={in_unit_range}, area={area}"
                        ),
                    }
                )

    write_csv(rows, out / "ir_lpr_corners.csv")

    summary = {
        "annotation_csv_files": len(csv_candidates),
        "rows": len(rows),
        "valid_rows": valid_rows,
        "invalid_rows": len(rows) - valid_rows,
    }
    return summary, problems


def audit_sivd(root: Path, out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    problems: list[dict[str, Any]] = []
    top_counter: Counter[str] = Counter()
    group_counter: Counter[str] = Counter()

    images = list(iter_images(root))
    for path in tqdm(images, desc="Audit SIVD", unit="img"):
        rel = path.relative_to(root)
        parts = rel.parts
        top = parts[0] if parts else "."
        group = "/".join(parts[:2]) if len(parts) >= 2 else top
        top_counter[top] += 1
        group_counter[group] += 1

    rows = [
        {"group": group, "image_count": count}
        for group, count in group_counter.most_common()
    ]
    write_csv(rows, out / "sivd_image_groups.csv")

    summary = {
        "images": len(images),
        "top_level_distribution": dict(top_counter),
        "largest_groups": dict(group_counter.most_common(30)),
    }
    return summary, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep audit Pelak-Khan datasets")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override PELAK_DATASET_ROOT from .env",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Open and verify every image (slow).",
    )
    parser.add_argument(
        "--hash-duplicates",
        action="store_true",
        help="SHA-256 every image and report exact duplicates (slow, reads all image bytes).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    # Windows PowerShell 5.1 often writes UTF-8 files with BOM.
    # utf-8-sig makes .env robust to that.
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    dataset_root = args.dataset_root
    if dataset_root is None:
        env_value = os.getenv("PELAK_DATASET_ROOT")
        if not env_value:
            print(
                "ERROR: PELAK_DATASET_ROOT is missing. "
                "Set it in .env or pass --dataset-root.",
                file=sys.stderr,
            )
            return 2
        dataset_root = Path(env_value)

    dataset_root = dataset_root.expanduser()
    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"ERROR: Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    out = repo_root / "artifacts" / "dataset_audit"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pelak-Khan Deep Dataset Audit")
    print("=" * 80)
    print(f"Started      : {now_string()}")
    print(f"Repository   : {repo_root}")
    print(f"Dataset root : {dataset_root}")
    print(f"Output       : {out}")
    print(f"Verify imgs  : {args.verify_images}")
    print(f"Hash dupes   : {args.hash_duplicates}")
    print("=" * 80)

    all_problems: list[dict[str, Any]] = []
    all_duplicates: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    inventory_rows: list[dict[str, Any]] = []

    present = {p.name: p for p in dataset_root.iterdir() if p.is_dir()}
    missing = [name for name in EXPECTED_DATASETS if name not in present]
    unexpected = sorted(name for name in present if name not in EXPECTED_DATASETS)

    if missing:
        for name in missing:
            all_problems.append(
                {
                    "dataset": name,
                    "problem": "missing_expected_dataset",
                    "path": "",
                    "details": "",
                }
            )

    auditors = {
        "hezarai_persian_plate_ocr": audit_hezarai,
        "imlp": audit_imlp,
        "iranis": audit_iranis,
        "ir_lpr": audit_ir_lpr,
        "ir_lpr_corners": audit_corners,
        "sivd": audit_sivd,
    }

    for name in EXPECTED_DATASETS:
        root = present.get(name)
        if root is None:
            continue

        print(f"\n[{name}] inventory...")
        inventory = file_inventory(root)
        inventory_rows.append(
            {
                "dataset": name,
                "files": inventory["files"],
                "images": inventory["images"],
                "gb": inventory["gb"],
            }
        )

        print(
            f"[{name}] files={inventory['files']:,} "
            f"images={inventory['images']:,} size={inventory['gb']:.3f} GB"
        )

        auditor = auditors[name]
        print(f"[{name}] annotation audit...")
        detail_summary, problems = auditor(root, out)
        all_problems.extend(problems)

        if args.verify_images:
            all_problems.extend(verify_images(root))

        if args.hash_duplicates:
            all_duplicates.extend(hash_duplicates(root))

        summaries[name] = {
            "inventory": inventory,
            "audit": detail_summary,
            "problem_count": sum(1 for p in all_problems if p["dataset"] == name),
        }

    write_csv(inventory_rows, out / "dataset_summary.csv")
    write_csv(all_problems, out / "problems.csv")
    if args.hash_duplicates:
        write_csv(all_duplicates, out / "duplicates.csv")

    final_summary = {
        "generated_at": now_string(),
        "dataset_root": str(dataset_root),
        "expected_datasets": EXPECTED_DATASETS,
        "missing_datasets": missing,
        "unexpected_directories": unexpected,
        "total_problem_rows": len(all_problems),
        "total_duplicate_rows": len(all_duplicates),
        "datasets": summaries,
    }
    write_json(final_summary, out / "summary.json")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"Problem rows : {len(all_problems):,}")
    if args.hash_duplicates:
        print(f"Duplicate rows: {len(all_duplicates):,}")
    print(f"Summary JSON : {out / 'summary.json'}")
    print(f"Summary CSV  : {out / 'dataset_summary.csv'}")
    print(f"Problems CSV : {out / 'problems.csv'}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
