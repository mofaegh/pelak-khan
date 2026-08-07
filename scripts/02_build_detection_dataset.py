#!/usr/bin/env python
"""
Build Pelak-Khan plate detection dataset (YOLO format).

Sources:
1) IMLP: uses only VOC/XML objects whose class is exactly "plate".
2) IR-LPR-corners: converts the four normalized plate corners to an axis-aligned box.

Raw datasets are never modified.

Default output:
    <PELAK_GENERATED_ROOT>/plate_detection_v1/

Structure:
    plate_detection_v1/
    â”œâ”€â”€ images/
    â”‚   â”œâ”€â”€ train/
    â”‚   â”œâ”€â”€ val/
    â”‚   â””â”€â”€ test/
    â”œâ”€â”€ labels/
    â”‚   â”œâ”€â”€ train/
    â”‚   â”œâ”€â”€ val/
    â”‚   â””â”€â”€ test/
    â”œâ”€â”€ dataset.yaml
    â”œâ”€â”€ manifest.csv
    â””â”€â”€ build_summary.json

Example:
    python scripts/02_build_detection_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")
CLASS_ID = 0
CLASS_NAME = "plate"


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_json(data: Any, path: Path) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def find_image_for_xml(xml_path: Path, filename: str) -> Optional[Path]:
    """
    Find the image paired with an IMLP XML without relying on a global basename
    index (IMLP contains many duplicate basenames).
    """
    filename = (filename or "").strip()
    if not filename:
        return None

    parent = xml_path.parent
    grandparent = parent.parent

    candidates = [
        parent / filename,
        grandparent / "images" / filename,
        grandparent / "image" / filename,
        grandparent / "Images" / filename,
        grandparent / "JPEGImages" / filename,
        parent / "images" / filename,
        parent / "image" / filename,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Some annotations may store a filename whose extension differs only by
    # case or image format. Search locally by stem before giving up.
    target_stem = Path(filename).stem.lower()
    local_dirs = [
        parent,
        grandparent / "images",
        grandparent / "image",
        grandparent / "Images",
        grandparent / "JPEGImages",
    ]
    for directory in local_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for ext in IMAGE_EXTS:
            candidate = directory / f"{Path(filename).stem}{ext}"
            if candidate.exists():
                return candidate
        try:
            for candidate in directory.iterdir():
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in IMAGE_EXTS
                    and candidate.stem.lower() == target_stem
                ):
                    return candidate
        except OSError:
            pass

    return None


def parse_imlp_samples(imlp_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    xml_files = sorted(imlp_root.rglob("*.xml"))

    for xml_path in tqdm(xml_files, desc="Parse IMLP", unit="xml"):
        try:
            ann = ET.parse(xml_path).getroot()
        except Exception as exc:
            problems.append(
                {
                    "source": "imlp",
                    "problem": "xml_parse_error",
                    "path": safe_rel(xml_path, imlp_root),
                    "details": repr(exc),
                }
            )
            continue

        filename = (ann.findtext("filename") or "").strip()

        try:
            width = float(ann.findtext("size/width", "nan"))
            height = float(ann.findtext("size/height", "nan"))
        except ValueError:
            width = height = math.nan

        if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
            problems.append(
                {
                    "source": "imlp",
                    "problem": "invalid_image_size_in_xml",
                    "path": safe_rel(xml_path, imlp_root),
                    "details": f"width={width}, height={height}",
                }
            )
            continue

        boxes: list[tuple[float, float, float, float]] = []

        for obj in ann.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name != CLASS_NAME:
                continue

            box = obj.find("bndbox")
            if box is None:
                continue

            try:
                xmin = float(box.findtext("xmin", "nan"))
                ymin = float(box.findtext("ymin", "nan"))
                xmax = float(box.findtext("xmax", "nan"))
                ymax = float(box.findtext("ymax", "nan"))
            except (TypeError, ValueError):
                continue

            if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
                continue

            if xmax <= xmin or ymax <= ymin:
                continue

            # Audit has already shown these boxes are valid, but clamp
            # defensively in case the raw data changes.
            xmin = max(0.0, min(xmin, width))
            xmax = max(0.0, min(xmax, width))
            ymin = max(0.0, min(ymin, height))
            ymax = max(0.0, min(ymax, height))

            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))

        # IMLP also contains character/digit annotations. For the detector we
        # intentionally retain only images containing at least one "plate".
        if not boxes:
            continue

        image_path = find_image_for_xml(xml_path, filename)
        if image_path is None:
            problems.append(
                {
                    "source": "imlp",
                    "problem": "missing_image_for_plate_xml",
                    "path": safe_rel(xml_path, imlp_root),
                    "details": filename,
                }
            )
            continue

        samples.append(
            {
                "source": "imlp",
                "source_root": imlp_root,
                "image_path": image_path,
                "annotation_path": xml_path,
                "width": width,
                "height": height,
                "boxes_abs": boxes,
                "boxes_norm": [
                    (
                        xmin / width,
                        ymin / height,
                        xmax / width,
                        ymax / height,
                    )
                    for xmin, ymin, xmax, ymax in boxes
                ],
            }
        )

    return samples, problems


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_corner_samples(
    corners_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    csv_files = sorted(corners_root.rglob("annotations.csv"))
    if not csv_files:
        return [], [
            {
                "source": "ir_lpr_corners",
                "problem": "annotations_csv_not_found",
                "path": "",
                "details": "",
            }
        ]

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:
            problems.append(
                {
                    "source": "ir_lpr_corners",
                    "problem": "csv_parse_error",
                    "path": safe_rel(csv_path, corners_root),
                    "details": repr(exc),
                }
            )
            continue

        expected = ["image", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
        missing = [col for col in expected if col not in df.columns]
        if missing:
            problems.append(
                {
                    "source": "ir_lpr_corners",
                    "problem": "missing_columns",
                    "path": safe_rel(csv_path, corners_root),
                    "details": ",".join(missing),
                }
            )
            continue

        image_dir = csv_path.parent / "images"

        for _, rec in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Parse IR-LPR-corners",
            unit="row",
        ):
            image_name = str(rec["image"]).strip()
            image_path = image_dir / image_name

            if not image_path.exists():
                problems.append(
                    {
                        "source": "ir_lpr_corners",
                        "problem": "missing_image",
                        "path": image_name,
                        "details": safe_rel(csv_path, corners_root),
                    }
                )
                continue

            try:
                coords = [
                    float(rec["x1"]), float(rec["y1"]),
                    float(rec["x2"]), float(rec["y2"]),
                    float(rec["x3"]), float(rec["y3"]),
                    float(rec["x4"]), float(rec["y4"]),
                ]
            except Exception as exc:
                problems.append(
                    {
                        "source": "ir_lpr_corners",
                        "problem": "non_numeric_coordinates",
                        "path": image_name,
                        "details": repr(exc),
                    }
                )
                continue

            if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in coords):
                problems.append(
                    {
                        "source": "ir_lpr_corners",
                        "problem": "coordinates_out_of_range",
                        "path": image_name,
                        "details": repr(coords),
                    }
                )
                continue

            points = [
                (coords[0], coords[1]),
                (coords[2], coords[3]),
                (coords[4], coords[5]),
                (coords[6], coords[7]),
            ]

            if polygon_area(points) <= 1e-6:
                problems.append(
                    {
                        "source": "ir_lpr_corners",
                        "problem": "degenerate_polygon",
                        "path": image_name,
                        "details": repr(points),
                    }
                )
                continue

            xmin = min(x for x, _ in points)
            ymin = min(y for _, y in points)
            xmax = max(x for x, _ in points)
            ymax = max(y for _, y in points)

            if xmax <= xmin or ymax <= ymin:
                problems.append(
                    {
                        "source": "ir_lpr_corners",
                        "problem": "degenerate_bbox",
                        "path": image_name,
                        "details": f"{xmin},{ymin},{xmax},{ymax}",
                    }
                )
                continue

            samples.append(
                {
                    "source": "ir_lpr_corners",
                    "source_root": corners_root,
                    "image_path": image_path,
                    "annotation_path": csv_path,
                    "width": None,
                    "height": None,
                    "boxes_abs": None,
                    "boxes_norm": [(xmin, ymin, xmax, ymax)],
                    "corners_norm": points,
                }
            )

    return samples, problems


def split_from_key(
    key: str,
    train_ratio: float,
    val_ratio: float,
) -> str:
    """
    Deterministic split based on SHA-1. Re-running the builder produces the
    same split as long as the source relative path stays the same.
    """
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)

    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def to_yolo_line(box: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    vals = [x_center, y_center, width, height]
    vals = [max(0.0, min(1.0, value)) for value in vals]

    return (
        f"{CLASS_ID} "
        f"{vals[0]:.8f} {vals[1]:.8f} "
        f"{vals[2]:.8f} {vals[3]:.8f}"
    )


def unique_output_name(sample: dict[str, Any]) -> str:
    source = sample["source"]
    image_path: Path = sample["image_path"]
    source_root: Path = sample["source_root"]

    rel = safe_rel(image_path, source_root)
    digest = hashlib.sha1(f"{source}|{rel}".encode("utf-8")).hexdigest()[:12]

    # Keep a readable stem but sanitize characters that are awkward in scripts.
    stem = image_path.stem
    safe_stem = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_"
        for ch in stem
    )
    safe_stem = safe_stem[:80] or "image"

    return f"{source}__{digest}__{safe_stem}{image_path.suffix.lower()}"


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"

    raise ValueError(f"Unsupported link mode: {mode}")


def validate_ratios(train: float, val: float, test: float) -> None:
    values = (train, val, test)
    if any(v <= 0 for v in values):
        raise ValueError("All split ratios must be > 0.")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {sum(values):.12f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build YOLO plate-detection dataset from IMLP + IR-LPR-corners."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override PELAK_DATASET_ROOT.",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=None,
        help="Override PELAK_GENERATED_ROOT.",
    )
    parser.add_argument(
        "--name",
        default="plate_detection_v1",
        help="Generated dataset directory name.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="copy",
        help="hardlink saves disk space when source/output are on the same volume.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output dataset before rebuilding.",
    )
    args = parser.parse_args()

    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    raw_root_value = (
        str(args.dataset_root)
        if args.dataset_root is not None
        else os.getenv("PELAK_DATASET_ROOT")
    )
    generated_root_value = (
        str(args.generated_root)
        if args.generated_root is not None
        else os.getenv("PELAK_GENERATED_ROOT")
    )

    if not raw_root_value:
        print(
            "ERROR: PELAK_DATASET_ROOT is not configured.",
            file=sys.stderr,
        )
        return 2

    if not generated_root_value:
        print(
            "ERROR: PELAK_GENERATED_ROOT is not configured.\n"
            "Add it to .env, for example:\n"
            r"PELAK_GENERATED_ROOT=Y:\projects\Github\pelak-khan1\datasets_generated",
            file=sys.stderr,
        )
        return 2

    raw_root = Path(raw_root_value)
    generated_root = Path(generated_root_value)
    output_root = generated_root / args.name

    imlp_root = raw_root / "imlp"
    corners_root = raw_root / "ir_lpr_corners"

    for required in (imlp_root, corners_root):
        if not required.exists():
            print(f"ERROR: Required dataset not found: {required}", file=sys.stderr)
            return 2

    if output_root.exists():
        if not args.overwrite:
            print(
                f"ERROR: Output already exists: {output_root}\n"
                "Use --overwrite to rebuild it.",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(output_root)

    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pelak-Khan Detection Dataset Builder")
    print("=" * 80)
    print(f"Started        : {now_string()}")
    print(f"Raw root       : {raw_root}")
    print(f"Generated root : {generated_root}")
    print(f"Output         : {output_root}")
    print(
        f"Split          : train={args.train_ratio:.2f}, "
        f"val={args.val_ratio:.2f}, test={args.test_ratio:.2f}"
    )
    print(f"Materialize    : {args.link_mode}")
    print("=" * 80)

    imlp_samples, imlp_problems = parse_imlp_samples(imlp_root)
    corner_samples, corner_problems = parse_corner_samples(corners_root)

    samples = imlp_samples + corner_samples
    problems = imlp_problems + corner_problems

    if not samples:
        print("ERROR: No detection samples were produced.", file=sys.stderr)
        return 3

    # Split each source independently. This avoids a validation/test set that
    # accidentally consists almost entirely of one source/domain.
    for sample in samples:
        source_root: Path = sample["source_root"]
        image_path: Path = sample["image_path"]
        rel = safe_rel(image_path, source_root)
        key = f"{sample['source']}|{rel}"
        sample["split"] = split_from_key(
            key,
            args.train_ratio,
            args.val_ratio,
        )
        sample["output_name"] = unique_output_name(sample)

    manifest_rows: list[dict[str, Any]] = []
    link_mode_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_split_counts: Counter[tuple[str, str]] = Counter()
    box_counts: Counter[str] = Counter()
    used_output_names: set[str] = set()

    for sample in tqdm(samples, desc="Materialize dataset", unit="img"):
        source = sample["source"]
        split = sample["split"]
        output_name = sample["output_name"]
        source_root: Path = sample["source_root"]
        image_path: Path = sample["image_path"]

        if output_name in used_output_names:
            raise RuntimeError(f"Output-name collision: {output_name}")
        used_output_names.add(output_name)

        output_image = output_root / "images" / split / output_name
        output_label = output_root / "labels" / split / f"{Path(output_name).stem}.txt"

        actual_link_mode = materialize_file(
            image_path,
            output_image,
            args.link_mode,
        )
        link_mode_counts[actual_link_mode] += 1

        yolo_lines = [to_yolo_line(box) for box in sample["boxes_norm"]]
        output_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        split_counts[split] += 1
        source_counts[source] += 1
        source_split_counts[(source, split)] += 1
        box_counts[split] += len(yolo_lines)

        manifest_rows.append(
            {
                "source": source,
                "split": split,
                "source_image": safe_rel(image_path, source_root),
                "source_annotation": safe_rel(
                    sample["annotation_path"],
                    source_root,
                ),
                "output_image": safe_rel(output_image, output_root),
                "output_label": safe_rel(output_label, output_root),
                "box_count": len(yolo_lines),
                "materialization": actual_link_mode,
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        output_root / "manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(problems).to_csv(
        output_root / "build_problems.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Ultralytics YOLO dataset YAML.
    yaml_text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )
    (output_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "generated_at": now_string(),
        "raw_dataset_root": str(raw_root),
        "output_root": str(output_root),
        "class": {"id": CLASS_ID, "name": CLASS_NAME},
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "samples_total": len(samples),
        "boxes_total": sum(len(s["boxes_norm"]) for s in samples),
        "samples_by_source": dict(source_counts),
        "samples_by_split": dict(split_counts),
        "boxes_by_split": dict(box_counts),
        "samples_by_source_and_split": {
            f"{source}/{split}": count
            for (source, split), count in sorted(source_split_counts.items())
        },
        "materialization": dict(link_mode_counts),
        "build_problem_count": len(problems),
        "build_problem_types": dict(
            Counter(problem["problem"] for problem in problems)
        ),
    }
    write_json(summary, output_root / "build_summary.json")

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(f"Images total : {summary['samples_total']:,}")
    print(f"Boxes total  : {summary['boxes_total']:,}")
    print(f"IMLP images  : {source_counts['imlp']:,}")
    print(f"Corners imgs : {source_counts['ir_lpr_corners']:,}")
    print(
        "Splits       : "
        + ", ".join(f"{s}={split_counts[s]:,}" for s in SPLITS)
    )
    print(f"Problems     : {len(problems):,}")
    print(f"dataset.yaml : {output_root / 'dataset.yaml'}")
    print(f"manifest.csv : {output_root / 'manifest.csv'}")
    print(f"summary.json : {output_root / 'build_summary.json'}")
    print("=" * 80)

    if problems:
        print(
            "\nNOTE: Build problems were written to build_problems.csv. "
            "Do not start training until they are reviewed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

