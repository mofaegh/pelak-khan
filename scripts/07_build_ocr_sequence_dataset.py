#!/usr/bin/env python
"""
Build a clean sequence-OCR dataset (v1) from hezarai_persian_plate_ocr.

Policy for OCR v1:
- Preserve the dataset's native train/validation/test split.
- Normalize Persian/Arabic digits to ASCII internally.
- Normalize Arabic letter variants to Persian forms.
- Keep only 8-character labels containing letters/digits.
- Exclude punctuation-coded/special plate labels for this first baseline.
- Copy images (never hardlink) so raw data cannot be modified accidentally.

Default output:
    <PELAK_GENERATED_ROOT>/ocr_sequence_v1/

Outputs:
    images/train/
    images/val/
    images/test/
    manifest.csv
    skipped.csv
    charset.json
    dataset_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    ASCII_DIGITS + ASCII_DIGITS,
)

CHAR_TRANSLATION = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ة": "ه",
    "ۀ": "ه",
})

INVISIBLE = {
    "\u200c", "\u200d", "\u200e", "\u200f",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\ufeff",
}

SPLIT_MAP = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    for ch in INVISIBLE:
        text = text.replace(ch, "")
    text = text.translate(DIGIT_TRANSLATION).translate(CHAR_TRANSLATION)
    return "".join(ch for ch in text if not ch.isspace())


def infer_split(csv_path: Path, dataset_root: Path) -> str | None:
    try:
        rel_parts = csv_path.relative_to(dataset_root).parts
    except ValueError:
        rel_parts = csv_path.parts

    for part in reversed(rel_parts[:-1]):
        mapped = SPLIT_MAP.get(part.lower())
        if mapped:
            return mapped
    return None


def find_image(csv_path: Path, filename: str) -> Path | None:
    candidates = [
        csv_path.parent / "images" / filename,
        csv_path.parent / filename,
        csv_path.parent / "imgs" / filename,
        csv_path.parent / "Images" / filename,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def safe_name(split: str, image_path: Path, root: Path) -> str:
    rel = str(image_path.relative_to(root)) if image_path.is_relative_to(root) else str(image_path)
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in image_path.stem)
    return f"{split}__{digest}__{stem}{image_path.suffix.lower()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--generated-root", type=Path, default=None)
    parser.add_argument("--name", default="ocr_sequence_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    raw_root_value = str(args.raw_root) if args.raw_root else os.getenv("PELAK_DATASET_ROOT")
    generated_root_value = str(args.generated_root) if args.generated_root else os.getenv("PELAK_GENERATED_ROOT")

    if not raw_root_value or not generated_root_value:
        print("ERROR: PELAK_DATASET_ROOT and PELAK_GENERATED_ROOT must be configured.", file=sys.stderr)
        return 2

    raw_root = Path(raw_root_value)
    generated_root = Path(generated_root_value)
    source_root = raw_root / "hezarai_persian_plate_ocr"
    output_root = generated_root / args.name

    if not source_root.exists():
        print(f"ERROR: source dataset not found: {source_root}", file=sys.stderr)
        return 2

    if output_root.exists():
        if not args.overwrite:
            print(f"ERROR: output exists: {output_root}; use --overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_root.rglob("labels.csv"))
    if not csv_files:
        print("ERROR: no labels.csv found.", file=sys.stderr)
        return 3

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    charset = Counter()
    split_counts = Counter()
    raw_special_chars = Counter()
    raw_lengths = Counter()

    for csv_path in csv_files:
        split = infer_split(csv_path, source_root)
        if split is None:
            skipped.append({
                "split": "",
                "filename": "",
                "raw_label": "",
                "normalized_label": "",
                "reason": "cannot_infer_split",
                "source_csv": str(csv_path),
            })
            continue

        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
        cols = {str(c).strip().lower(): c for c in df.columns}
        filename_col = cols.get("filename")
        label_col = cols.get("label")

        if filename_col is None or label_col is None:
            raise RuntimeError(f"Unexpected CSV columns in {csv_path}: {list(df.columns)}")

        for _, rec in tqdm(df.iterrows(), total=len(df), desc=f"OCR {split}", unit="img"):
            filename = str(rec[filename_col]).strip()
            raw_label = "" if pd.isna(rec[label_col]) else str(rec[label_col]).strip()
            label = normalize_label(raw_label)

            raw_lengths[len(label)] += 1
            for ch in label:
                if not ch.isalnum():
                    raw_special_chars[ch] += 1

            reason = None
            if not label:
                reason = "empty_label"
            elif len(label) != 8:
                reason = f"length_{len(label)}"
            elif any(not ch.isalnum() for ch in label):
                reason = "contains_special_symbol"

            image_path = find_image(csv_path, filename)
            if image_path is None:
                reason = reason or "missing_image"

            if reason:
                skipped.append({
                    "split": split,
                    "filename": filename,
                    "raw_label": raw_label,
                    "normalized_label": label,
                    "reason": reason,
                    "source_csv": str(csv_path.relative_to(source_root)),
                })
                continue

            out_name = safe_name(split, image_path, source_root)
            out_image = output_root / "images" / split / out_name
            shutil.copy2(image_path, out_image)

            charset.update(label)
            split_counts[split] += 1

            rows.append({
                "split": split,
                "image": str(out_image.relative_to(output_root)).replace("\\", "/"),
                "label": label,
                "raw_label": raw_label,
                "length": len(label),
                "source_image": str(image_path.relative_to(source_root)).replace("\\", "/"),
            })

    pd.DataFrame(rows).to_csv(
        output_root / "manifest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(skipped).to_csv(
        output_root / "skipped.csv", index=False, encoding="utf-8-sig"
    )

    charset_sorted = sorted(charset.keys())
    charset_payload = {
        "characters": charset_sorted,
        "size": len(charset_sorted),
        "frequencies": dict(sorted(charset.items())),
        "note": "Digits are normalized to ASCII internally. Persian display conversion can be applied later.",
    }
    (output_root / "charset.json").write_text(
        json.dumps(charset_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "source": str(source_root),
        "output": str(output_root),
        "accepted_total": len(rows),
        "skipped_total": len(skipped),
        "accepted_by_split": dict(split_counts),
        "raw_label_length_distribution": dict(sorted(raw_lengths.items())),
        "raw_special_symbol_counts": dict(sorted(raw_special_chars.items())),
        "charset_size": len(charset_sorted),
        "charset": charset_sorted,
        "policy": {
            "required_length": 8,
            "letters_and_digits_only": True,
            "copy_images": True,
            "native_split_preserved": True,
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 80)
    print("OCR SEQUENCE DATASET V1 BUILT")
    print("=" * 80)
    print(f"Accepted : {len(rows):,}")
    print(f"Skipped  : {len(skipped):,}")
    print(f"Train    : {split_counts['train']:,}")
    print(f"Val      : {split_counts['val']:,}")
    print(f"Test     : {split_counts['test']:,}")
    print(f"Charset  : {len(charset_sorted)} -> {''.join(charset_sorted)}")
    print(f"Output   : {output_root}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
