#!/usr/bin/env python
"""
Validate OCR sequence dataset v1.

Checks:
- manifest consistency
- all images exist and are readable
- labels are exactly 8 chars
- labels use only charset.json characters
- exact image duplicates across train/val/test (SHA256)
- image dimension/aspect-ratio statistics
- label overlap statistics across splits
- generates a lightweight HTML sample browser

Default dataset:
    <PELAK_GENERATED_ROOT>/ocr_sequence_v1

Outputs:
    artifacts/ocr_dataset_qa/
        summary.json
        problems.csv
        cross_split_exact_duplicates.csv
        sample_rows.csv
        samples.html
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--samples-per-split", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    if args.dataset_root is not None:
        dataset_root = args.dataset_root
    else:
        generated = os.getenv("PELAK_GENERATED_ROOT")
        if not generated:
            print("ERROR: PELAK_GENERATED_ROOT missing from .env", file=sys.stderr)
            return 2
        dataset_root = Path(generated) / "ocr_sequence_v1"

    manifest_path = dataset_root / "manifest.csv"
    charset_path = dataset_root / "charset.json"

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    if not charset_path.exists():
        print(f"ERROR: charset not found: {charset_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    charset_payload = json.loads(charset_path.read_text(encoding="utf-8"))
    charset = set(charset_payload["characters"])

    required_cols = {"split", "image", "label"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"ERROR: manifest missing columns: {sorted(missing_cols)}", file=sys.stderr)
        return 2

    problems: list[dict[str, Any]] = []
    by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_counts = Counter()
    char_counts = Counter()
    widths: list[float] = []
    heights: list[float] = []
    aspects: list[float] = []
    labels_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}

    print("=" * 80)
    print("Pelak-Khan OCR Dataset QA")
    print("=" * 80)
    print(f"Dataset : {dataset_root}")
    print(f"Rows    : {len(df):,}")
    print(f"Charset : {len(charset)}")
    print("=" * 80)

    for idx, rec in df.iterrows():
        split = str(rec["split"]).strip()
        rel_image = str(rec["image"]).strip()
        label = str(rec["label"]).strip()
        image_path = dataset_root / Path(rel_image.replace("\\", "/"))

        if split not in SPLITS:
            problems.append({
                "row": idx,
                "split": split,
                "image": rel_image,
                "label": label,
                "problem": "invalid_split",
            })
            continue

        split_counts[split] += 1
        labels_by_split[split].add(label)

        if len(label) != 8:
            problems.append({
                "row": idx,
                "split": split,
                "image": rel_image,
                "label": label,
                "problem": f"label_length_{len(label)}",
            })

        unknown = sorted(set(label) - charset)
        if unknown:
            problems.append({
                "row": idx,
                "split": split,
                "image": rel_image,
                "label": label,
                "problem": f"unknown_chars:{''.join(unknown)}",
            })

        char_counts.update(label)

        if not image_path.exists():
            problems.append({
                "row": idx,
                "split": split,
                "image": rel_image,
                "label": label,
                "problem": "missing_image",
            })
            continue

        try:
            with Image.open(image_path) as im:
                im.load()
                w, h = im.size
                if w <= 0 or h <= 0:
                    raise ValueError("non-positive image dimensions")
                widths.append(float(w))
                heights.append(float(h))
                aspects.append(float(w) / float(h))
        except Exception as exc:
            problems.append({
                "row": idx,
                "split": split,
                "image": rel_image,
                "label": label,
                "problem": f"unreadable_image:{type(exc).__name__}",
            })
            continue

        digest = sha256_file(image_path)
        by_hash[digest].append((split, rel_image))

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(df):
            print(f"Checked {idx + 1:,}/{len(df):,}")

    duplicate_rows: list[dict[str, Any]] = []
    duplicate_groups = 0
    for digest, items in by_hash.items():
        present = sorted({split for split, _ in items})
        if len(present) <= 1:
            continue
        duplicate_groups += 1
        for split, rel_image in items:
            duplicate_rows.append({
                "duplicate_group": duplicate_groups,
                "sha256": digest,
                "split": split,
                "image": rel_image,
                "splits_present": ",".join(present),
                "group_size": len(items),
            })

    label_overlap = {
        "train_val": len(labels_by_split["train"] & labels_by_split["val"]),
        "train_test": len(labels_by_split["train"] & labels_by_split["test"]),
        "val_test": len(labels_by_split["val"] & labels_by_split["test"]),
    }

    out_root = repo_root / "artifacts" / "ocr_dataset_qa"
    out_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(problems).to_csv(
        out_root / "problems.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(duplicate_rows).to_csv(
        out_root / "cross_split_exact_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = random.Random(args.seed)
    sample_rows = []
    for split in SPLITS:
        subset = df[df["split"] == split].copy()
        if len(subset) > args.samples_per_split:
            chosen_idx = rng.sample(list(subset.index), args.samples_per_split)
            subset = subset.loc[chosen_idx]
        for _, rec in subset.iterrows():
            sample_rows.append({
                "split": split,
                "image": str(rec["image"]),
                "label": str(rec["label"]),
            })

    pd.DataFrame(sample_rows).to_csv(
        out_root / "sample_rows.csv", index=False, encoding="utf-8-sig"
    )

    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Pelak-Khan OCR Dataset Samples</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:24px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}",
        ".card{background:#1d1d1d;padding:12px;border-radius:10px}",
        ".card img{width:100%;height:100px;object-fit:contain;background:#fff;border-radius:6px}",
        ".label{font-size:20px;margin-top:8px;direction:ltr}",
        ".meta{opacity:.7;font-size:12px;word-break:break-all}",
        "</style></head><body>",
        "<h1>Pelak-Khan OCR Dataset Samples</h1>",
    ]

    current = None
    for row in sample_rows:
        if row["split"] != current:
            if current is not None:
                html_parts.append("</div>")
            current = row["split"]
            html_parts.append(f"<h2>{html.escape(current)}</h2><div class='grid'>")

        # Use file:// paths so this HTML is useful locally.
        abs_image = (dataset_root / Path(row["image"].replace("\\", "/"))).resolve()
        html_parts.append(
            "<div class='card'>"
            f"<img src='{abs_image.as_uri()}'>"
            f"<div class='label'>{html.escape(row['label'])}</div>"
            f"<div class='meta'>{html.escape(row['image'])}</div>"
            "</div>"
        )

    if current is not None:
        html_parts.append("</div>")
    html_parts.append("</body></html>")

    (out_root / "samples.html").write_text(
        "".join(html_parts), encoding="utf-8"
    )

    summary = {
        "dataset": str(dataset_root),
        "rows": len(df),
        "split_counts": dict(split_counts),
        "problem_count": len(problems),
        "cross_split_exact_duplicate_groups": duplicate_groups,
        "cross_split_exact_duplicate_rows": len(duplicate_rows),
        "unique_labels_by_split": {
            k: len(v) for k, v in labels_by_split.items()
        },
        "label_overlap_counts": label_overlap,
        "charset_size": len(charset),
        "charset": sorted(charset),
        "character_frequencies": dict(sorted(char_counts.items())),
        "image_width": stats(widths),
        "image_height": stats(heights),
        "image_aspect_ratio": stats(aspects),
        "samples_per_split": args.samples_per_split,
    }

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("OCR DATASET QA COMPLETE")
    print("=" * 80)
    print(f"Train                     : {split_counts['train']:,}")
    print(f"Val                       : {split_counts['val']:,}")
    print(f"Test                      : {split_counts['test']:,}")
    print(f"Problems                  : {len(problems):,}")
    print(f"Cross-split duplicate grp : {duplicate_groups:,}")
    print(f"Unique labels train       : {len(labels_by_split['train']):,}")
    print(f"Unique labels val         : {len(labels_by_split['val']):,}")
    print(f"Unique labels test        : {len(labels_by_split['test']):,}")
    print(f"Label overlap train/test  : {label_overlap['train_test']:,}")
    print(f"Aspect ratio median       : {summary['image_aspect_ratio']['median']:.3f}")
    print(f"Summary                   : {out_root / 'summary.json'}")
    print(f"Samples                   : {out_root / 'samples.html'}")
    print("=" * 80)

    return 1 if problems or duplicate_groups else 0


if __name__ == "__main__":
    raise SystemExit(main())
