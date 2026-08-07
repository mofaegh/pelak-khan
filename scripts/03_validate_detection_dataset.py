#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SPLITS = ("train", "val", "test")
CLASS_ID = 0


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_label(path: Path):
    boxes = []
    errors = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        return [], [f"read_error: {exc!r}"]

    if not lines:
        return [], ["empty_label_file"]

    for i, raw in enumerate(lines, start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"line_{i}: expected_5_fields")
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"line_{i}: non_numeric")
            continue

        if cls != CLASS_ID:
            errors.append(f"line_{i}: unexpected_class={cls}")

        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            errors.append(f"line_{i}: non_finite")
            continue

        if not (0 <= xc <= 1 and 0 <= yc <= 1):
            errors.append(f"line_{i}: center_out_of_range")
        if not (0 < w <= 1 and 0 < h <= 1):
            errors.append(f"line_{i}: size_out_of_range")

        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2
        eps = 1e-6
        if x1 < -eps or y1 < -eps or x2 > 1 + eps or y2 > 1 + eps:
            errors.append(f"line_{i}: edges_out_of_range")

        boxes.append((cls, xc, yc, w, h))

    return boxes, errors


def draw_sample(image_path: Path, boxes, caption: str, out_path: Path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        width, height = img.size
        lw = max(2, round(min(width, height) / 250))

        for idx, (_, xc, yc, bw, bh) in enumerate(boxes, start=1):
            x1 = (xc - bw / 2) * width
            y1 = (yc - bh / 2) * height
            x2 = (xc + bw / 2) * width
            y2 = (yc + bh / 2) * height
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=lw)
            draw.text((max(0, x1), max(0, y1 - 12)), f"plate #{idx}", fill=(255, 0, 0), font=font)

        bar_h = 26
        canvas = Image.new("RGB", (width, height + bar_h), (255, 255, 255))
        canvas.paste(img, (0, bar_h))
        ImageDraw.Draw(canvas).text((8, 7), caption, fill=(0, 0, 0), font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path, quality=92)


def build_html(qa_root: Path, rows, summary):
    cards = []
    current = None

    for row in sorted(rows, key=lambda r: (r["split"], r["source"], r["rendered_path"])):
        group = (row["split"], row["source"])
        if group != current:
            if current is not None:
                cards.append("</div></section>")
            cards.append(
                f"<section><h2>{html.escape(group[0])} / {html.escape(group[1])}</h2><div class='grid'>"
            )
            current = group

        rel = Path(row["rendered_path"]).as_posix()
        cards.append(
            f"<div class='card'><a href='{html.escape(rel)}' target='_blank'>"
            f"<img src='{html.escape(rel)}'></a>"
            f"<div class='meta'>boxes={row['box_count']}<br>"
            f"<small>{html.escape(row['source_image'])}</small></div></div>"
        )

    if current is not None:
        cards.append("</div></section>")

    status_class = "ok" if summary["problem_count"] == 0 else "bad"
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Pelak-Khan Detection QA</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;background:#f4f4f4;color:#222}}
.summary,section{{background:white;padding:18px;border-radius:10px;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
.card{{border:1px solid #ddd;border-radius:8px;overflow:hidden}}
.card img{{width:100%;height:240px;object-fit:contain;background:#111;display:block}}
.meta{{padding:10px;overflow-wrap:anywhere}}
.ok{{color:#087f23;font-weight:bold}} .bad{{color:#b00020;font-weight:bold}}
</style>
</head>
<body>
<h1>Pelak-Khan Detection Dataset QA</h1>
<div class="summary">
Generated: {html.escape(summary["generated_at"])}<br>
Images checked: {summary["images_checked"]}<br>
Labels checked: {summary["labels_checked"]}<br>
Boxes checked: {summary["boxes_checked"]}<br>
Problems: <span class="{status_class}">{summary["problem_count"]}</span><br>
Rendered samples: {summary["rendered_samples"]}
</div>
{''.join(cards)}
</body>
</html>"""
    (qa_root / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--samples-per-group", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    if args.dataset:
        dataset_root = args.dataset
    else:
        generated = os.getenv("PELAK_GENERATED_ROOT")
        if not generated:
            print("ERROR: PELAK_GENERATED_ROOT is missing.", file=sys.stderr)
            return 2
        dataset_root = Path(generated) / "plate_detection_v1"

    manifest_path = dataset_root / "manifest.csv"
    if not dataset_root.exists() or not manifest_path.exists():
        print(f"ERROR: dataset or manifest missing: {dataset_root}", file=sys.stderr)
        return 2

    qa_root = dataset_root / "qa"
    if qa_root.exists():
        shutil.rmtree(qa_root)
    qa_root.mkdir(parents=True)

    print("=" * 80)
    print("Pelak-Khan Detection Dataset QA")
    print("=" * 80)
    print(f"Dataset      : {dataset_root}")
    print(f"Samples/group: {args.samples_per_group}")
    print(f"Seed         : {args.seed}")
    print("=" * 80)

    problems = []
    total_images = 0
    total_labels = 0
    total_boxes = 0
    split_counts = Counter()
    box_counts = Counter()

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split

        images = {p.stem: p for p in image_dir.iterdir() if p.is_file()}
        labels = {p.stem: p for p in label_dir.glob("*.txt")}

        total_images += len(images)
        total_labels += len(labels)
        split_counts[split] = len(images)

        for stem in sorted(set(images) - set(labels)):
            problems.append({"split": split, "problem": "image_without_label", "path": str(images[stem]), "details": ""})
        for stem in sorted(set(labels) - set(images)):
            problems.append({"split": split, "problem": "label_without_image", "path": str(labels[stem]), "details": ""})

        for stem in tqdm(sorted(set(images) & set(labels)), desc=f"Validate {split}", unit="img"):
            image_path = images[stem]
            label_path = labels[stem]

            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as exc:
                problems.append({"split": split, "problem": "image_open_error", "path": str(image_path), "details": repr(exc)})

            boxes, errors = parse_label(label_path)
            total_boxes += len(boxes)
            box_counts[split] += len(boxes)

            for error in errors:
                problems.append({"split": split, "problem": "invalid_yolo_label", "path": str(label_path), "details": error})

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    rng = random.Random(args.seed)
    rendered_rows = []

    for (split, source), group in manifest.groupby(["split", "source"], sort=True):
        records = group.to_dict(orient="records")
        for rec in rng.sample(records, min(args.samples_per_group, len(records))):
            image_path = dataset_root / rec["output_image"]
            label_path = dataset_root / rec["output_label"]
            boxes, errors = parse_label(label_path)
            if errors:
                continue

            rendered_name = f"{source}__{Path(rec['output_image']).stem}.jpg"
            out_path = qa_root / "rendered" / split / rendered_name
            try:
                draw_sample(image_path, boxes, f"{split} | {source} | boxes={len(boxes)}", out_path)
            except Exception as exc:
                problems.append({"split": split, "problem": "render_error", "path": str(image_path), "details": repr(exc)})
                continue

            rendered_rows.append({
                "split": split,
                "source": source,
                "source_image": rec["source_image"],
                "box_count": len(boxes),
                "rendered_path": str(out_path.relative_to(qa_root)),
            })

    pd.DataFrame(problems).to_csv(qa_root / "qa_problems.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rendered_rows).to_csv(qa_root / "rendered_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": now_string(),
        "dataset_root": str(dataset_root),
        "images_checked": total_images,
        "labels_checked": total_labels,
        "boxes_checked": total_boxes,
        "images_by_split": dict(split_counts),
        "boxes_by_split": dict(box_counts),
        "problem_count": len(problems),
        "problem_types": dict(Counter(p["problem"] for p in problems)),
        "rendered_samples": len(rendered_rows),
        "samples_per_group_requested": args.samples_per_group,
        "seed": args.seed,
    }

    (qa_root / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_html(qa_root, rendered_rows, summary)

    print("\n" + "=" * 80)
    print("QA COMPLETE")
    print("=" * 80)
    print(f"Images checked : {total_images:,}")
    print(f"Labels checked : {total_labels:,}")
    print(f"Boxes checked  : {total_boxes:,}")
    print(f"Problems       : {len(problems):,}")
    print(f"Rendered       : {len(rendered_rows):,}")
    print(f"HTML report    : {qa_root / 'index.html'}")
    print(f"Summary        : {qa_root / 'qa_summary.json'}")
    print("=" * 80)

    if problems:
        print("\nDo NOT start training yet. Review qa_problems.csv.")
        return 1

    print("\nStructural QA passed. Inspect the boxes in index.html before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
