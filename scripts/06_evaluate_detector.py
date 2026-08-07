#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    load_dotenv(repo / ".env", encoding="utf-8-sig")
    gen = os.getenv("PELAK_GENERATED_ROOT")
    if not gen:
        print("ERROR: PELAK_GENERATED_ROOT missing", file=sys.stderr)
        return 2

    ds_root = Path(gen) / "plate_detection_v1"
    data_yaml = ds_root / "dataset.yaml"
    model_path = args.model or (
        repo / "artifacts" / "training" / "detector" /
        "yolo26n_detector_v1" / "weights" / "best.pt"
    )

    out = repo / "artifacts" / "evaluation" / "detector_v1_test"
    out.mkdir(parents=True, exist_ok=True)

    # Exact duplicate leakage check
    by_hash = defaultdict(list)
    for split in SPLITS:
        files = [x for x in (ds_root / "images" / split).iterdir()
                 if x.is_file() and x.suffix.lower() in IMAGE_EXTS]
        print(f"Hashing {split}: {len(files)} images")
        for i, path in enumerate(files, 1):
            by_hash[sha256_file(path)].append((split, path))
            if i % 500 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

    dup_rows = []
    group = 0
    for digest, items in by_hash.items():
        splits = sorted({s for s, _ in items})
        if len(splits) <= 1:
            continue
        group += 1
        for split, path in items:
            dup_rows.append({
                "duplicate_group": group,
                "sha256": digest,
                "split": split,
                "image": str(path),
                "splits_present": ",".join(splits),
                "group_size": len(items),
            })

    pd.DataFrame(dup_rows).to_csv(
        out / "cross_split_exact_duplicates.csv",
        index=False, encoding="utf-8-sig"
    )

    print(f"Cross-split duplicate groups: {group}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(out),
        name="ultralytics",
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(model_path),
        "dataset": str(ds_root),
        "split": "test",
        "cross_split_exact_duplicate_groups": group,
        "cross_split_exact_duplicate_rows": len(dup_rows),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
    }
    (out / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nTEST EVALUATION COMPLETE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if group:
        print("\nWARNING: exact duplicates exist across splits.")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
