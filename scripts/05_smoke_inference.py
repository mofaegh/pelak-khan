#!/usr/bin/env python
"""
Run a smoke inference check for the Pelak-Khan plate detector.

Purpose:
- Confirm the trained .pt checkpoint loads correctly.
- Run inference on unseen TEST images only.
- Save annotated predictions and a CSV/JSON summary.
- This is NOT an accuracy evaluation; the 1-epoch smoke model is expected to be weak.

Default inputs:
  Model:
    artifacts/training/detector/yolo26n_smoke/weights/best.pt

  Dataset:
    <PELAK_GENERATED_ROOT>/plate_detection_v1

Default output:
    artifacts/inference/detector_smoke/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke inference test for Pelak-Khan plate detector."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Override trained checkpoint path.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override plate_detection_v1 root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override inference output directory.",
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Low threshold is intentional for the 1-epoch smoke model.",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.samples <= 0:
        print("ERROR: --samples must be > 0", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    generated_root = os.getenv("PELAK_GENERATED_ROOT")

    model_path = (
        args.model
        if args.model is not None
        else repo_root
        / "artifacts"
        / "training"
        / "detector"
        / "yolo26n_smoke"
        / "weights"
        / "best.pt"
    )

    if args.dataset is not None:
        dataset_root = args.dataset
    else:
        if not generated_root:
            print("ERROR: PELAK_GENERATED_ROOT is missing.", file=sys.stderr)
            return 2
        dataset_root = Path(generated_root) / "plate_detection_v1"

    output_root = (
        args.output
        if args.output is not None
        else repo_root / "artifacts" / "inference" / "detector_smoke"
    )

    manifest_path = dataset_root / "manifest.csv"

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    test_rows = manifest[manifest["split"] == "test"].copy()

    if test_rows.empty:
        print("ERROR: no test rows found in manifest.", file=sys.stderr)
        return 3

    # Balanced sampling from both sources when possible.
    rng = random.Random(args.seed)
    selected: list[dict[str, str]] = []

    groups = list(test_rows.groupby("source", sort=True))
    per_group = max(1, args.samples // max(1, len(groups)))

    for source, group in groups:
        records = group.to_dict(orient="records")
        take = min(per_group, len(records))
        selected.extend(rng.sample(records, take))

    # Fill any remainder from still-unused test records.
    if len(selected) < args.samples:
        used = {row["output_image"] for row in selected}
        remaining = [
            row
            for row in test_rows.to_dict(orient="records")
            if row["output_image"] not in used
        ]
        take = min(args.samples - len(selected), len(remaining))
        if take:
            selected.extend(rng.sample(remaining, take))

    selected = selected[: args.samples]

    if output_root.exists():
        shutil.rmtree(output_root)
    rendered_root = output_root / "rendered"
    crops_root = output_root / "crops"
    rendered_root.mkdir(parents=True, exist_ok=True)
    crops_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pelak-Khan Detector Smoke Inference")
    print("=" * 80)
    print(f"Model   : {model_path}")
    print(f"Dataset : {dataset_root}")
    print(f"Images  : {len(selected)}")
    print(f"imgsz   : {args.imgsz}")
    print(f"conf    : {args.conf}")
    print(f"device  : {args.device}")
    print("=" * 80)

    model = YOLO(str(model_path))

    rows: list[dict[str, Any]] = []
    total_predictions = 0
    images_with_detection = 0
    all_confidences: list[float] = []

    for idx, rec in enumerate(selected, start=1):
        image_path = dataset_root / rec["output_image"]

        if not image_path.exists():
            rows.append(
                {
                    "source": rec["source"],
                    "image": rec["output_image"],
                    "gt_boxes": rec.get("box_count", ""),
                    "pred_boxes": 0,
                    "max_conf": "",
                    "mean_conf": "",
                    "status": "missing_image",
                }
            )
            continue

        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        pred_count = 0 if boxes is None else len(boxes)
        confidences: list[float] = []

        if boxes is not None and pred_count:
            confidences = [float(v) for v in boxes.conf.cpu().tolist()]

        total_predictions += pred_count
        if pred_count:
            images_with_detection += 1
            all_confidences.extend(confidences)

        # Save Ultralytics-rendered prediction image.
        plotted = result.plot()
        # result.plot() is BGR ndarray; convert to RGB with channel reversal.
        plotted_rgb = plotted[..., ::-1]
        rendered_path = rendered_root / f"{idx:03d}__{image_path.name}"
        Image.fromarray(plotted_rgb).save(rendered_path, quality=92)

        # Save every predicted plate crop for easy visual inspection.
        if boxes is not None and pred_count:
            with Image.open(image_path) as original:
                original = original.convert("RGB")
                width, height = original.size

                for det_idx, xyxy in enumerate(boxes.xyxy.cpu().tolist(), start=1):
                    x1, y1, x2, y2 = xyxy
                    x1 = max(0, min(width, int(round(x1))))
                    y1 = max(0, min(height, int(round(y1))))
                    x2 = max(0, min(width, int(round(x2))))
                    y2 = max(0, min(height, int(round(y2))))

                    if x2 > x1 and y2 > y1:
                        crop = original.crop((x1, y1, x2, y2))
                        crop.save(
                            crops_root / f"{idx:03d}__det{det_idx:02d}.jpg",
                            quality=95,
                        )

        rows.append(
            {
                "source": rec["source"],
                "image": rec["output_image"],
                "gt_boxes": int(rec["box_count"]),
                "pred_boxes": pred_count,
                "max_conf": max(confidences) if confidences else "",
                "mean_conf": (
                    sum(confidences) / len(confidences)
                    if confidences
                    else ""
                ),
                "status": "ok",
            }
        )

        print(
            f"[{idx:02d}/{len(selected):02d}] "
            f"{rec['source']:<18} "
            f"GT={rec['box_count']} "
            f"PRED={pred_count} "
            f"MAX_CONF={max(confidences) if confidences else 0:.4f}"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "predictions.csv"
    pd.DataFrame(rows).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "generated_at": now_string(),
        "model": str(model_path),
        "dataset": str(dataset_root),
        "samples_requested": args.samples,
        "samples_processed": len(rows),
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "device": args.device,
        "images_with_at_least_one_detection": images_with_detection,
        "images_without_detection": len(rows) - images_with_detection,
        "total_predictions": total_predictions,
        "mean_predictions_per_image": (
            total_predictions / len(rows) if rows else 0.0
        ),
        "max_prediction_confidence": (
            max(all_confidences) if all_confidences else None
        ),
        "mean_prediction_confidence": (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else None
        ),
        "note": (
            "This is only a smoke inference check. "
            "Do not interpret these numbers as detector quality."
        ),
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("SMOKE INFERENCE COMPLETE")
    print("=" * 80)
    print(f"Images processed : {len(rows)}")
    print(f"Images detected  : {images_with_detection}")
    print(f"Predictions      : {total_predictions}")
    print(f"Rendered images  : {rendered_root}")
    print(f"Plate crops      : {crops_root}")
    print(f"CSV              : {csv_path}")
    print(f"Summary          : {summary_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
