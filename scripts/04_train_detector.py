#!/usr/bin/env python
"""
Train Pelak-Khan plate detector with Ultralytics YOLO.

Modes
-----
smoke:
    Fast CPU sanity check on a small fraction of the training set.
    Intended for the development laptop.

full:
    Full training configuration intended for a CUDA GPU server.

Examples
--------
Smoke test on laptop:
    python scripts/04_train_detector.py --mode smoke

Full training on a GPU server:
    python scripts/04_train_detector.py --mode full --device 0 --epochs 100 --batch 16 --imgsz 640

Notes
-----
- Raw datasets are never modified.
- The canonical generated dataset is expected at:
    <PELAK_GENERATED_ROOT>/plate_detection_v1/dataset.yaml
- Training outputs default to:
    artifacts/training/detector/
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pelak-Khan plate detector."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="smoke=quick CPU validation, full=real training.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics pretrained checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset.yaml path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Ultralytics device, e.g. "cpu", "0", "0,1".',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override training image size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override dataloader workers.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Override fraction of training set to use (0 < fraction <= 1).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override Ultralytics run name.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Override training output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    generated_root = os.getenv("PELAK_GENERATED_ROOT")

    if args.dataset is not None:
        dataset_yaml = args.dataset
    else:
        if not generated_root:
            print(
                "ERROR: PELAK_GENERATED_ROOT is missing from .env.",
                file=sys.stderr,
            )
            return 2
        dataset_yaml = (
            Path(generated_root)
            / "plate_detection_v1"
            / "dataset.yaml"
        )

    if not dataset_yaml.exists():
        print(
            f"ERROR: dataset.yaml not found: {dataset_yaml}",
            file=sys.stderr,
        )
        return 2

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except Exception as exc:
        print(
            f"ERROR: ML dependencies are not importable: {exc!r}",
            file=sys.stderr,
        )
        return 2

    # Stable, conservative defaults for the Windows development laptop.
    if args.mode == "smoke":
        config: dict[str, Any] = {
            "epochs": 1,
            "batch": 2,
            "imgsz": 320,
            "workers": 0,
            "fraction": 0.05,
            "device": "cpu",
            "val": False,
            "plots": False,
            "save": True,
            "cache": False,
            "patience": 0,
            "seed": args.seed,
            "deterministic": True,
            "verbose": True,
        }
        default_name = "yolo26n_smoke"
    else:
        # GPU-server defaults. These can be overridden on the CLI.
        config = {
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "workers": 4,
            "fraction": 1.0,
            "device": "0",
            "val": True,
            "plots": True,
            "save": True,
            "cache": False,
            "patience": 20,
            "seed": args.seed,
            "deterministic": True,
            "verbose": True,
        }
        default_name = "yolo26n_full"

    # Apply CLI overrides.
    for key in ("epochs", "batch", "imgsz", "workers", "fraction", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if not (0 < float(config["fraction"]) <= 1.0):
        print(
            f"ERROR: fraction must satisfy 0 < fraction <= 1, got {config['fraction']}",
            file=sys.stderr,
        )
        return 2

    project_dir = (
        args.project
        if args.project is not None
        else repo_root / "artifacts" / "training" / "detector"
    )
    run_name = args.name or default_name

    project_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pelak-Khan Plate Detector Training")
    print("=" * 80)
    print(f"Started       : {now_string()}")
    print(f"Mode          : {args.mode}")
    print(f"Python        : {platform.python_version()}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Ultralytics   : {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model         : {args.model}")
    print(f"Dataset       : {dataset_yaml}")
    print(f"Project       : {project_dir}")
    print(f"Run name      : {run_name}")
    print("-" * 80)
    for key in sorted(config):
        print(f"{key:14}: {config[key]}")
    print("=" * 80)

    if args.mode == "full" and str(config["device"]).lower() != "cpu":
        if not torch.cuda.is_available():
            print(
                "ERROR: Full mode requested a CUDA device, "
                "but torch.cuda.is_available() is False.",
                file=sys.stderr,
            )
            return 3

    try:
        model = YOLO(args.model)

        results = model.train(
            data=str(dataset_yaml),
            project=str(project_dir),
            name=run_name,
            exist_ok=True,
            **config,
        )
    except Exception as exc:
        print("\nTRAINING FAILED")
        print(repr(exc))
        return 4

    run_dir = project_dir / run_name
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"

    summary = {
        "finished_at": now_string(),
        "mode": args.mode,
        "model": args.model,
        "dataset": str(dataset_yaml),
        "project": str(project_dir),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": config,
        "best_exists": best_path.exists(),
        "best_path": str(best_path),
        "last_exists": last_path.exists(),
        "last_path": str(last_path),
        "results_type": type(results).__name__,
    }

    summary_path = run_dir / "pelak_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Run directory : {run_dir}")
    print(f"best.pt       : {best_path} | exists={best_path.exists()}")
    print(f"last.pt       : {last_path} | exists={last_path.exists()}")
    print(f"Summary       : {summary_path}")
    print("=" * 80)

    if not last_path.exists():
        print(
            "WARNING: Training completed but last.pt was not found. "
            "Inspect the Ultralytics run directory before proceeding."
        )
        return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
