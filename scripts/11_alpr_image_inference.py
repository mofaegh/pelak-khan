#!/usr/bin/env python
"""
Pelak-Khan end-to-end image ALPR inference:
    image -> plate detector -> crop -> CRNN+CTC OCR -> results

Default models:
    Detector:
      artifacts/training/detector/yolo26n_detector_v1/weights/best.pt
    OCR:
      artifacts/training/ocr/crnn_ctc_v1/best.pt

Outputs:
    artifacts/inference/alpr_image_v1/<run_name>/
        results.csv
        results.json
        annotated/
        crops/

Notes:
- OCR text is saved in CSV/JSON.
- Annotated images draw boxes and detection confidence only, because
  OpenCV's default text renderer is not suitable for Persian Unicode.
- OCR confidence is intentionally not reported yet; the current CRNN+CTC
  checkpoint has not been calibrated for sequence confidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.rnn(x)
        return self.classifier(x)


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def ctc_decode(logits: torch.Tensor, idx_to_char: dict[int, str], blank_idx: int = 0) -> list[str]:
    # logits: T,B,C
    seqs = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
    outputs = []
    for seq in seqs:
        chars = []
        prev = None
        for token in seq:
            if token != blank_idx and token != prev:
                chars.append(idx_to_char[token])
            prev = token
        outputs.append("".join(chars))
    return outputs


def find_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    if source.is_dir():
        return sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    raise FileNotFoundError(source)


def expand_box(x1, y1, x2, y2, w, h, pad_frac: float):
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_frac
    py = bh * pad_frac

    nx1 = max(0, int(round(x1 - px)))
    ny1 = max(0, int(round(y1 - py)))
    nx2 = min(w, int(round(x2 + px)))
    ny2 = min(h, int(round(y2 + py)))
    return nx1, ny1, nx2, ny2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--detector", type=Path, default=None)
    parser.add_argument("--ocr", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-imgsz", type=int, default=640)
    parser.add_argument("--pad-frac", type=float, default=0.04)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    detector_path = args.detector or (
        repo_root / "artifacts" / "training" / "detector" /
        "yolo26n_detector_v1" / "weights" / "best.pt"
    )
    ocr_path = args.ocr or (
        repo_root / "artifacts" / "training" / "ocr" /
        "crnn_ctc_v1" / "best.pt"
    )

    if not detector_path.exists():
        print(f"ERROR: detector model not found: {detector_path}", file=sys.stderr)
        return 2
    if not ocr_path.exists():
        print(f"ERROR: OCR model not found: {ocr_path}", file=sys.stderr)
        return 2

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    images = find_images(args.source)
    if not images:
        print(f"ERROR: no images found in {args.source}", file=sys.stderr)
        return 2

    run_name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    out_root = repo_root / "artifacts" / "inference" / "alpr_image_v1" / run_name
    annotated_dir = out_root / "annotated"
    crops_dir = out_root / "crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Pelak-Khan End-to-End ALPR Image Inference")
    print("=" * 88)
    print(f"Source       : {args.source}")
    print(f"Images       : {len(images)}")
    print(f"Detector     : {detector_path}")
    print(f"OCR          : {ocr_path}")
    print(f"Device       : {device}")
    print(f"Det conf     : {args.det_conf}")
    print(f"Crop padding : {args.pad_frac:.1%}")
    print(f"Output       : {out_root}")
    print("=" * 88)

    detector = YOLO(str(detector_path))

    ckpt = load_checkpoint(ocr_path, device)
    chars = list(ckpt["charset"])
    blank_idx = int(ckpt.get("blank_idx", 0))
    image_h = int(ckpt["image_h"])
    image_w = int(ckpt["image_w"])
    hidden = int(ckpt["hidden_size"])
    num_classes = int(ckpt["num_classes"])
    idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}

    ocr_model = CRNN(num_classes=num_classes, hidden_size=hidden).to(device)
    ocr_model.load_state_dict(ckpt["model_state"])
    ocr_model.eval()

    ocr_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_h, image_w), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    rows = []
    total_detections = 0
    start_all = time.time()

    for img_idx, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            rows.append({
                "source_image": str(image_path),
                "status": "unreadable_image",
                "plate_index": "",
                "plate_text": "",
                "det_confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "crop_path": "",
            })
            continue

        h, w = frame.shape[:2]
        annotated = frame.copy()

        det_results = detector.predict(
            source=frame,
            imgsz=args.det_imgsz,
            conf=args.det_conf,
            device=str(device),
            verbose=False,
        )

        boxes = det_results[0].boxes
        image_detections = []

        if boxes is not None and len(boxes) > 0:
            for plate_idx, box in enumerate(boxes, 1):
                xyxy = box.xyxy[0].detach().cpu().tolist()
                conf = float(box.conf[0].detach().cpu())

                x1, y1, x2, y2 = xyxy
                cx1, cy1, cx2, cy2 = expand_box(
                    x1, y1, x2, y2, w, h, args.pad_frac
                )

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)
                tensor = ocr_transform(crop_pil).unsqueeze(0).to(device)

                with torch.inference_mode():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=(device.type == "cuda"),
                    ):
                        logits = ocr_model(tensor)

                plate_text = ctc_decode(
                    logits, idx_to_char, blank_idx=blank_idx
                )[0]

                total_detections += 1
                safe_stem = "".join(
                    ch if ch.isalnum() or ch in "-_" else "_"
                    for ch in image_path.stem
                )
                crop_name = f"{img_idx:05d}_{safe_stem}_plate_{plate_idx:02d}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                cv2.rectangle(
                    annotated,
                    (int(round(x1)), int(round(y1))),
                    (int(round(x2)), int(round(y2))),
                    (0, 255, 0),
                    2,
                )
                # ASCII-only overlay; Persian OCR text stays in CSV/JSON.
                overlay = f"plate {plate_idx} det={conf:.3f}"
                cv2.putText(
                    annotated,
                    overlay,
                    (int(round(x1)), max(20, int(round(y1)) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                record = {
                    "source_image": str(image_path),
                    "status": "ok",
                    "plate_index": plate_idx,
                    "plate_text": plate_text,
                    "det_confidence": conf,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "crop_x1": cx1,
                    "crop_y1": cy1,
                    "crop_x2": cx2,
                    "crop_y2": cy2,
                    "crop_path": str(crop_path),
                }
                rows.append(record)
                image_detections.append(record)

        if not image_detections:
            rows.append({
                "source_image": str(image_path),
                "status": "no_plate_detected",
                "plate_index": "",
                "plate_text": "",
                "det_confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "crop_path": "",
            })

        annotated_name = f"{img_idx:05d}_{image_path.name}"
        cv2.imwrite(str(annotated_dir / annotated_name), annotated)

        texts = [r["plate_text"] for r in image_detections]
        print(
            f"[{img_idx}/{len(images)}] {image_path.name} -> "
            f"{len(image_detections)} plate(s) | "
            f"{texts if texts else 'NO DETECTION'}"
        )

    elapsed = time.time() - start_all

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "results.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(args.source),
        "detector": str(detector_path),
        "ocr": str(ocr_path),
        "device": str(device),
        "images_processed": len(images),
        "detections": total_detections,
        "elapsed_seconds": elapsed,
        "results": rows,
    }
    (out_root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print("ALPR IMAGE INFERENCE COMPLETE")
    print("=" * 88)
    print(f"Images processed : {len(images)}")
    print(f"Plate detections : {total_detections}")
    print(f"Elapsed seconds  : {elapsed:.2f}")
    print(f"Results CSV      : {out_root / 'results.csv'}")
    print(f"Results JSON     : {out_root / 'results.json'}")
    print(f"Crops            : {crops_dir}")
    print(f"Annotated        : {annotated_dir}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
