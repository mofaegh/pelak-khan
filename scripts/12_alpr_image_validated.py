#!/usr/bin/env python
"""
Pelak-Khan validated end-to-end image ALPR:
    image -> detector -> crop -> OCR -> Iranian plate validator -> accepted/rejected

Baseline plate rule for v1:
    DD L DDD DD
stored as:
    DDLDDDDD
Example:
    11س11111

Outputs:
    artifacts/inference/alpr_image_v2/<run_name>/
        results.csv
        results.json
        accepted.csv
        rejected.csv
        annotated/
        crops/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.rnn = nn.LSTM(
            512, hidden_size, num_layers=2, bidirectional=True,
            dropout=0.2, batch_first=False
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


def ctc_decode(logits, idx_to_char, blank_idx=0):
    seqs = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
    out = []
    for seq in seqs:
        chars, prev = [], None
        for token in seq:
            if token != blank_idx and token != prev:
                chars.append(idx_to_char[token])
            prev = token
        out.append("".join(chars))
    return out


def find_images(source: Path):
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    raise FileNotFoundError(source)


def expand_box(x1, y1, x2, y2, width, height, pad_frac):
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    return (
        max(0, int(round(x1 - px))),
        max(0, int(round(y1 - py))),
        min(width, int(round(x2 + px))),
        min(height, int(round(y2 + py))),
    )


def edge_suspect(x1, y1, x2, y2, width, height, margin_frac):
    mx = width * margin_frac
    my = height * margin_frac
    return x1 <= mx or y1 <= my or x2 >= (width - mx) or y2 >= (height - my)


def validate_standard_plate(text: str, allowed_letters: set[str]):
    """
    v1 standard plate:
      positions 0-1: digits
      position 2: Persian plate letter from OCR charset
      positions 3-7: digits
    """
    reasons = []

    if len(text) != 8:
        reasons.append(f"invalid_length_{len(text)}")
        return False, reasons

    if not text[:2].isdigit():
        reasons.append("first_two_not_digits")

    if text[2] not in allowed_letters:
        reasons.append("position_3_not_valid_letter")

    if not text[3:].isdigit():
        reasons.append("last_five_not_digits")

    return len(reasons) == 0, reasons


def display_plate(text: str):
    if len(text) == 8:
        return f"{text[:2]} {text[2]} {text[3:6]} | {text[6:]}"
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--detector", type=Path, default=None)
    p.add_argument("--ocr", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--det-imgsz", type=int, default=640)
    p.add_argument("--pad-frac", type=float, default=0.04)
    p.add_argument("--edge-margin-frac", type=float, default=0.01)
    p.add_argument("--allow-edge", action="store_true",
                   help="Do not reject otherwise-valid detections near image edges.")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    detector_path = args.detector or (
        repo / "artifacts/training/detector/yolo26n_detector_v1/weights/best.pt"
    )
    ocr_path = args.ocr or (
        repo / "artifacts/training/ocr/crnn_ctc_v1/best.pt"
    )

    if not detector_path.exists():
        print(f"ERROR detector missing: {detector_path}", file=sys.stderr)
        return 2
    if not ocr_path.exists():
        print(f"ERROR OCR missing: {ocr_path}", file=sys.stderr)
        return 2

    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    images = find_images(args.source)
    if not images:
        print("ERROR: no images found", file=sys.stderr)
        return 2

    run_name = args.name or time.strftime("run_%Y%m%d_%H%M%S")
    out = repo / "artifacts/inference/alpr_image_v2" / run_name
    crops_dir = out / "crops"
    annotated_dir = out / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(detector_path))

    ckpt = load_checkpoint(ocr_path, device)
    chars = list(ckpt["charset"])
    allowed_letters = {ch for ch in chars if not ch.isdigit()}
    idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
    blank_idx = int(ckpt.get("blank_idx", 0))

    ocr_model = CRNN(
        num_classes=int(ckpt["num_classes"]),
        hidden_size=int(ckpt["hidden_size"]),
    ).to(device)
    ocr_model.load_state_dict(ckpt["model_state"])
    ocr_model.eval()

    ocr_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((int(ckpt["image_h"]), int(ckpt["image_w"])), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    records = []
    accepted_count = 0
    rejected_count = 0

    print("=" * 90)
    print("Pelak-Khan Validated ALPR")
    print("=" * 90)
    print(f"Images          : {len(images)}")
    print(f"Device          : {device}")
    print(f"Detector conf   : {args.det_conf}")
    print(f"Edge margin     : {args.edge_margin_frac:.1%}")
    print(f"Allowed letters : {''.join(sorted(allowed_letters))}")
    print(f"Output          : {out}")
    print("=" * 90)

    for image_idx, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        annotated = frame.copy()
        result = detector.predict(
            source=frame,
            imgsz=args.det_imgsz,
            conf=args.det_conf,
            device=str(device),
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"[{image_idx}/{len(images)}] {image_path.name}: NO DETECTION")
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)
            continue

        for plate_idx, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            det_conf = float(box.conf[0].detach().cpu())
            edge = edge_suspect(
                x1, y1, x2, y2, w, h, args.edge_margin_frac
            )

            cx1, cy1, cx2, cy2 = expand_box(
                x1, y1, x2, y2, w, h, args.pad_frac
            )
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = ocr_transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ):
                    logits = ocr_model(tensor)

            text = ctc_decode(logits, idx_to_char, blank_idx)[0]
            format_valid, reasons = validate_standard_plate(text, allowed_letters)

            accepted = format_valid and (args.allow_edge or not edge)
            if edge and not args.allow_edge:
                reasons.append("touches_image_edge")

            status = "ACCEPTED" if accepted else "REJECTED"
            if accepted:
                accepted_count += 1
                color = (0, 255, 0)
            else:
                rejected_count += 1
                color = (0, 0, 255)

            crop_name = f"{image_idx:05d}_{image_path.stem}_plate_{plate_idx:02d}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            cv2.rectangle(
                annotated,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                color,
                2,
            )
            cv2.putText(
                annotated,
                f"{status} det={det_conf:.3f}",
                (int(round(x1)), max(20, int(round(y1)) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            rec = {
                "source_image": str(image_path),
                "plate_index": plate_idx,
                "raw_text": text,
                "display_text": display_plate(text),
                "det_confidence": det_conf,
                "format_valid": format_valid,
                "edge_suspect": edge,
                "accepted": accepted,
                "status": status,
                "reject_reasons": "|".join(reasons),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop_path": str(crop_path),
            }
            records.append(rec)

            print(
                f"[{image_idx}/{len(images)}] {image_path.name} "
                f"plate#{plate_idx}: {text!r} -> {status} "
                f"(format={format_valid}, edge={edge}, det={det_conf:.3f})"
                + (f" reasons={reasons}" if reasons else "")
            )

        cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    df = pd.DataFrame(records)
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    df[df["accepted"] == True].to_csv(
        out / "accepted.csv", index=False, encoding="utf-8-sig"
    )
    df[df["accepted"] == False].to_csv(
        out / "rejected.csv", index=False, encoding="utf-8-sig"
    )

    payload = {
        "images_processed": len(images),
        "detections": len(records),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "results": records,
    }
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("VALIDATED ALPR COMPLETE")
    print("=" * 90)
    print(f"Detections : {len(records)}")
    print(f"Accepted   : {accepted_count}")
    print(f"Rejected   : {rejected_count}")
    print(f"Accepted CSV: {out / 'accepted.csv'}")
    print(f"Rejected CSV: {out / 'rejected.csv'}")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
