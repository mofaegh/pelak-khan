#!/usr/bin/env python
"""
Evaluate Pelak-Khan CRNN+CTC OCR on the untouched TEST split.

Outputs:
    artifacts/evaluation/ocr_crnn_ctc_v1_test/
        evaluation_summary.json
        predictions.csv
        errors.csv
        per_position_accuracy.csv
        error_pairs.csv

Metrics:
- Exact accuracy: all 8 plate characters must match.
- CER: character error rate (lower is better).
- Mean edit distance.
- Per-position accuracy for predictions of length 8.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class OCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, dataset_root: Path, image_h: int, image_w: int):
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_h, image_w), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_image = str(row["image"])
        image_path = self.dataset_root / Path(rel_image.replace("\\", "/"))
        label = str(row["label"])

        with Image.open(image_path) as im:
            image = self.transform(im.convert("RGB"))

        return image, label, rel_image


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


def collate(batch):
    images, labels, paths = zip(*batch)
    return torch.stack(images, 0), list(labels), list(paths)


def decode(logits: torch.Tensor, idx_to_char: dict[int, str], blank_idx: int = 0):
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


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="ocr_crnn_ctc_v1_test")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    generated_root = os.getenv("PELAK_GENERATED_ROOT")
    if not generated_root:
        print("ERROR: PELAK_GENERATED_ROOT missing from .env", file=sys.stderr)
        return 2

    dataset_root = Path(generated_root) / "ocr_sequence_v1"
    manifest_path = dataset_root / "manifest.csv"

    model_path = args.model or (
        repo_root / "artifacts" / "training" / "ocr" /
        "crnn_ctc_v1" / "best.pt"
    )

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = load_checkpoint(model_path, device)

    chars = list(ckpt["charset"])
    blank_idx = int(ckpt.get("blank_idx", 0))
    image_h = int(ckpt["image_h"])
    image_w = int(ckpt["image_w"])
    hidden = int(ckpt["hidden_size"])
    num_classes = int(ckpt["num_classes"])

    idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}

    model = CRNN(num_classes=num_classes, hidden_size=hidden).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    df = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    ds = OCRDataset(test_df, dataset_root, image_h, image_w)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        collate_fn=collate,
    )

    out_root = repo_root / "artifacts" / "evaluation" / args.name
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    total_edits = 0
    total_chars = 0
    exact = 0
    length_counter = Counter()
    substitution_pairs = Counter()
    pos_correct = [0] * 8
    pos_total = [0] * 8

    start = time.time()

    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            preds = decode(logits, idx_to_char, blank_idx=blank_idx)

            for path, truth, pred in zip(paths, labels, preds):
                ed = edit_distance(pred, truth)
                ok = pred == truth
                exact += int(ok)
                total_edits += ed
                total_chars += len(truth)
                length_counter[len(pred)] += 1

                if len(pred) == len(truth) == 8:
                    for i, (t, p) in enumerate(zip(truth, pred)):
                        pos_total[i] += 1
                        if t == p:
                            pos_correct[i] += 1
                        else:
                            substitution_pairs[(t, p)] += 1

                rows.append({
                    "image": path,
                    "truth": truth,
                    "prediction": pred,
                    "exact": ok,
                    "edit_distance": ed,
                    "pred_length": len(pred),
                })

    elapsed = time.time() - start

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_root / "predictions.csv", index=False, encoding="utf-8-sig")
    pred_df[~pred_df["exact"]].to_csv(
        out_root / "errors.csv", index=False, encoding="utf-8-sig"
    )

    pos_rows = [
        {
            "position": i + 1,
            "correct": pos_correct[i],
            "total_length8_predictions": pos_total[i],
            "accuracy": (pos_correct[i] / pos_total[i]) if pos_total[i] else None,
        }
        for i in range(8)
    ]
    pd.DataFrame(pos_rows).to_csv(
        out_root / "per_position_accuracy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pair_rows = [
        {"truth_char": t, "pred_char": p, "count": c}
        for (t, p), c in substitution_pairs.most_common()
    ]
    pd.DataFrame(pair_rows).to_csv(
        out_root / "error_pairs.csv", index=False, encoding="utf-8-sig"
    )

    n = len(rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": str(model_path),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_val_exact_acc": float(ckpt.get("val_exact_acc", -1)),
        "checkpoint_val_cer": float(ckpt.get("val_cer", -1)),
        "dataset": str(dataset_root),
        "split": "test",
        "samples": n,
        "exact_correct": exact,
        "exact_accuracy": exact / max(n, 1),
        "cer": total_edits / max(total_chars, 1),
        "mean_edit_distance": total_edits / max(n, 1),
        "prediction_length_distribution": dict(sorted(length_counter.items())),
        "image_h": image_h,
        "image_w": image_w,
        "charset_size": len(chars),
        "device": str(device),
        "batch": args.batch,
        "elapsed_seconds": elapsed,
        "images_per_second": n / max(elapsed, 1e-9),
    }

    (out_root / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 84)
    print("Pelak-Khan OCR TEST Evaluation")
    print("=" * 84)
    print(f"Checkpoint epoch : {summary['checkpoint_epoch']}")
    print(f"Test samples     : {n:,}")
    print(f"Exact correct    : {exact:,}")
    print(f"Exact accuracy   : {summary['exact_accuracy']:.4%}")
    print(f"CER              : {summary['cer']:.4%}")
    print(f"Mean edit dist.  : {summary['mean_edit_distance']:.4f}")
    print(f"Speed            : {summary['images_per_second']:.1f} images/s")
    print(f"Errors CSV       : {out_root / 'errors.csv'}")
    print(f"Summary          : {out_root / 'evaluation_summary.json'}")
    print("=" * 84)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
