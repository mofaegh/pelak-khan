#!/usr/bin/env python
"""
Train a Persian license-plate OCR baseline with CRNN + CTC.

Dataset:
    <PELAK_GENERATED_ROOT>/ocr_sequence_v1/
        images/{train,val,test}/
        manifest.csv
        charset.json

Outputs:
    artifacts/training/ocr/crnn_ctc_v1/
        best.pt
        last.pt
        history.csv
        training_summary.json

Metrics:
- CER: character error rate (lower is better)
- Exact accuracy: full 8-character plate exact match (higher is better)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ----------------------------
# Reproducibility
# ----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Dataset
# ----------------------------

class OCRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: Path,
        char_to_idx: dict[str, int],
        image_h: int,
        image_w: int,
        train: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.char_to_idx = char_to_idx

        aug = []
        if train:
            aug = [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.18, contrast=0.18)],
                    p=0.45,
                ),
                transforms.RandomAffine(
                    degrees=2.0,
                    translate=(0.02, 0.05),
                    scale=(0.96, 1.04),
                    shear=1.5,
                    fill=255,
                ),
            ]

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                *aug,
                transforms.Resize((image_h, image_w), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.dataset_root / Path(str(row["image"]).replace("\\", "/"))
        label = str(row["label"])

        with Image.open(image_path) as im:
            image = im.convert("RGB")
        image = self.transform(image)

        target = torch.tensor(
            [self.char_to_idx[ch] for ch in label],
            dtype=torch.long,
        )
        return image, target, label


def collate_batch(batch):
    images, targets, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets, dim=0)
    return images, targets_concat, target_lengths, list(labels)


# ----------------------------
# Model
# ----------------------------

class CRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # H/2, W/2

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # H/4, W/4

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),    # H/8, W/4

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),    # H/16, W/4
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
        x = self.cnn(x)                      # B,C,H,W
        x = x.mean(dim=2)                    # B,C,W
        x = x.permute(2, 0, 1).contiguous() # T,B,C
        x, _ = self.rnn(x)
        x = self.classifier(x)               # T,B,classes
        return x


# ----------------------------
# Metrics
# ----------------------------

def ctc_greedy_decode(
    logits: torch.Tensor,
    idx_to_char: dict[int, str],
    blank_idx: int = 0,
) -> list[str]:
    # logits: T,B,C
    pred = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
    decoded = []

    for seq in pred:
        chars = []
        prev = None
        for token in seq:
            if token != blank_idx and token != prev:
                chars.append(idx_to_char[token])
            prev = token
        decoded.append("".join(chars))

    return decoded


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return prev[-1]


@dataclass
class EvalResult:
    loss: float
    cer: float
    exact_acc: float


# ----------------------------
# Train / Eval loops
# ----------------------------

def run_epoch(
    model,
    loader,
    criterion,
    device,
    idx_to_char,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    total_edits = 0
    total_chars = 0
    exact = 0

    for images, targets, target_lengths, labels in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            T, B, _ = log_probs.shape
            input_lengths = torch.full(
                size=(B,),
                fill_value=T,
                dtype=torch.long,
                device=device,
            )
            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

        decoded = ctc_greedy_decode(logits.detach(), idx_to_char)
        for pred_text, true_text in zip(decoded, labels):
            total_edits += edit_distance(pred_text, true_text)
            total_chars += len(true_text)
            exact += int(pred_text == true_text)

    return EvalResult(
        loss=total_loss / max(total_samples, 1),
        cer=total_edits / max(total_chars, 1),
        exact_acc=exact / max(total_samples, 1),
    )


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-h", type=int, default=32)
    parser.add_argument("--image-w", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default="crnn_ctc_v1")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", encoding="utf-8-sig")

    generated_root = os.getenv("PELAK_GENERATED_ROOT")
    if not generated_root:
        print("ERROR: PELAK_GENERATED_ROOT missing from .env", file=sys.stderr)
        return 2

    dataset_root = Path(generated_root) / "ocr_sequence_v1"
    manifest_path = dataset_root / "manifest.csv"
    charset_path = dataset_root / "charset.json"

    if not manifest_path.exists() or not charset_path.exists():
        print(f"ERROR: OCR dataset incomplete: {dataset_root}", file=sys.stderr)
        return 2

    if args.mode == "smoke":
        epochs = args.epochs or 1
        batch_size = args.batch or 32
        workers = args.workers if args.workers is not None else 0
        patience = args.patience or 2
        fraction = args.fraction if args.fraction is not None else 0.10
    else:
        epochs = args.epochs or 40
        batch_size = args.batch or 128
        workers = args.workers if args.workers is not None else 8
        patience = args.patience or 8
        fraction = args.fraction if args.fraction is not None else 1.0

    seed_everything(args.seed)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    charset_payload = json.loads(charset_path.read_text(encoding="utf-8"))
    chars = list(charset_payload["characters"])

    # CTC blank = 0. Actual characters start at 1.
    char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
    idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
    num_classes = len(chars) + 1

    df = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if fraction < 1.0:
        train_df = train_df.sample(
            frac=fraction,
            random_state=args.seed,
        ).reset_index(drop=True)

    train_ds = OCRDataset(
        train_df, dataset_root, char_to_idx,
        args.image_h, args.image_w, train=True
    )
    val_ds = OCRDataset(
        val_df, dataset_root, char_to_idx,
        args.image_h, args.image_w, train=False
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
        collate_fn=collate_batch,
    )

    model = CRNN(num_classes=num_classes, hidden_size=args.hidden).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    output_dir = repo_root / "artifacts" / "training" / "ocr" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("Pelak-Khan OCR Training — CRNN + CTC")
    print("=" * 84)
    print(f"Mode          : {args.mode}")
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Dataset       : {dataset_root}")
    print(f"Train samples : {len(train_ds):,}")
    print(f"Val samples   : {len(val_ds):,}")
    print(f"Test reserved : {len(test_df):,}")
    print(f"Charset       : {len(chars)} chars")
    print(f"Classes + CTC : {num_classes}")
    print(f"Image         : {args.image_w}x{args.image_h}")
    print(f"Batch         : {batch_size}")
    print(f"Epochs        : {epochs}")
    print(f"Workers       : {workers}")
    print(f"Fraction      : {fraction}")
    print(f"Output        : {output_dir}")
    print("=" * 84)

    history = []
    best_exact = -1.0
    best_cer = math.inf
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_res = run_epoch(
            model, train_loader, criterion, device,
            idx_to_char, optimizer=optimizer, scaler=scaler
        )

        with torch.no_grad():
            val_res = run_epoch(
                model, val_loader, criterion, device,
                idx_to_char, optimizer=None, scaler=None
            )

        scheduler.step(val_res.exact_acc)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = (
            val_res.exact_acc > best_exact
            or (
                math.isclose(val_res.exact_acc, best_exact, rel_tol=0, abs_tol=1e-12)
                and val_res.cer < best_cer
            )
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "charset": chars,
            "blank_idx": 0,
            "image_h": args.image_h,
            "image_w": args.image_w,
            "hidden_size": args.hidden,
            "num_classes": num_classes,
            "val_exact_acc": val_res.exact_acc,
            "val_cer": val_res.cer,
        }

        torch.save(checkpoint, output_dir / "last.pt")

        if improved:
            best_exact = val_res.exact_acc
            best_cer = val_res.cer
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_res.loss,
            "train_cer": train_res.cer,
            "train_exact_acc": train_res.exact_acc,
            "val_loss": val_res.loss,
            "val_cer": val_res.cer,
            "val_exact_acc": val_res.exact_acc,
            "best_val_exact_acc": best_exact,
            "best_val_cer": best_cer,
            "seconds": elapsed,
        }
        history.append(row)

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_res.loss:.4f} CER {train_res.cer:.4f} "
            f"exact {train_res.exact_acc:.4f} | "
            f"val loss {val_res.loss:.4f} CER {val_res.cer:.4f} "
            f"exact {val_res.exact_acc:.4f} | "
            f"lr {lr_now:.2e} | "
            f"{elapsed:.1f}s"
            + ("  BEST" if improved else "")
        )

        if args.mode == "full" and epochs_without_improvement >= patience:
            print(f"Early stopping: no validation improvement for {patience} epochs.")
            break

    total_seconds = time.time() - start_time

    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "device": str(device),
        "torch": torch.__version__,
        "dataset": str(dataset_root),
        "output": str(output_dir),
        "charset": chars,
        "charset_size": len(chars),
        "num_classes_with_blank": num_classes,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples_reserved": len(test_df),
        "image_h": args.image_h,
        "image_w": args.image_w,
        "batch": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_val_exact_acc": best_exact,
        "best_val_cer": best_cer,
        "total_seconds": total_seconds,
        "best_exists": (output_dir / "best.pt").exists(),
        "last_exists": (output_dir / "last.pt").exists(),
    }

    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("OCR TRAINING COMPLETE")
    print("=" * 84)
    print(f"Best val exact accuracy : {best_exact:.4%}")
    print(f"Best val CER            : {best_cer:.4%}")
    print(f"best.pt                 : {output_dir / 'best.pt'}")
    print(f"last.pt                 : {output_dir / 'last.pt'}")
    print(f"Summary                 : {output_dir / 'training_summary.json'}")
    print("=" * 84)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
