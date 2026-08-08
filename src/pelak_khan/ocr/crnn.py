from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """CRNN architecture used by Pelak-Khan OCR v1 checkpoints."""

    def __init__(self, num_classes: int, hidden_size: int = 256) -> None:
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
            512,
            hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.permute(2, 0, 1).contiguous()
        x, _ = self.rnn(x)
        return self.classifier(x)


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def ctc_greedy_decode(
    logits: torch.Tensor,
    idx_to_char: dict[int, str],
    blank_idx: int = 0,
) -> list[str]:
    sequences = logits.argmax(dim=2).permute(1, 0).cpu().tolist()
    decoded: list[str] = []
    for sequence in sequences:
        chars: list[str] = []
        previous = None
        for token in sequence:
            if token != blank_idx and token != previous:
                char = idx_to_char.get(token)
                if char is not None:
                    chars.append(char)
            previous = token
        decoded.append("".join(chars))
    return decoded
