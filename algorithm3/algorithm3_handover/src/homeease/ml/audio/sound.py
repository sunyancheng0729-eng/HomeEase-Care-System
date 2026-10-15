"""Environmental sound-event classification (HELP / IMPACT / ABNORMAL / DAILY / UNKNOWN).

Two families are compared: clip statistics of log-mel/MFCC with a classical model,
and a small CNN on log-mel patches. UNKNOWN is a decision outcome (low confidence),
not a trained class. Labels are research/assistive hints and never overrule fusion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from homeease.ml.audio.features import logmel, mfcc, stats_vector

CLASSES = ["HELP", "IMPACT", "ABNORMAL", "DAILY"]


def clip_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate([stats_vector(logmel(x, n_mels=64)), stats_vector(mfcc(x))])


def logmel_patch(x: np.ndarray, frames: int = 256) -> np.ndarray:
    lm = logmel(x, n_mels=64)
    lm = (lm - lm.mean()) / (lm.std() + 1e-6)
    if len(lm) >= frames:
        start = (len(lm) - frames) // 2
        lm = lm[start : start + frames]
    else:
        lm = np.vstack([lm, np.zeros((frames - len(lm), lm.shape[1]))])
    return lm.T[None].astype(np.float32)  # (1, mels, frames)


class SmallCNN(nn.Module):
    def __init__(self, n_out: int) -> None:
        super().__init__()
        self.cfg = {"arch": "small_cnn", "n_out": n_out}

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.net = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def train_cnn(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, n_out: int, seed: int = 0, epochs: int = 30) -> tuple[SmallCNN, dict[str, Any]]:
    from homeease.ml.common.torch_models import train

    counts = np.bincount(ytr, minlength=n_out).astype(float)
    cw = counts.sum() / np.maximum(counts, 1) / n_out

    def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        x = x.copy()
        for i in range(len(x)):
            shift = int(rng.integers(-20, 21))
            x[i] = np.roll(x[i], shift, axis=-1)
            if rng.random() < 0.5:
                f0 = int(rng.integers(0, 56))
                x[i, :, f0 : f0 + 8, :] = 0.0
        return x

    model = SmallCNN(n_out)
    return train(model, Xtr, ytr, Xva, yva, epochs=epochs, lr=1e-3, batch=32, seed=seed, patience=6, multiclass=True, augment=augment, class_weights=cw)


def decide(probs: np.ndarray, classes: list[str], min_conf: float = 0.5) -> str:
    k = int(np.argmax(probs))
    return classes[k] if probs[k] >= min_conf else "UNKNOWN"
