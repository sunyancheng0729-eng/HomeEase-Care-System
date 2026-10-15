"""Speaker verification research path (WeSpeaker ResNet34 ONNX embeddings).

Verification output is supporting evidence only; it never unlocks automatic
cancellation of an MVP emergency (audio.auto_cancel_emergency is fixed to false).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homeease.core.paths import models_dir
from homeease.ml.audio.features import kaldi_fbank


class WeSpeakerEmbedder:
    def __init__(self, onnx_name: str = "voxceleb_resnet34_LM.onnx") -> None:
        import onnxruntime as ort

        p = models_dir() / "audio" / "wespeaker" / onnx_name
        if not p.is_file():
            raise FileNotFoundError(f"speaker model missing: {p}")
        self.sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.model_id = Path(onnx_name).stem

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        f = kaldi_fbank(audio16k)
        f = f - f.mean(axis=0, keepdims=True)  # CMN
        e = self.sess.run(None, {self.inp: f[None].astype(np.float32)})[0][0]
        return e / (np.linalg.norm(e) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def eer(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """labels: 1 genuine, 0 impostor."""
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    P, N = y.sum(), (1 - y).sum()
    if P == 0 or N == 0:
        return {"eer": None, "threshold": None}
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    frr = 1 - tp / P
    far = fp / N
    i = int(np.argmin(np.abs(far - frr)))
    return {"eer": float((far[i] + frr[i]) / 2), "threshold": float(s[i])}


def tar_at_far(scores: np.ndarray, labels: np.ndarray, far: float = 0.001) -> dict[str, Any]:
    imp = np.sort(scores[labels == 0])[::-1]
    gen = scores[labels == 1]
    n_imp = len(imp)
    out: dict[str, Any] = {"target_far": far, "impostor_trials": int(n_imp), "genuine_trials": int(len(gen))}
    if n_imp == 0 or len(gen) == 0:
        out["tar"] = None
        return out
    min_measurable = 1.0 / n_imp
    k = int(np.floor(far * n_imp))
    th = float(imp[k]) + 1e-9 if k < n_imp else float(imp[-1])
    out.update(threshold=th, tar=float(np.mean(gen > th)), achieved_far=float(np.mean(imp > th)), min_measurable_far=min_measurable,
               reliable=n_imp >= 10.0 / far, note=None if n_imp >= 10.0 / far else f"only {n_imp} impostor trials: FAR={far} cannot be estimated reliably (need >= {int(10 / far)})")
    return out
