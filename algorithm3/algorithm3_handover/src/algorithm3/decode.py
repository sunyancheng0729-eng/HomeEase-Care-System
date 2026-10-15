"""Audio input helpers: any common container to float32 mono at 16 kHz (memory only, nothing is written)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from homeease.ml.audio.features import SR, resample

MAX_SECONDS = 120.0  # longer input is cut: the demo is for short recordings


def to_mono16k(samples: Any, sr: int) -> np.ndarray:
    """Array (mono, or channels in the last axis, int16 or float) at ``sr`` -> float32 mono at 16 kHz, at most MAX_SECONDS."""
    x = np.asarray(samples)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if int(sr) != SR:
        x = resample(x, int(sr), SR)
    return x[: int(MAX_SECONDS * SR)]


def decode_bytes(data: bytes) -> np.ndarray:
    """WAV / FLAC / OGG through soundfile, anything else (webm and opus from a browser recording, mp3, m4a) through PyAV, which
    faster-whisper installs anyway."""
    try:
        import soundfile as sf

        x, fs = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return to_mono16k(x, int(fs))
    except Exception:
        pass
    import av  # PyAV

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return to_mono16k(np.concatenate(chunks).astype(np.float32), SR)
