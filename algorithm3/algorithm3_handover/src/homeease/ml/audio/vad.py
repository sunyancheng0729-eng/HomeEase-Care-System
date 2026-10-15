"""Voice activity detection for the speech branch only (10 ms frame decisions).

Environmental sounds (impacts, glass) are NOT filtered by VAD: the sound-event
branch reads its own audio windows independently of these decisions.
"""

from __future__ import annotations

import numpy as np

from homeease.ml.audio.features import SR

HOP = 160  # 10 ms at 16 kHz


class EnergyVAD:
    """Baseline: adaptive noise floor (low percentile of frame energy) + margin + hangover."""

    name = "energy"

    def __init__(self, margin_db: float = 9.0, hangover: int = 8, min_speech: int = 5, max_floor_db: float = -50.0) -> None:
        self.margin_db, self.hangover, self.min_speech, self.max_floor_db = margin_db, hangover, min_speech, max_floor_db

    def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        n = len(x) // HOP
        if n == 0:
            return np.zeros(0, dtype=np.int8)
        fr = x[: n * HOP].reshape(n, HOP).astype(np.float64)
        e = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-10)
        # Noise-floor estimate from quiet frames, capped so that a window filled with speech
        # (no quiet frames) still exceeds the threshold.
        floor = min(float(np.percentile(e, 10)), self.max_floor_db)
        raw = e > floor + self.margin_db
        out = np.zeros(n, dtype=np.int8)
        hang = 0
        for i, v in enumerate(raw):
            if v:
                hang = self.hangover
                out[i] = 1
            elif hang > 0:
                hang -= 1
                out[i] = 1
        # remove very short bursts (clicks)
        i = 0
        while i < n:
            if out[i]:
                j = i
                while j < n and out[j]:
                    j += 1
                if j - i < self.min_speech:
                    out[i:j] = 0
                i = j
            else:
                i += 1
        return out


class WebRTCVAD:
    name = "webrtc"

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)

    def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        n30 = 480
        out = []
        for i in range(0, len(pcm) - n30 + 1, n30):
            v = self.vad.is_speech(pcm[i : i + n30].tobytes(), sr)
            out += [1 if v else 0] * 3
        n = len(x) // HOP
        out = (out + [0] * n)[:n]
        return np.asarray(out, dtype=np.int8)


class SileroVAD:
    """Silero VAD (MIT) via the silero-vad package (TorchScript model)."""

    name = "silero"

    def __init__(self, threshold: float = 0.5) -> None:
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        self.th = threshold

    def probs(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        import torch

        self.model.reset_states()
        chunk = 512
        ps = []
        with torch.no_grad():
            for i in range(0, len(x) - chunk + 1, chunk):
                ps.append(float(self.model(torch.from_numpy(x[i : i + chunk].astype(np.float32)), sr)))
        n = len(x) // HOP
        if not ps:
            return np.zeros(n)
        # map each 10 ms frame to the 32 ms chunk containing its centre (no cumulative drift)
        centre = np.arange(n) * HOP + HOP // 2
        idx = np.minimum(centre // chunk, len(ps) - 1)
        return np.asarray(ps)[idx]

    def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        return (self.probs(x, sr) >= self.th).astype(np.int8)


def create(name: str) -> object:
    return {"energy": EnergyVAD, "webrtc": WebRTCVAD, "silero": SileroVAD}[name]()


def segments(frames: np.ndarray, min_gap: int = 20) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    i, n = 0, len(frames)
    while i < n:
        if frames[i]:
            j = i
            while j < n and frames[j]:
                j += 1
            if out and i * 10 - out[-1][1] * 1000 < min_gap * 10:
                out[-1] = (out[-1][0], j / 100.0)
            else:
                out.append((i / 100.0, j / 100.0))
            i = j
        else:
            i += 1
    return out
