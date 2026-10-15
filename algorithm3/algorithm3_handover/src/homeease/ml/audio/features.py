"""Audio front-end: loading, resampling, log-mel, MFCC and Kaldi-style fbank (numpy only)."""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
from scipy import signal as sps

SR = 16000


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    import soundfile as sf

    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return resample(x, fs, sr)


def resample(x: np.ndarray, fs: int, sr: int = SR) -> np.ndarray:
    if fs == sr:
        return x.astype(np.float32)
    g = gcd(int(fs), int(sr))
    return sps.resample_poly(x, sr // g, fs // g).astype(np.float32)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(np.asarray(m) / 1127.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(n_fft: int, n_mels: int, sr: int = SR, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    """Triangular filters on the HTK/Kaldi mel scale (1127 ln(1 + f/700))."""
    fmax = fmax or sr / 2.0
    bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2.0, bins)
    mpts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    m_f = _hz_to_mel(freqs)
    fb = np.zeros((n_mels, bins))
    for i in range(n_mels):
        lo, ce, hi = mpts[i], mpts[i + 1], mpts[i + 2]
        up = (m_f - lo) / (ce - lo)
        down = (hi - m_f) / (hi - ce)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def frames(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def logmel(x: np.ndarray, sr: int = SR, n_mels: int = 64, win_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    win, hop = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    n_fft = 1 << (win - 1).bit_length()
    fr = frames(x, win, hop) * np.hanning(win)[None, :]
    spec = np.abs(np.fft.rfft(fr, n=n_fft, axis=1)) ** 2
    mel = spec @ mel_filterbank(n_fft, n_mels, sr).T
    return np.log(mel + 1e-10).astype(np.float32)  # (T, n_mels)


def mfcc(x: np.ndarray, sr: int = SR, n_mfcc: int = 20, n_mels: int = 40) -> np.ndarray:
    from scipy.fft import dct

    lm = logmel(x, sr, n_mels)
    return dct(lm, type=2, axis=1, norm="ortho")[:, :n_mfcc].astype(np.float32)


def kaldi_fbank(x: np.ndarray, sr: int = SR, n_mels: int = 80, frame_ms: float = 25.0, shift_ms: float = 10.0, preemph: float = 0.97) -> np.ndarray:
    """Approximation of Kaldi fbank (povey window, DC removal, pre-emphasis, power spectrum,
    log mel energies) as used by WeSpeaker; input scaled to 16-bit range like Kaldi."""
    x = np.asarray(x, dtype=np.float64) * 32768.0
    win, hop = int(sr * frame_ms / 1000), int(sr * shift_ms / 1000)
    fr = frames(x, win, hop).copy()
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= preemph * fr[:, :-1]
    fr[:, 0] -= preemph * fr[:, 0]
    povey = np.power(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1)), 0.85)
    fr *= povey[None, :]
    n_fft = 1 << (win - 1).bit_length()
    spec = np.abs(np.fft.rfft(fr, n=n_fft, axis=1)) ** 2
    mel = spec @ mel_filterbank(n_fft, n_mels, sr, 20.0, sr / 2.0).T
    feats = np.log(np.maximum(mel, np.finfo(np.float64).eps))
    return feats.astype(np.float32)


def stats_vector(feat: np.ndarray) -> np.ndarray:
    """Clip-level summary: mean, std, max and delta-std per coefficient."""
    d = np.diff(feat, axis=0) if len(feat) > 1 else np.zeros_like(feat)
    return np.concatenate([feat.mean(0), feat.std(0), feat.max(0), d.std(0)]).astype(np.float32)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / max(len(noise), 1))))
    off = int(rng.integers(0, max(1, len(noise) - len(speech) + 1)))
    n = noise[off : off + len(speech)]
    ps = float(np.mean(speech**2)) + 1e-12
    pn = float(np.mean(n**2)) + 1e-12
    return (speech + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))).astype(np.float32)


def simulate_loudspeaker(x: np.ndarray, sr: int = SR, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crude TV/replay simulation: band-limited 200-4000 Hz + short synthetic room reverb."""
    rng = rng or np.random.default_rng(0)
    sos = sps.butter(4, [200, 4000], btype="band", fs=sr, output="sos")
    y = sps.sosfilt(sos, x)
    ir_len = int(0.25 * sr)
    ir = rng.normal(0, 1, ir_len) * np.exp(-np.arange(ir_len) / (0.05 * sr))
    ir[0] = 1.0
    y = sps.fftconvolve(y, ir / np.abs(ir).sum() * 4.0)[: len(x)]
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)
