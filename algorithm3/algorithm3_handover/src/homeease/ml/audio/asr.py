"""Local ASR with faster-whisper (MIT; CTranslate2). No cloud service is used.

faster-whisper imports PyAV, whose bundled FFmpeg clashes with OpenCV's in one
process on macOS; the server therefore runs ASR in a separate spawned process
(`ASRProcess`). Research scripts that do not import cv2 may use `WhisperASR` directly.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import numpy as np

from homeease.core.paths import models_dir

DEFAULT_MODEL = "faster-whisper-tiny"
TEMPERATURE_ENV = "HOMEEASE_ASR_TEMPERATURE"  # research runs: "0" = deterministic decoding (see WhisperASR)


def env_temperature() -> float | None:
    v = os.environ.get(TEMPERATURE_ENV, "").strip()
    return float(v) if v else None


class WhisperASR:
    """faster-whisper decoding. By default faster-whisper retries a decode that looks bad (compression ratio, low avg_logprob: typically noise
    or a hallucination) at temperatures 0.2 ... 1.0 with RANDOM sampling, so two identical calls can return different text. That is the measured
    behaviour of the experiments (A3, A3b, A6 numbers move by about 0.01 macro-F1 and a clip or two in the noise probe between runs). ``temperature``
    (or the environment variable ``HOMEEASE_ASR_TEMPERATURE``, or ``audio.asr_temperature`` in the product) fixes ONE temperature and switches the
    fallback sampling off: with 0.0 the same input always gives the same text. None keeps the library default."""

    temperature: float | None = None  # class default, so an object built without __init__ (tests) keeps the library default

    def __init__(self, model: str = DEFAULT_MODEL, compute_type: str = "int8", temperature: float | None = None) -> None:
        from faster_whisper import WhisperModel

        self.temperature = temperature if temperature is not None else env_temperature()

        path = models_dir() / "audio" / model
        if not (path / "model.bin").is_file():
            raise FileNotFoundError(f"ASR model not found at {path}; run the audio model download (docs/data_sources.md)")
        self.model = WhisperModel(str(path), device="cpu", compute_type=compute_type)
        self.model_id = model

    def transcribe(self, audio: np.ndarray, language: str | None = None, *, hotwords: str | None = None, initial_prompt: str | None = None,
                   beam_size: int = 1, restrict_languages: tuple[str, ...] | None = None) -> dict[str, Any]:
        """``hotwords`` / ``initial_prompt`` bias decoding toward a vocabulary (both end up in the Whisper prompt).
        ``restrict_languages`` (with language=None): automatic language ID, but only among these languages."""
        x = audio.astype(np.float32)
        lang_source = "forced" if language else "auto"
        if language is None and restrict_languages:
            _, _, probs = self.model.detect_language(x)
            allowed = {l: p for l, p in probs if l in restrict_languages}
            if allowed:
                language = max(allowed, key=lambda k: allowed[k])
                lang_source = "auto_restricted"
        extra = {} if self.temperature is None else {"temperature": float(self.temperature)}
        segs, info = self.model.transcribe(x, language=language, beam_size=beam_size, vad_filter=False, condition_on_previous_text=False,
                                           hotwords=hotwords, initial_prompt=initial_prompt, **extra)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        avg_lp = float(np.mean([s.avg_logprob for s in segs])) if segs else None
        nsp = float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0
        return {"text": text, "language": info.language, "language_prob": float(info.language_probability), "language_source": lang_source,
                "avg_logprob": avg_lp, "no_speech_prob": nsp, "model": self.model_id, "temperature": self.temperature}


_worker: WhisperASR | None = None


def _init(model: str, temperature: float | None = None) -> None:
    global _worker
    _worker = WhisperASR(model, temperature=temperature)


def _run(audio: np.ndarray, language: str | None, opts: dict[str, Any] | None = None) -> dict[str, Any]:
    assert _worker is not None
    return _worker.transcribe(audio, language, **(opts or {}))


class ASRProcess:
    """One spawned worker process holding the ASR model (isolates PyAV from OpenCV)."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float | None = None) -> None:
        self._model, self._temperature = model, temperature
        self.pool = self._new_pool()

    def _new_pool(self) -> cf.ProcessPoolExecutor:
        return cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(self._model, self._temperature))

    def transcribe(self, audio: np.ndarray, language: str | None = None, timeout: float = 30.0, **decode: Any) -> dict[str, Any]:
        try:
            return self.pool.submit(_run, audio, language, decode or None).result(timeout=timeout)
        except BrokenProcessPool:
            # The worker process died (memory, a crash in the native library). A broken pool refuses every later call, so a node that runs for
            # weeks would stay deaf until restarted: the next call gets a new worker. This call is reported as failed, it is not retried.
            old, self.pool = self.pool, self._new_pool()
            old.shutdown(wait=False, cancel_futures=True)
            raise

    def close(self) -> None:
        self.pool.shutdown(cancel_futures=True)
