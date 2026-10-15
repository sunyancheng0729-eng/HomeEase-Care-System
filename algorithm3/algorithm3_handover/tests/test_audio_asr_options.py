"""WhisperASR decoding options (hotwords / prompt / language restriction) and their path through ASRProcess - fake model, no download."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from homeease.ml.audio import asr as A


class FakeModel:
    def __init__(self, lang_probs: list[tuple[str, float]] | None = None, segments: list[Any] | None = None) -> None:
        self.lang_probs = lang_probs or [("en", 0.6), ("zh", 0.3), ("ms", 0.07), ("ta", 0.03)]
        self.segments = segments if segments is not None else [types.SimpleNamespace(text=" I fell ", avg_logprob=-0.4, no_speech_prob=0.1)]
        self.transcribe_kw: dict[str, Any] = {}
        self.detect_calls = 0

    def detect_language(self, audio: np.ndarray, **kw: Any) -> tuple[str, float, list[tuple[str, float]]]:
        self.detect_calls += 1
        return self.lang_probs[0][0], self.lang_probs[0][1], self.lang_probs

    def transcribe(self, audio: np.ndarray, **kw: Any) -> tuple[list[Any], Any]:
        self.transcribe_kw = kw
        lang = kw.get("language") or self.lang_probs[0][0]
        return iter(self.segments), types.SimpleNamespace(language=lang, language_probability=0.8)


def _asr(model: FakeModel) -> A.WhisperASR:
    a = A.WhisperASR.__new__(A.WhisperASR)  # skip the real model load
    a.model = model  # type: ignore[assignment]
    a.model_id = "faster-whisper-fake"
    return a


X = np.zeros(16000, dtype=np.float64)


def test_defaults_keep_the_original_decoding() -> None:
    m = FakeModel()
    r = _asr(m).transcribe(X, None)
    kw = m.transcribe_kw
    assert kw["language"] is None and kw["beam_size"] == 1 and kw["vad_filter"] is False and kw["condition_on_previous_text"] is False
    assert kw["hotwords"] is None and kw["initial_prompt"] is None and m.detect_calls == 0
    assert r["text"] == "I fell" and r["language_source"] == "auto" and r["model"] == "faster-whisper-fake" and r["avg_logprob"] == pytest.approx(-0.4)


def test_forced_language_is_passed_through() -> None:
    m = FakeModel()
    r = _asr(m).transcribe(X, "zh")
    assert m.transcribe_kw["language"] == "zh" and r["language_source"] == "forced" and m.detect_calls == 0


def test_hotwords_and_initial_prompt_reach_the_model() -> None:
    m = FakeModel()
    _asr(m).transcribe(X, "en", hotwords="help, ambulance", initial_prompt="Safety check.", beam_size=3)
    assert m.transcribe_kw["hotwords"] == "help, ambulance" and m.transcribe_kw["initial_prompt"] == "Safety check." and m.transcribe_kw["beam_size"] == 3


def test_restricted_language_id_picks_the_best_allowed_language() -> None:
    m = FakeModel([("ms", 0.5), ("en", 0.2), ("zh", 0.25), ("ta", 0.05)])  # unrestricted ID would say Malay
    r = _asr(m).transcribe(X, None, restrict_languages=("en", "zh"))
    assert m.detect_calls == 1 and m.transcribe_kw["language"] == "zh" and r["language_source"] == "auto_restricted"


def test_restricted_language_id_falls_back_to_the_models_own_choice_when_nothing_is_allowed() -> None:
    m = FakeModel([("ms", 0.7), ("ta", 0.3)])
    r = _asr(m).transcribe(X, None, restrict_languages=("en", "zh"))
    assert m.transcribe_kw["language"] is None and r["language_source"] == "auto"


def test_restriction_is_ignored_when_the_language_is_forced() -> None:
    m = FakeModel()
    _asr(m).transcribe(X, "en", restrict_languages=("zh",))
    assert m.detect_calls == 0 and m.transcribe_kw["language"] == "en"


def test_no_segments_means_empty_text_and_certain_silence_not_confidence() -> None:
    m = FakeModel(segments=[])
    r = _asr(m).transcribe(X, None)
    assert r["text"] == "" and r["avg_logprob"] is None and r["no_speech_prob"] == 1.0


def test_worker_run_forwards_the_decoding_options(monkeypatch: pytest.MonkeyPatch) -> None:
    m = FakeModel()
    monkeypatch.setattr(A, "_worker", _asr(m))
    A._run(X, "en", {"hotwords": "help", "beam_size": 2})
    assert m.transcribe_kw["hotwords"] == "help" and m.transcribe_kw["beam_size"] == 2
    A._run(X, None)  # the old call shape still works
    assert m.transcribe_kw["hotwords"] is None


def test_asr_process_wrapper_keeps_its_old_positional_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Pool:
        def submit(self, fn: Any, audio: Any, language: Any, opts: Any) -> Any:
            seen.update(fn=fn, language=language, opts=opts)
            return types.SimpleNamespace(result=lambda timeout: {"text": "ok", "timeout": timeout})

    p = A.ASRProcess.__new__(A.ASRProcess)
    p.pool = Pool()  # type: ignore[assignment]
    assert p.transcribe(X, "en", 5.0)["timeout"] == 5.0 and seen["opts"] is None
    p.transcribe(X, None, hotwords="help", restrict_languages=("en", "zh"))
    assert seen["fn"] is A._run and seen["opts"] == {"hotwords": "help", "restrict_languages": ("en", "zh")}


# ------------------------------------------------------------------------------------------------ review (L2-05): the source of run-to-run variation is switchable
def test_by_default_no_temperature_is_passed_so_the_library_fallback_sampling_stays_as_measured() -> None:
    m = FakeModel()
    r = _asr(m).transcribe(X, None)
    assert "temperature" not in m.transcribe_kw and r["temperature"] is None


def test_a_fixed_temperature_switches_the_fallback_sampling_off() -> None:
    m = FakeModel()
    a = _asr(m)
    a.temperature = 0.0
    r = a.transcribe(X, None)
    assert m.transcribe_kw["temperature"] == 0.0 and r["temperature"] == 0.0  # a single temperature, not the 0.0 ... 1.0 ladder


def test_the_environment_variable_and_the_config_reach_the_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(A.TEMPERATURE_ENV, "0")
    assert A.env_temperature() == 0.0
    monkeypatch.setenv(A.TEMPERATURE_ENV, "")
    assert A.env_temperature() is None
    monkeypatch.delenv(A.TEMPERATURE_ENV)
    assert A.env_temperature() is None
    from homeease.core.config import AudioCfg

    assert AudioCfg().asr_temperature is None and AudioCfg(asr_temperature=0.0).asr_temperature == 0.0
    with pytest.raises(Exception):
        AudioCfg(asr_temperature=1.5)


def test_the_worker_process_gets_the_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class FakeASR:
        def __init__(self, model: str, temperature: float | None = None) -> None:
            seen.update(model=model, temperature=temperature)

    monkeypatch.setattr(A, "WhisperASR", FakeASR)
    monkeypatch.setattr(A, "_worker", None)  # _init sets the module global: monkeypatch puts the old value back
    A._init("faster-whisper-tiny", 0.0)
    assert seen == {"model": "faster-whisper-tiny", "temperature": 0.0}
