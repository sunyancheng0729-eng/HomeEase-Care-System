"""Voice inquiry: reply-intent method from the config (rule_v2 default, v1 selectable), policy semantics and raw-audio discard.
Stub runtime / recorder / ASR / TTS - no microphone, speech engine or model."""

from __future__ import annotations

import threading
import types
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from homeease.core.config import AudioCfg
from homeease.ml.audio.inquiry import AudioInquiryService

SR = 16000


class Events:
    def __init__(self) -> None:
        self.recs: list[dict[str, Any]] = []

    def voice_response(self, event_id: str, rec: dict[str, Any]) -> None:
        self.recs.append(rec)


class Reply:
    """A scripted ASR: returns a fixed transcript, or raises."""

    def __init__(self, text: str | None = None, error: Exception | None = None, logprob: float = -0.2) -> None:
        self.text, self.error, self.logprob = text, error, logprob
        self.calls: list[tuple[Any, ...]] = []

    def transcribe(self, audio: np.ndarray, language: Any) -> dict[str, Any]:  # exactly the two-argument shape the service has always used
        self.calls.append((audio.shape, language))
        if self.error:
            raise self.error
        return {"text": self.text, "avg_logprob": self.logprob, "model": "scripted"}


def speech_like(seconds: float, sr: int = SR) -> np.ndarray:
    n = int(seconds * sr)
    return (0.3 * np.sin(np.arange(n) / 5) * (np.arange(n) > sr)).astype(np.float32)


def make(intent_method: str | None = None, text: str | None = "I'm fine", recorder: Any = None, asr: Any = None, spoken: bool = True) -> tuple[AudioInquiryService, Events, list[np.ndarray]]:
    taken: list[np.ndarray] = []

    def rec_fn(seconds: float, sr: int = SR) -> np.ndarray:
        x = speech_like(seconds)
        taken.append(x)
        return x

    cfg = AudioCfg(**({"intent_method": intent_method} if intent_method else {}))
    ev = Events()
    svc = AudioInquiryService.__new__(AudioInquiryService)
    svc.rt = types.SimpleNamespace(clock=types.SimpleNamespace(utc_now=lambda: datetime(2026, 10, 8, tzinfo=timezone.utc)), events=ev, subject_id=1, broadcaster=None)  # type: ignore[assignment]
    svc.cfg = types.SimpleNamespace(audio=cfg)  # type: ignore[assignment]
    svc._state = {"enabled": True, "consent": True, "microphone": True, "last": None, "errors": []}
    svc.recorder = recorder or rec_fn
    svc.asr = asr if asr is not None else Reply(text)
    svc.tts = lambda t, lang: spoken
    svc._busy = threading.Lock()
    return svc, ev, taken


def test_default_intent_method_is_rule_v2_and_v1_stays_selectable() -> None:
    assert AudioCfg().intent_method == "rule_v2" and AudioCfg(intent_method="rule").intent_method == "rule"
    with pytest.raises(ValidationError):
        AudioCfg(intent_method="combined")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AudioCfg(auto_cancel_emergency=True)  # still impossible


@pytest.mark.parametrize("text,v2,v1", [
    ("I'm not hurt", "REASSURANCE", "HELP"),  # v1 reads the word 'hurt'
    ("\u6c92\u4e8b", "REASSURANCE", "OTHER"),  # Whisper writes Mandarin in Traditional characters
    ("\u6211\u8dcc\u5012\u4e86", "HELP", "OTHER"),
    ("Ouch", "HELP", "OTHER"),
    ("I'm fine but I can't get up", "HELP", "HELP"),
    ("What time is it", "OTHER", "OTHER"),
])
def test_method_from_config_decides_the_intent(text: str, v2: str, v1: str) -> None:
    for method, want in (("rule_v2", v2), ("rule", v1), (None, v2)):
        svc, ev, _ = make(method, text)
        rec = svc.run_inquiry("e1")
        assert rec["status"] == "RESPONDED" and rec["intent"] == want, (method, text, rec)
        assert rec["intent_method"] == (method or "rule_v2") and ev.recs == [rec]
        assert rec["speaker_verified"] is None and "not verified" in rec["source_note"]  # intent is evidence about the words, not about who spoke


def test_asr_is_called_the_way_it_always_was() -> None:
    asr = Reply("help")
    svc, _, _ = make(asr=asr)
    svc.run_inquiry("e1")
    assert asr.calls == [((int(10.0 * SR),), None)]  # plain decoding, automatic language (A3b: no bias / forced language adopted)


def test_failures_are_never_read_as_safe() -> None:
    svc, ev, _ = make(asr=Reply(error=RuntimeError("model crashed")))
    rec = svc.run_inquiry("e1")
    assert rec["status"] == "audio_unavailable" and rec.get("intent") is None and "asr failed" in rec["error"]  # the recogniser failed: nobody was "unsure"

    svc, ev, _ = make(text="")
    rec = svc.run_inquiry("e1")
    assert rec["intent"] == "UNCERTAIN" and rec["intent_reason"] == "empty_transcript"

    svc, ev, _ = make(recorder=lambda s, sr=SR: np.zeros(int(s * sr), dtype=np.float32))
    rec = svc.run_inquiry("e1")
    assert rec["status"] == "NO_RESPONSE" and rec["intent"] is None

    def broken(s: float, sr: int = SR) -> np.ndarray:
        raise RuntimeError("microphone unplugged")

    svc, ev, _ = make(recorder=broken)
    rec = svc.run_inquiry("e1")
    assert rec["status"] == "audio_unavailable" and rec.get("intent") is None


def test_prompt_failure_is_recorded_but_listening_goes_on() -> None:
    svc, ev, _ = make(text="help", spoken=False)
    rec = svc.run_inquiry("e1")
    assert rec["prompt_played"] is False and "TTS playback failed" in rec["prompt_error"] and rec["intent"] == "HELP"


def test_raw_samples_are_discarded_on_every_path() -> None:
    for kwargs in ({"text": "I'm fine"}, {"asr": Reply(error=RuntimeError("x"))}, {"text": "hello", "asr": Reply("hello")}):
        svc, _, taken = make(**kwargs)
        svc.run_inquiry("e1")
        assert taken and not np.any(taken[0])  # zeroed after the reply, after an ASR failure

    def silent(s: float, sr: int = SR) -> np.ndarray:
        x = np.zeros(int(s * sr), dtype=np.float32)
        x[:10] = 1e-9  # still below the VAD
        silent.buf = x  # type: ignore[attr-defined]
        return x

    svc, _, _ = make(recorder=silent)
    assert svc.run_inquiry("e1")["status"] == "NO_RESPONSE" and not np.any(silent.buf)  # type: ignore[attr-defined]


def test_a_read_only_recorder_buffer_does_not_break_the_inquiry() -> None:
    def frozen(s: float, sr: int = SR) -> np.ndarray:
        x = speech_like(s)
        x.setflags(write=False)
        return x

    svc, ev, _ = make(recorder=frozen, text="help")
    assert svc.run_inquiry("e1")["intent"] == "HELP" and len(ev.recs) == 1


def test_the_record_carries_labels_and_a_short_transcript_but_no_audio() -> None:
    svc, ev, _ = make(text="I'm fine " * 40)
    rec = svc.run_inquiry("e1")
    assert len(rec["transcript"]) <= 100
    assert not any(isinstance(v, np.ndarray) for v in rec.values())
    assert all(isinstance(v, (str, bool, int, float, type(None))) or (isinstance(v, list) and all(isinstance(x, str) for x in v)) for v in rec.values())  # scalars, or the list of reason codes
