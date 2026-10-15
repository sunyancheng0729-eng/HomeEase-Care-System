"""Facade over the Algorithm 3 modules. No database, no network, nothing is written; audio stays in memory.

The heavy objects (sound classifier, speech recogniser, voice activity detector) are built on first use. One call is processed at a time
(``Algorithm3`` holds a lock), which is what the demo and a simple integration need.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np

from homeease.core.clock import VirtualClock
from homeease.core.config import AppConfig, load_config
from homeease.ml.audio.features import SR
from homeease.ml.audio.monitor import SoundMonitor
from homeease.ml.audio.pipeline import Function3Pipeline, SpeechSegment

from algorithm3 import policy
from algorithm3.decode import decode_bytes, to_mono16k

_T0_MONO_MS = 1000.0
_T0_UTC = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


class Algorithm3:
    """``Algorithm3(vad="auto")``: ``vad`` is "auto" (Silero when torch is installed, otherwise the energy detector), "silero" or "energy"."""

    def __init__(self, vad: str = "auto", overrides: dict[str, Any] | None = None) -> None:
        base: dict[str, Any] = {"audio": {"enabled": True, "sound_monitor": {"enabled": True}, "help_listener": {"enabled": True}}}
        if overrides:
            base = _merge(base, overrides)
        self.cfg: AppConfig = load_config(None, base)
        self._vad_pref = vad
        self._lock = threading.RLock()
        self._classifier: Any = None
        self._classifier_error: str | None = None
        self._asr: Any = None
        self._asr_error: str | None = None
        self._vad: Any = None
        self.vad_name: str | None = None
        self.vad_note: str | None = None

    # ------------------------------------------------------------------ models (lazy)
    def _get_vad(self) -> Any:
        if self._vad is not None:
            return self._vad
        from homeease.ml.audio import vad as vadmod

        want = self.cfg.audio.vad if self._vad_pref == "auto" else self._vad_pref
        try:
            self._vad = vadmod.create(want)
            self.vad_name, self.vad_note = want, None
        except Exception as e:  # torch / silero-vad not installed
            self._vad = vadmod.create("energy")
            self.vad_name = "energy"
            self.vad_note = f"{want} unavailable ({type(e).__name__}); the energy detector is used"
        return self._vad

    def _get_classifier(self) -> Any:
        if self._classifier is None and self._classifier_error is None:
            try:
                from homeease.ml.audio.sound_package import load_sound_classifier

                self._classifier = load_sound_classifier(self.cfg.audio.sound_monitor.model_id)
            except Exception as e:
                self._classifier_error = f"{type(e).__name__}: {e}"
        if self._classifier is None:
            raise RuntimeError(f"sound model not available: {self._classifier_error}")
        return self._classifier

    def _get_asr(self) -> Any:
        if self._asr is None and self._asr_error is None:
            try:
                from homeease.ml.audio.asr import WhisperASR

                self._asr = WhisperASR()
            except Exception as e:
                self._asr_error = f"{type(e).__name__}: {e}"
        if self._asr is None:
            raise RuntimeError(f"speech recogniser not available: {self._asr_error}")
        return self._asr

    def warm_up(self) -> dict[str, Any]:
        """Load every model now (a few seconds). Returns what is available."""
        with self._lock:
            self._get_vad()
            for fn in (self._get_classifier, self._get_asr):
                try:
                    fn()
                except Exception:
                    pass
        return self.info()

    def info(self) -> dict[str, Any]:
        sm = self.cfg.audio.sound_monitor
        return {
            "vad": self.vad_name, "vad_note": self.vad_note,
            "sound_model": getattr(self._classifier, "model_id", None) and f"{self._classifier.model_id}@{getattr(self._classifier, 'model_version', '?')}",
            "sound_model_error": self._classifier_error,
            "asr_model": getattr(self._asr, "model_id", None), "asr_error": self._asr_error,
            "intent_method": self.cfg.audio.intent_method,
            "gates": {"onset_db": self.cfg.audio.gate.onset_db, "speech_min_s": self.cfg.audio.gate.speech_min_s,
                      "speech_end_silence_s": self.cfg.audio.gate.speech_end_silence_s},
            "sound": {"window_s": sm.window_s, "hop_s": sm.hop_s, "confirm_windows": sm.confirm_windows, "cooldown_s": sm.cooldown_s},
            "help_listener": {"min_utterance_s": self.cfg.audio.help_listener.min_utterance_s,
                              "max_utterance_s": self.cfg.audio.help_listener.max_utterance_s,
                              "cooldown_s": self.cfg.audio.help_listener.cooldown_s,
                              "min_asr_logprob": self.cfg.audio.help_listener.min_asr_logprob},
            "inquiry_response_window_s": self.cfg.audio.inquiry_response_window_s,
            "escalate_on_no_cancel": self.cfg.audio.escalate_on_no_cancel,
        }

    # ------------------------------------------------------------------ input
    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Bytes of an audio file (wav, flac, ogg, webm, mp3 ...) -> float32 mono 16 kHz."""
        return decode_bytes(data)

    # ------------------------------------------------------------------ 1. daily sound labels
    def label_sounds(self, samples: Any, sr: int = SR) -> dict[str, Any]:
        """Run a recording through the live code path (ring buffer, two gates, classifier, event labelling).

        Returns ``events`` (category, confidence, start_s, end_s, level; no audio), ``speech`` (segments cut by the speech gate: time
        and length only) and ``status``. Labels are assistive hints and never raise or cancel an alarm."""
        x = to_mono16k(samples, sr)
        with self._lock:
            clf = self._get_classifier()
            vad = self._get_vad()
            clock = VirtualClock(_T0_MONO_MS, _T0_UTC)
            mon = SoundMonitor(self.cfg.audio.sound_monitor, clf, clock, store=None)
            pipe = Function3Pipeline(self.cfg.audio, clock, mon, voice_sink=_wipe_sink, vad=vad)
            try:
                rep = pipe.process_array(x, SR, t0_mono_ms=_T0_MONO_MS)
            finally:
                pipe.close()
        events = [{"category": e.category, "confidence": round(e.confidence, 3), "start_s": round((e.t_start_mono_ms - _T0_MONO_MS) / 1000.0, 2),
                   "end_s": round((e.t_end_mono_ms - _T0_MONO_MS) / 1000.0, 2), "duration_s": round(e.duration_s, 2),
                   "rms_db": round(e.rms_db, 1), "peak_db": round(e.peak_db, 1), "windows": e.n_windows, "model": e.model_id}
                  for e in rep.events]
        speech = [{"start_s": round((s["t_start_mono_ms"] - _T0_MONO_MS) / 1000.0, 2), "end_s": round((s["t_end_mono_ms"] - _T0_MONO_MS) / 1000.0, 2),
                   "duration_s": s["duration_s"]} for s in rep.speech]
        return {"duration_s": round(rep.duration_s, 2), "events": events, "speech": speech, "vad": self.vad_name, "vad_note": self.vad_note,
                "status": rep.status.get("status"), "raw_audio_stored": False}

    # ------------------------------------------------------------------ 2. reply to the question
    def classify_reply(self, audio: Any = None, sr: int = SR, text: str | None = None, enrollment: Any = None) -> dict[str, Any]:
        """Read one reply. ``audio`` is the recording of the answer window, ``text`` a transcript (skips the recogniser).

        Returns a reply record with ``status`` (RESPONDED | NO_RESPONSE | audio_unavailable), ``intent`` (HELP | REASSURANCE | OTHER |
        UNCERTAIN), ``intent_reason``, ``cues``, ``transcript``, ``asr_avg_logprob``, the voiceprint evidence (``cancel_eligible``,
        ``cancel_reasons``) and ``trusted_reassurance`` (the node's valid-cancel test)."""
        from homeease.ml.audio.intent_v2 import RuleIntentV2

        rec: dict[str, Any] = {"status": "RESPONDED", "intent": None, "transcript": None, "asr_avg_logprob": None,
                               "cancel_eligible": False, "cancel_reasons": ["no_enrollment"], "speaker_verified": None,
                               "voiceprint_enrolled": enrollment is not None}
        if text is not None:
            res = {"text": text, "avg_logprob": None}
            x = None
        else:
            x = to_mono16k(audio, sr)
            from homeease.ml.audio.vad import EnergyVAD, segments

            if not segments(EnergyVAD().frames(x)):
                rec.update(status="NO_RESPONSE", intent=None, cancel_reasons=["no_speech"])
                rec["trusted_reassurance"] = False
                return rec
            try:
                with self._lock:
                    res = self._get_asr().transcribe(x, None)
            except Exception as e:
                rec.update(status="audio_unavailable", error=f"asr failed: {type(e).__name__}", cancel_reasons=["asr_failed"])
                rec["trusted_reassurance"] = False
                return rec
        intent_model = RuleIntentV2()
        it = intent_model.predict(res["text"], res.get("avg_logprob"))
        cues = it.get("cues") or []
        rec.update(intent=it["intent"], intent_reason=it.get("reason"), cues=list(cues), transcript=(res["text"] or "")[:200],
                   asr_avg_logprob=res.get("avg_logprob"), language=res.get("language"))
        if x is not None:
            try:
                from homeease.ml.audio.cancel_gate import CancelGate

                gate = CancelGate.from_config(self.cfg.audio)
                dec = gate.evaluate(x, it, enrollment, res.get("avg_logprob"))
                rec.update(dec.to_record())
            except Exception as e:  # the voiceprint part is evidence only: its failure never changes the intent
                rec["cancel_reasons"] = [f"gate_error:{type(e).__name__}"]
        rec["trusted_reassurance"] = policy.voice_cancel_verdict(rec, self.cfg.audio.voiceprint.min_asr_logprob)
        return rec

    # ------------------------------------------------------------------ 3. spoken call for help
    def detect_spoken_help(self, samples: Any, sr: int = SR) -> dict[str, Any]:
        """Run a recording through the speech gate and the listener rules (0.3 to 6 s, recogniser log-probability at least -1.0, a clear
        call, 30 s cooldown). Returns every speech segment with its verdict (``help`` or the reason it was ignored)."""
        from homeease.ml.audio.help_listener import HelpListener

        x = to_mono16k(samples, sr)
        segs: list[SpeechSegment] = []
        with self._lock:
            vad = self._get_vad()
            asr = self._get_asr()
            clock = VirtualClock(_T0_MONO_MS, _T0_UTC)
            calls: list[dict[str, Any]] = []
            listener = HelpListener(self.cfg, clock, on_help=calls.append, consent=lambda: True, asr=asr)
            pipe = Function3Pipeline(self.cfg.audio, clock, None, voice_sink=segs.append, vad=vad)
            try:
                pipe.process_array(x, SR, t0_mono_ms=_T0_MONO_MS)
            finally:
                pipe.close()
            judged: list[dict[str, Any]] = []
            for seg in segs:
                clock.set(seg.t_end_mono_ms)  # stream time, so the cooldown behaves as it does on a live stream
                verdict = listener.process(seg)
                row = {"start_s": round((seg.t_start_mono_ms - _T0_MONO_MS) / 1000.0, 2), "duration_s": seg.duration_s, "verdict": verdict}
                if verdict == "help" and calls:
                    d = calls[-1]["details"]
                    row.update(cues=d.get("cues"), asr_avg_logprob=d.get("asr_avg_logprob"))
                judged.append(row)
        return {"segments": judged, "help": any(j["verdict"] == "help" for j in judged), "vad": self.vad_name, "vad_note": self.vad_note,
                "raw_audio_stored": False}

    # ------------------------------------------------------------------ 4. decisions
    inquiry_outcome = staticmethod(policy.inquiry_outcome)
    alarm_tag = staticmethod(policy.alarm_tag)


def _wipe_sink(seg: SpeechSegment) -> None:
    """The sound demo only needs the number and length of the speech segments: the samples are zeroed at once."""
    try:
        seg.samples[:] = 0.0
    except Exception:
        pass


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
