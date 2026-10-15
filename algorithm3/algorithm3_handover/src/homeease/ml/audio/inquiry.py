"""Optional voice inquiry after a suspected fall (feature-flagged, consent-gated).

Product rules (proposal 6.3, revised slide 1, prompt 9.1):
* Notifications are sent immediately; the inquiry never delays them.
* A reply is supporting evidence only. REASSURANCE never closes or downgrades an event.
  An explicit HELP reply raises PENDING to EMERGENCY (escalation is the safe direction).
* The check ends in a cancel or a no-cancel (``EventService.voice_response``). A trusted REASSURANCE (matching voice when one is enrolled)
  or the elder page "I'm OK" button only stops the escalation; every other result, silence included, raises a PENDING event to EMERGENCY
  (``audio.escalate_on_no_cancel``). A check that could not be made neither cancels nor escalates: not_run (audio off, no consent) and
  audio_unavailable (microphone or recogniser failure) are failures of the audio layer, not answers. A question that was not delivered
  (no older adult page, no speech output) is not an answer either: only a spoken call for help counts then (``prompt_played`` False).
* Unclear speech or silence is recorded as UNCERTAIN / NO_RESPONSE, never "safe"; a recogniser that failed is audio_unavailable.
* The response window starts when the prompt has finished playing (a person who is still being asked is not "silent"); a reply that comes
  while the question is still asked is heard too (the question's own words are read as OTHER, never as a reassurance); prompt_id is kept.
* One check per event: a PENDING event that is decided again or upgraded later does not start a second one.
* The reply comes from the speech segments of the live microphone stream while that stream runs (the help listener's, see
  ``Runtime.voice_stream_ready``): a second input stream on the same device would fail or disturb it. Without a running stream
  the inquiry records on its own (``sd.rec``), as it always did.
* Raw audio is held only in memory for the inquiry window and then discarded.
* Consent is read again at the moment of use (when the event arrives and just before the microphone opens), not once at
  start-up: a withdrawal takes effect at once and a grant needs no restart. A second suspected fall while an inquiry is
  running gets a ``not_run: inquiry_busy`` marker instead of silently nothing.
* The reply transcript is not stored (``privacy.persist_voice_transcript``, default off): only intent, reason, cues.
* With a voiceprint enrollment the record also carries cancel-eligibility EVIDENCE (`cancel_eligible`, `speaker_score`,
  reason codes; ml/audio/cancel_gate.py). The only part acted on is the voiceprint verdict (`speaker_verified`), and only to decide
  whether a reply stops the escalation; an event keeps its level and status whatever it says (the proposal text, not Figure 3-3's
  auto-cancel box, is the policy).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from homeease.core.util import get_logger, new_id
from homeease.ml.audio import cancel_gate as cg

if TYPE_CHECKING:
    from homeease.ml.audio.pipeline import SpeechSegment

log = get_logger(__name__)

# "What is the matter?" in both languages (an open question: an answer is read for its intent, not as a yes or no). The order of
# ``audio.languages`` is the order in which they are spoken and shown.
PROMPTS = {"zh": "\u8bf7\u95ee\u4f60\u600e\u4e48\u5566\uff1f", "en": "What's wrong? Are you OK?"}
PLAYBACK_S = 4.0  # approximate playback time of one prompt on the elder device (browser speech synthesis cannot report its end)
REPLY_TAIL_S = 6.0  # a reply that began inside the response window may run on this much longer
REPLY_GAP_S = 0.25  # silence put between two speech segments of one reply (the recogniser sees separate utterances)
MAX_REPLY_S = 30.0  # longest reply handed to the recogniser
REMEMBERED_EVENTS = 256  # events whose inquiry state is kept (one check per event)


class _ReplyWindow:
    """The speech segments of the live stream that fall into one inquiry's response window."""

    def __init__(self, opened_ms: float) -> None:
        self.opened_ms = opened_ms
        self._segments: list[SpeechSegment] = []
        self._lock = threading.Lock()

    def add(self, seg: "SpeechSegment") -> None:
        with self._lock:
            self._segments.append(seg)

    def take(self) -> list["SpeechSegment"]:
        with self._lock:
            out, self._segments = self._segments, []
        return out

    def has_segments(self) -> bool:
        with self._lock:
            return bool(self._segments)


class AudioInquiryService:
    cancel_gate: Any = None  # class defaults, so a partially constructed service (tests build one with __new__) simply has no voiceprint
    enrollment: Any = None
    _window: Any = None  # the open _ReplyWindow of an inquiry that takes its reply from the live stream
    _asked: Any = None  # event id -> "running" | "done" | "not_run" | "busy": one check per event
    _stop_evt: Any = None
    _state_lock = threading.Lock()  # guards _window and _asked (shared by all instances: held for a few instructions only)

    def __init__(self, runtime: Any, recorder: Any = None, asr: Any = None, tts: Any = None, cancel_gate: Any = None, enrollment: Any = None) -> None:
        self.rt = runtime
        self.cfg = runtime.cfg
        self._state: dict[str, Any] = {"enabled": bool(self.cfg.audio.enabled), "consent": False, "microphone": None, "last": None, "errors": []}
        self.recorder = recorder or self._default_recorder
        self.asr = asr
        self.tts = tts
        self.cancel_gate = cancel_gate  # ml.audio.cancel_gate.CancelGate; built from audio.voiceprint on first use
        self.enrollment = enrollment  # ml.audio.cancel_gate.VoiceprintEnrollment; in memory only (persisting a voiceprint is a privacy decision, not made here)
        self._busy = threading.Lock()
        self._asked = {}
        self._stop_evt = threading.Event()
        self._check()

    # -------------------------------------------------------------- checks
    def _read_consent(self) -> bool:
        """The latest recorded audio consent, read now. Unknown (database error) is no consent."""
        try:
            fn = getattr(self.rt, "audio_consent", None)
            if callable(fn):
                ok = bool(fn())
            else:  # a minimal runtime stand-in without the helper
                from sqlalchemy import select

                from homeease.backend.storage.models import Consent

                with self.rt.db.session() as s:
                    c = s.execute(select(Consent).where(Consent.subject_id == self.rt.subject_id, Consent.kind == "audio").order_by(Consent.created_at.desc(), Consent.id.desc())).scalars().first()
                    ok = bool(c and c.granted)
        except Exception as e:
            log.warning("audio consent could not be read: %s", type(e).__name__)
            ok = False
        self._state["consent"] = ok
        return ok

    def _check(self) -> None:
        self._read_consent()
        try:
            import sounddevice as sd

            self._state["microphone"] = any(d.get("max_input_channels", 0) > 0 for d in sd.query_devices())
        except Exception as e:  # pragma: no cover - hardware dependent
            self._state["microphone"] = False
            self._state["errors"].append(f"microphone check failed: {e}")

    def status(self) -> dict[str, Any]:
        self._read_consent()  # never report a consent that was withdrawn (or granted) after start-up
        return {**self._state, "auto_cancel_emergency": self.cfg.audio.auto_cancel_emergency, "voiceprint_enrolled": self.enrollment is not None,
                "reply_source": "stream" if self._stream_ready() else "recorder", "listening": self._window is not None}

    def set_enrollment(self, enrollment: Any) -> None:
        """Install (or with None remove) the enrolled voiceprint. Memory only."""
        self.enrollment = enrollment

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()  # a response window that is being waited for ends now
        if self.asr is not None and hasattr(self.asr, "close"):
            self.asr.close()

    # ------------------------------------------------------------- default IO
    def _default_recorder(self, seconds: float, sr: int = 16000) -> np.ndarray:
        import sounddevice as sd

        x = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        return x[:, 0]

    def _speak(self, text: str, lang: str, context: dict[str, Any] | None = None) -> bool:
        mode = self.cfg.audio.tts
        if self.tts is not None:
            return bool(self.tts(text, lang))
        if mode in ("system", "system_say"):
            # macOS say / Windows SAPI through one abstraction; text and voice never reach a command string
            from homeease.ml.audio import tts as tts_mod

            return tts_mod.play(text, lang)
        if mode == "browser":
            # the elder page speaks it (speech synthesis) and shows the countdown: it needs the event, the window and every language
            ctx = dict(context or {})
            self.rt.broadcaster.publish("voice_prompt", {"text": text, "lang": lang, **ctx}, self.rt.subject_id, audience="elder")
            time.sleep(float(ctx.get("listen_after_s", PLAYBACK_S)))  # approximate playback time on the elder device
            return True
        return False

    def _prompt_texts(self, lang: str | None) -> list[tuple[str, str]]:
        """(language, text) pairs in speaking order: ``audio.languages`` (every language that has a text), or just ``lang`` when given."""
        if lang:
            return [(lang, PROMPTS.get(lang, PROMPTS["en"]))]
        order = [l for l in self.cfg.audio.languages if l in PROMPTS]
        return [(l, PROMPTS[l]) for l in dict.fromkeys(order)] or [("en", PROMPTS["en"])]

    def _elder_pages(self) -> int | None:
        """How many older-adult pages are connected to the node (they show and speak the question). None: the runtime cannot say (a stand-in
        without a broadcaster), and the question is taken as delivered."""
        fn = getattr(getattr(self.rt, "broadcaster", None), "pages_connected", None)
        try:
            return fn("elder") if callable(fn) else None
        except Exception:
            return None

    def _play_prompt(self, event_id: str, prompt_id: str, texts: list[tuple[str, str]], window_s: float) -> tuple[bool, str | None]:
        """Ask the question. Returns (delivered, why not): a question nobody could hear or see is not asked, and silence after it says nothing."""
        if self.tts is None and self.cfg.audio.tts == "none":
            return False, "audio.tts is none: the question is never asked"
        if self.tts is None and self.cfg.audio.tts == "browser":
            if self._elder_pages() == 0:
                return False, "no older adult page is connected: the question reached nobody"
            # one broadcast with every language: the page chooses (or speaks them in order) and counts the window down
            wait = PLAYBACK_S + 2.0 * (len(texts) - 1)
            ctx = {"event_id": event_id, "prompt_id": prompt_id, "window_s": window_s, "listen_after_s": wait, "languages": [l for l, _ in texts], "texts": dict(texts)}
            ok = self._speak(texts[0][1], texts[0][0], ctx)
        else:
            ok = True
            for l, t in texts:
                ok = self._speak(t, l) and ok
        return ok, None if ok else "TTS playback failed"

    # ------------------------------------------------------------- one check per event
    def _state_of(self, event_id: str) -> str | None:
        with self._state_lock:
            return (self._asked or {}).get(event_id)

    def _store(self, event_id: str, state: str) -> None:
        """(caller holds _state_lock) newest last: the oldest events are forgotten first."""
        if self._asked is None:
            self._asked = {}
        self._asked.pop(event_id, None)
        self._asked[event_id] = state
        while len(self._asked) > REMEMBERED_EVENTS:
            self._asked.pop(next(iter(self._asked)))

    def _set_state(self, event_id: str, state: str) -> None:
        with self._state_lock:
            self._store(event_id, state)

    def _claim(self, event_id: str) -> tuple[bool, str | None]:
        """Take the right to ask about this event. (False, state) when a check for it is running or done; otherwise (True, what it was)."""
        with self._state_lock:
            seen = (self._asked or {}).get(event_id)
            if seen in ("running", "done"):
                return False, seen
            self._store(event_id, "running")
            return True, seen

    # ------------------------------------------------------------- inquiry
    def on_event(self, event_id: str, level: str) -> None:
        """A suspected fall (PENDING or EMERGENCY decision). Decided again or upgraded later, the same event is not asked twice."""
        if not (self._state["enabled"] and self._read_consent()):
            if self._state_of(event_id) in (None, "busy"):  # one "not run" record per event, not one per decision
                self._set_state(event_id, "not_run")
                self._attach(event_id, {"status": "not_run", "reason": "audio disabled or no consent"})
            return
        claimed, before = self._claim(event_id)
        if not claimed:
            return
        threading.Thread(target=self._run, args=(event_id, before), name="voice-inquiry", daemon=True).start()

    def _run(self, event_id: str, before: str | None = None) -> None:
        if not self._busy.acquire(blocking=False):
            # one microphone, one inquiry at a time: the later event is marked instead of silently having no voice record
            self._set_state(event_id, "busy")
            if before != "busy":
                self._attach(event_id, {"status": "not_run", "reason": "inquiry_busy"})
            return
        try:
            self.run_inquiry(event_id, require_consent=True)
        except Exception as e:
            # an event whose check broke must not look as if nobody asked: the record says it could not be made (never "safe", never a channel)
            log.exception("voice inquiry failed")
            self._attach(event_id, {"prompt_id": new_id("pr_"), "started_at": self.rt.clock.utc_now().isoformat(), "status": "audio_unavailable",
                                    "error": f"inquiry failed: {type(e).__name__}", **self._no_evidence(cg.AUDIO_UNAVAILABLE)})
        finally:
            self._set_state(event_id, "done")
            self._busy.release()

    def _intent_model(self) -> Any:
        """Reply-intent method from the config. rule_v2 is the default; "rule" keeps the original v1 rule."""
        if self.cfg.audio.intent_method == "rule":
            from homeease.ml.audio.intent import RuleIntent

            return RuleIntent()
        from homeease.ml.audio.intent_v2 import RuleIntentV2

        return RuleIntentV2()

    # ------------------------------------------------------------- reply source
    def _stream_ready(self) -> bool:
        """Is the live microphone stream running with its speech branch (the runtime knows)? Then it is the reply source."""
        fn = getattr(self.rt, "voice_stream_ready", None)
        try:
            return bool(fn()) if callable(fn) else False
        except Exception:
            return False

    def _stream_speaking(self) -> bool:
        fn = getattr(self.rt, "voice_stream_speaking", None)
        try:
            return bool(fn()) if callable(fn) else False
        except Exception:
            return False

    def listening_since(self) -> float | None:
        """Pipeline-clock time (ms) from which speech belongs to the open inquiry (the moment the question was asked), or None when no
        inquiry takes its reply from the stream."""
        w = self._window
        return None if w is None else w.opened_ms

    def offer_segment(self, seg: "SpeechSegment") -> bool:
        """A speech segment of the live stream. True when it is the reply of the inquiry that is listening (it began after the question
        was asked); False leaves it to the help listener. The samples are not kept by this method."""
        with self._state_lock:
            w = self._window
            if w is None or seg.t_start_mono_ms < w.opened_ms:
                return False
            w.add(seg)
            return True

    def _wait(self, seconds: float) -> None:
        (self._stop_evt or threading.Event()).wait(max(seconds, 0.0))

    def _stopping(self) -> bool:
        return self._stop_evt is not None and self._stop_evt.is_set()

    def _open_window(self, opened_ms: float | None = None) -> "_ReplyWindow":
        """Start taking replies from the stream: speech that begins after ``opened_ms`` (default: now) belongs to this question."""
        window = _ReplyWindow(float(self.rt.clock.mono_ms()) if opened_ms is None else opened_ms)
        with self._state_lock:
            self._window = window
        return window

    def _close_window(self, window: "_ReplyWindow | None") -> None:
        """Stop taking replies and wipe whatever was handed over and not used (idempotent)."""
        if window is None:
            return
        with self._state_lock:
            if self._window is window:
                self._window = None
        for seg in window.take():
            try:
                seg.samples[:] = 0.0
            except Exception:  # pragma: no cover - a read-only buffer
                pass

    def _listen_stream(self, window: "_ReplyWindow", window_s: float) -> list["SpeechSegment"]:
        """Wait out the response window, then collect the reply from the stream: segments that began after the question was asked, plus
        the time a reply that began inside the window needs to finish."""
        try:
            self._wait(window_s)
            end = time.monotonic() + REPLY_TAIL_S
            while self._stream_speaking() and time.monotonic() < end and not self._stopping():
                self._wait(0.1)
            if time.monotonic() < end:
                self._wait(0.2)  # the segment is cut a moment after the speech run closes
        finally:
            with self._state_lock:
                if self._window is window:
                    self._window = None  # nothing more is handed over
        return window.take()

    @staticmethod
    def _assemble(segments: list["SpeechSegment"]) -> np.ndarray:
        """One mono array from the reply's segments in time order (a copy: the segments are zeroed by the caller)."""
        gap = np.zeros(int(REPLY_GAP_S * 16000), dtype=np.float32)
        parts: list[np.ndarray] = []
        for seg in sorted(segments, key=lambda g: g.t_start_mono_ms):
            parts += [np.asarray(seg.samples, dtype=np.float32), gap]
        return np.concatenate(parts[:-1])[: int(MAX_REPLY_S * 16000)] if parts else np.zeros(0, dtype=np.float32)

    def _shared_asr(self) -> Any:
        """The node's one ASR engine (the runtime builds it once; the help listener uses the same), else a private one."""
        fn = getattr(self.rt, "asr_engine", None)
        if callable(fn):
            return fn()
        from homeease.ml.audio.asr import ASRProcess

        return ASRProcess(f"faster-whisper-{self.cfg.audio.asr_model}", temperature=self.cfg.audio.asr_temperature)

    def run_inquiry(self, event_id: str, lang: str | None = None, require_consent: bool = False) -> dict[str, Any]:
        """Prompt, listen, recognise, attach. With ``require_consent`` (the production path) the consent is checked again
        before the prompt, again just before listening (the prompt takes seconds to play) and once more when the reply is in."""
        prompt_id = new_id("pr_")
        texts = self._prompt_texts(lang)
        window_s = float(self.cfg.audio.inquiry_response_window_s)
        rec: dict[str, Any] = {"prompt_id": prompt_id, "lang": texts[0][0], "languages": [l for l, _ in texts], "window_s": window_s, "started_at": self.rt.clock.utc_now().isoformat()}
        asked_ms: float | None = None
        try:  # the audio marker of the alarm tag is stamped on the sensors' time base (mono -> UTC), not on the wall clock read above
            asked_ms = float(self.rt.clock.mono_ms())
            rec["marker_at"] = self.rt.clock.mono_to_utc(asked_ms).isoformat()
        except Exception:  # a clock stand-in without the mapping: the wall-clock start is used
            pass
        if require_consent and not self._read_consent():
            rec.update(status="not_run", reason="audio consent withdrawn")
            self._attach(event_id, rec)
            return rec
        # With a running stream a reply that comes while the question is still being asked is heard as well (the question's own words are
        # read as OTHER, so an echo of it is never a cancel); without one the recorder starts after the prompt, as it always did.
        window = self._open_window(asked_ms) if self._stream_ready() else None
        try:
            spoken, why = self._play_prompt(event_id, prompt_id, texts, window_s)
            rec["prompt_played"] = spoken
            if not spoken:
                # Nobody was asked, but the room is still listened to: a call for help is a call whether or not a question was put. What
                # EventService.voice_response does with the rest of such a record is "not run" (silence after no question says nothing).
                rec["prompt_error"] = why
            if require_consent and not self._read_consent():
                rec.update(status="not_run", reason="audio consent withdrawn before listening")
                self._attach(event_id, rec)
                return rec
            rec["listen_started_at"] = self.rt.clock.utc_now().isoformat()
            if window is None and self._stream_ready():  # the stream came up while the question was played
                window = self._open_window()
            if window is not None and (self._stream_ready() or window.has_segments()):
                return self._reply_from_stream(event_id, rec, window, window_s, require_consent)
        finally:
            self._close_window(window)
        rec["reply_source"] = "recorder"
        try:
            audio = self.recorder(window_s)
        except Exception as e:
            rec.update(status="audio_unavailable", error=str(e)[:200], **self._no_evidence(cg.AUDIO_UNAVAILABLE))
            self._attach(event_id, rec)
            return rec
        try:
            if require_consent and not self._read_consent():  # withdrawn while it listened: what was heard is not used
                rec.update(status="not_run", reason="audio consent withdrawn while listening")
                self._attach(event_id, rec)
                return rec
            return self._handle_reply(event_id, rec, audio)
        finally:
            try:
                audio[:] = 0.0  # discard the raw samples on EVERY path (silence, ASR failure, reply), not only after a reply
            except Exception:  # pragma: no cover - a recorder that hands out a read-only buffer
                pass

    def _reply_from_stream(self, event_id: str, rec: dict[str, Any], window: "_ReplyWindow", window_s: float, require_consent: bool) -> dict[str, Any]:
        rec["reply_source"] = "stream"
        segments = self._listen_stream(window, window_s)
        rec["segments"] = len(segments)
        audio = np.zeros(0, dtype=np.float32)
        try:
            if self._stopping():  # the node is shutting down: the window was cut short, so silence in it means nothing
                rec.update(status="not_run", reason="node stopped while listening")
                self._attach(event_id, rec)
                return rec
            if require_consent and not self._read_consent():
                rec.update(status="not_run", reason="audio consent withdrawn while listening")
                self._attach(event_id, rec)
                return rec
            if not segments and not self._stream_ready():
                # no speech AND the stream is gone: nothing is known about the room (a failure, not a silent person)
                rec.update(status="audio_unavailable", error="microphone stream lost during the response window", **self._no_evidence(cg.AUDIO_UNAVAILABLE))
                self._attach(event_id, rec)
                return rec
            audio = self._assemble(segments)
            return self._handle_reply(event_id, rec, audio, speech=bool(segments))
        finally:
            for seg in segments:
                try:
                    seg.samples[:] = 0.0
                except Exception:  # pragma: no cover - a read-only buffer
                    pass
            try:
                audio[:] = 0.0
            except Exception:  # pragma: no cover
                pass

    def _handle_reply(self, event_id: str, rec: dict[str, Any], audio: np.ndarray, speech: bool | None = None) -> dict[str, Any]:
        """Recognise and judge one reply. ``speech`` is the verdict of a voice activity detector that already ran (the stream's);
        None means this method checks the recording itself."""
        from homeease.ml.audio.vad import EnergyVAD, segments

        if speech is None:
            speech = bool(segments(EnergyVAD().frames(audio)))
        if not speech:
            rec.update(status="NO_RESPONSE", intent=None, **self._no_evidence(cg.NO_SPEECH))
            self._attach(event_id, rec)
            return rec
        try:
            if self.asr is None:
                self.asr = self._shared_asr()
            res = self.asr.transcribe(audio, None)
        except Exception as e:
            # A recogniser that cannot be built or that failed heard nothing: that is the audio layer failing, not a person who did not
            # answer. Unlike UNCERTAIN (it ran and was unsure) it neither escalates nor counts as a channel, and it stays visible.
            rec.update(status="audio_unavailable", error=f"asr failed: {str(e)[:150]}", **self._no_evidence(cg.ASR_FAILED))
            self._state["asr"] = {"state": "degraded", "error": type(e).__name__, "at": self.rt.clock.utc_now().isoformat()}
            self._note_error(f"asr failed: {type(e).__name__}")
            self._attach(event_id, rec)
            return rec
        self._state["asr"] = {"state": "ok", "error": None, "at": self.rt.clock.utc_now().isoformat()}
        # the answer to an open question is read for what it says (a bare "yes" is no longer the answer to a yes/no question)
        it = self._intent_model().predict(res["text"], res.get("avg_logprob"))
        rec.update(status="RESPONDED", intent=it["intent"], intent_reason=it["reason"], intent_method=self.cfg.audio.intent_method, transcript=(res["text"] or "")[:100], asr_model=res.get("model"),
                   asr_avg_logprob=res.get("avg_logprob"), source_note="speaker identity not verified (TV/visitor speech possible)")
        rec.update(self._cancel_evidence(audio, it, res.get("avg_logprob")))
        rec["voiceprint_enrolled"] = self.enrollment is not None  # a reply that could not be compared with an enrolled voice is not a trusted "I'm fine"
        if rec.get("speaker_verified") is not None:
            rec["source_note"] = "voiceprint compared with the enrollment; it only decides whether the reply stops the voice check's escalation, the event is never closed or downgraded by it (TV, replay and visitors can still pass or fail it)"
        self._attach(event_id, rec)
        return rec

    # ------------------------------------------------------------- cancel-eligibility evidence
    @staticmethod
    def _no_evidence(reason: str) -> dict[str, Any]:
        """Fields of a reply that never reached the voiceprint gate: explicitly NOT eligible, never missing."""
        return {"cancel_eligible": False, "cancel_reasons": [reason], "cancel_policy": cg.POLICY, "speaker_verified": None, "speaker_score": None}

    def _cancel_evidence(self, audio: np.ndarray, intent: dict[str, Any], asr_conf: float | None) -> dict[str, Any]:
        """Voiceprint match + reply intent -> cancel_eligible EVIDENCE. Without an enrollment it is False (no_enrollment) and the speaker model is not even loaded.
        Any failure is False with a reason code; nothing in the result is ever used to change an event."""
        try:
            if self.cancel_gate is None:
                self.cancel_gate = cg.CancelGate.from_config(self.cfg.audio)
            return self.cancel_gate.evaluate(audio, intent, self.enrollment, asr_conf).to_record()
        except Exception as e:  # pragma: no cover - the gate does not raise for model/audio problems; this is the last line of defence
            log.exception("cancel-eligibility evaluation failed")
            self._state["errors"].append(f"voiceprint gate: {str(e)[:150]}")
            return self._no_evidence(cg.GATE_ERROR)

    def _note_error(self, text: str) -> None:
        errors = self._state["errors"]
        errors.append(text[:200])
        del errors[:-20]  # the newest few: a failing recogniser must not grow this without bound

    def _attach(self, event_id: str, rec: dict[str, Any]) -> None:
        self._state["last"] = {"event_id": event_id, **{k: rec.get(k) for k in ("status", "intent", "prompt_id")}}
        try:
            self.rt.events.voice_response(event_id, rec)
        except Exception as e:
            log.exception("attaching voice response failed")
            self._note_error(str(e))
            if rec.get("status") == "RESPONDED" and rec.get("intent") == "HELP":
                self._call_for_help(rec)

    def _call_for_help(self, rec: dict[str, Any]) -> None:
        """The reply could not be attached to its event (database error, event gone): a call for help is still raised, as an alarm of its
        own, like the listener's. Labels only, never the words."""
        try:
            rt = self.rt
            rt.events.spoken_help(rt.subject_id, idem=f"vi_{rec.get('prompt_id')}", source_mode=getattr(rt, "source_mode", "live"), room_id=getattr(rt, "room_id", None),
                                  details={"intent": "HELP", "origin": "voice_check", "asr_avg_logprob": rec.get("asr_avg_logprob")})
        except Exception as e:
            log.exception("spoken call for help could not be raised")
            self._note_error(f"spoken help could not be raised: {type(e).__name__}")
