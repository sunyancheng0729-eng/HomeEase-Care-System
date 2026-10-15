"""Continuous listening for a spontaneous spoken call for help (Chinese or English).

The speech branch of the live microphone stream (``Function3Pipeline.voice_sink``) hands over speech segments. Each one is recognised
by the same ASR process the voice inquiry uses, read by the reply-intent rule, and only a short utterance that is a clear call
("\u6551\u547d", "help me") raises an alarm: ``on_help`` ends in ``EventService.spoken_help``, which treats it like the help button
(EMERGENCY, caregivers notified, no dual-channel gate).

* Everything is gated before it can alarm: consent and the enabled flags are read again for every segment, the utterance must be
  short (a conversation or a broadcast is not a call), the recogniser must be sure of it, and the intent must be HELP through a cue
  that is a call (not a remark such as "ouch"). Refusals ("\u6211\u4e0d\u9700\u8981\u5e2e\u52a9"), thanks, offers ("can I help you") and an ambulance that is only
  mentioned are not calls. After an alarm the cooldown keeps one long cry or a television from raising a second event.
* A recogniser call takes seconds: consent withdrawn or the listener stopped meanwhile drops what was heard, nothing is raised after that.
* A segment that belongs to a running voice inquiry is not judged here: the inquiry's own reply path escalates (no second event).
* Samples are a transient hand-off: queued in memory (bounded, the oldest is dropped and counted), zeroed after the judgement, never
  written, never logged. The words are not kept either: only labels (intent, cue class, ASR confidence, duration) leave this module.
* Failures are never "quiet": an ASR that cannot be built or fails, a full queue and a failing alarm callback show in ``status()``
  as degraded reasons.

Default OFF (``audio.help_listener.enabled``); it only runs inside an enabled audio layer with the subject's audio consent.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from homeease.core.clock import Clock, iso
from homeease.core.config import AppConfig
from homeease.core.util import get_logger, new_id
from homeease.ml.audio.intent_v2 import normalise_v2

if TYPE_CHECKING:
    from homeease.ml.audio.pipeline import SpeechSegment

log = get_logger(__name__)

QUEUE_SEGMENTS = 4  # speech waiting for the recogniser; a slow ASR must not pile up audio in memory
IDLE_WAIT_S = 0.5  # the worker's wait for work (also how quickly stop() is noticed)
STOP_JOIN_S = 5.0  # stop() waits this long for a worker that is inside a recogniser call
OVERFLOW_MEMORY_S = 60.0  # a dropped segment keeps the status degraded for this long
MAX_UNITS = 14  # words (English) and characters (Chinese) of a call: "I fell and I can't get up, please help me" is 10

# Cues of the reply-intent lexicon (ml/audio/intent_v2.py) that are a call, not a remark. The lexicon was built to read the answer to
# a question that has just been asked; a listener hears every conversation and programme, where "oh no", "ouch" and the Chinese
# equivalents of "oh dear" and "not good" are everyday exclamations. Descriptions of a state (stuck, on the floor, not well) stay
# with the sensors and the inquiry.
CALL_CUES = frozenset({"help", "need_help", "ambulance", "get_doctor", "cant_do", "zh_help", "zh_cant", "zh_ambulance"})

# A cue that needs a request around it. "ambulance", "doctor" and the Chinese word for ambulance are as much words of a news item or a
# passing remark as of a call; they count only as the object of a verb of asking ("call an ambulance", or "call an ambulance quickly"
# in Chinese) or when they are the whole utterance.
WEAK_CUES = frozenset({"ambulance", "get_doctor", "zh_ambulance"})
_REQUEST = re.compile(
    r"(?:\u53eb|\u8bf7|\u627e|\u558a|\u6253|\u62e8|\u5feb|\u8d76\u7d27|\u8d76\u5feb|\u9a6c\u4e0a|\u7acb\u523b|\u9ebb\u70e6|\u9001\u6211|\u9001\u4ed6|\u9001\u5979|\u5e26\u6211|\u7ed9\u6211|\u8981|\u9700\u8981)\S{0,3}(?:\u6551\u62a4|\u6025\u6551|\u6551\u63f4|\u533b\u751f|\u5927\u592b|\u533b\u9662)"
    r"|(?:^|\s)(?:call|get|fetch|send|bring|phone|dial|need|want|find|summon|ring|take me to|rush me to)\s+(?:\w+\s+){0,3}?(?:an?\s+|the\s+|my\s+)?(?:ambulance|doctor|nurse|paramedics?|hospital|emergency)"
    r"|(?:\u6551\u62a4\u8f66?|\u6025\u6551|\u6551\u63f4|\u533b\u751f|\u5927\u592b)(?:\u5feb|\u8d76\u7d27|\u8d76\u5feb|\u9a6c\u4e0a|\u7acb\u523b)?\u6765"
    r"|^(?:an?\s+)?(?:ambulance|doctor|emergency|paramedics?|\u6551\u62a4\u8f66?|\u6025\u6551|\u6551\u63f4|\u533b\u751f|\u5927\u592b)$"
)

# Spans that mention help without calling for it: thanks, an offer or a question put to someone else, "help is on the way", a help desk,
# an ambulance in the news. They are blanked out before the call cues are looked for, so a real call in the same utterance still counts.
# Refusals ("don't call an ambulance", or "I do not need help" in Chinese) are read by the reply-intent lexicon itself (a reassurance, never a help cue). Only what is
# listed here is removed: everything else stays a call, because a call that is missed costs more than a short false alarm.
_NOT_A_CALL = re.compile("|".join([
    r"(?:\u8c22\u8c22|\u591a\u8c22|\u611f\u8c22|\u8c22\u4e86|\u8c22)(?:\u4f60\u4eec?|\u60a8|\u5927\u5bb6|\u4f60\u7684|\u60a8\u7684|\u5927\u5bb6\u7684|\u7684)?(?:\u5e2e\u52a9|\u5e2e\u5fd9|\u5e2e\u6211|\u5e2e\u4e86\u6211|\u5e2e|\u6551\u4e86\u6211|\u6551\u6211|\u76f8\u52a9|\u6551\u547d\u4e4b\u6069)",
    r"(?:\u4f60\u4eec?|\u60a8)(?:\u662f\u5426|\u6709\u6ca1\u6709|\u8fd8)?(?:\u9700\u8981|\u8981|\u60f3\u8981|\u60f3)(?:\u4ec0\u4e48|\u4efb\u4f55|\u70b9|\u4e00\u70b9)?(?:\u5e2e\u52a9|\u5e2e\u5fd9|\u5e2e|\u6551\u62a4\u8f66?|\u533b\u751f|\u5927\u592b)",
    r"(?:\u9700\u8981|\u8981|\u60f3\u8981)(?:\u4ec0\u4e48|\u4efb\u4f55)?(?:\u5e2e\u52a9|\u5e2e\u5fd9|\u5e2e)[\u5417\u561b\u4e48\u5462]|\u8981\u4e0d\u8981(?:\u6211)?(?:\u5e2e\u52a9|\u5e2e\u5fd9|\u5e2e)",
    r"\u6211(?:\u6765|\u53bb|\u53ef\u4ee5|\u80fd|\u4f1a|\u5c31|\u5148)?(?:\u5e2e|\u66ff)(?:\u4f60|\u60a8|\u4ed6|\u5979|\u5927\u5bb6)|\u8ba9\u6211(?:\u6765)?(?:\u5e2e|\u66ff)",
    r"\u6211(?:\u6765|\u53bb|\u53ef\u4ee5|\u80fd|\u4f1a|\u5c31)(?:\u53eb|\u8bf7|\u627e|\u6253|\u62e8|\u558a)(?:\u4e2a|\u4e00\u4e2a)?(?:\u6551\u62a4\u8f66?|\u533b\u751f|\u5927\u592b|\u6551\u63f4)",
    r"\u6551\u62a4\u8f66(?:\u6765\u4e86|\u5230\u4e86|\u5df2\u5230|\u5230\u8fbe|\u8d76\u5230|\u9a76\u6765|\u9a76\u8fc7|\u5f00\u6765|\u5f00\u8fc7|\u7ecf\u8fc7|\u7684\u58f0\u97f3|\u9e23\u7b1b)",
    r"(?<![a-z'])(?:thanks?|thank you|thank u|thankyou|cheers|appreciate)(?: \w+){0,2}? (?:for )?(?:(?:your|the|all|his|her|their|that|any) )*(?:help|helping|assistance)(?![a-z'])",
    r"(?<![a-z'])(?:(?:can|could|may|shall|should|will|would|let|let's|lemme) (?:me|i|us|we)(?: \w+)? help|i(?: can| could| will|'ll| shall| would) help)(?![a-z'])",
    r"(?<![a-z'])(?:do|did|does|would|will|can) (?:you|he|she|they) (?:need|want|require) (?:any |some |a )?(?:help|assistance)(?![a-z'])|(?<![a-z'])need any help(?![a-z'])",
    r"(?<![a-z'])help (?:your ?self|him ?self|her ?self|them ?selves|our ?selves|my ?self)(?![a-z'])",
    r"(?<![a-z'])help(?:'s| is| will be| has) (?:on (?:the|its) way|coming|here|arriving|there|nearby)(?![a-z'])",
    r"(?<![a-z'])help ?(?:desk|line|centre|center|page|file|menu|section|button|wanted)(?![a-z'])",
    r"(?<![a-z'])(?:the |an? )?ambulance (?:is |was |has |had |will be )?(?:coming|here|arrived|arriving|on its way|on the way|came|passed|passing)(?![a-z'])",
]))

_UNIT = re.compile(r"[\u4e00-\u9fff]|[a-z0-9']+")


def _units(text: str) -> int:
    return len(_UNIT.findall(normalise_v2(text)))


class HelpListener:
    """Judges speech segments for a spoken call for help. ``submit`` is cheap and runs on the stream's thread; the ASR runs in the
    listener's own worker thread (``start``/``stop``). ``process`` judges one segment synchronously and returns ``"help"`` or the reason
    it was ignored (the same reason codes as ``status()["skipped"]``)."""

    def __init__(
        self,
        cfg: AppConfig,
        clock: Clock,
        on_help: Callable[[dict[str, Any]], Any],
        consent: Callable[[], bool] | None = None,
        asr: Any = None,
        asr_factory: Callable[[], Any] | None = None,
        intent: Any = None,
        inquiry_since: Callable[[], float | None] | None = None,
        queue_segments: int = QUEUE_SEGMENTS,
        warm_up: bool = False,
        on_state: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.lcfg = cfg.audio.help_listener
        self.clock = clock
        self._on_help = on_help
        self._consent = consent  # None = no consent provider = never judge (fail closed)
        self._asr = asr
        self._asr_factory = asr_factory
        self._intent = intent
        self._inquiry_since = inquiry_since
        self._max_queue = max(int(queue_segments), 1)
        self._warm_up = bool(warm_up)  # load the recogniser at start, not at the first call for help (a model load takes seconds)
        self._q: deque[SpeechSegment] = deque()
        self._cond = threading.Condition()
        self._lock = threading.Lock()  # counters: written by the stream's thread (submit) and the worker
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_trigger_ms: float | None = None
        self._last_trigger_utc: str | None = None
        self._last_drop_ms: float | None = None
        self._asr_state: str | None = None  # None | "unavailable" (cannot be built) | "error" (a call failed): cleared by the next success
        self._call_failed = False  # the last alarm callback raised: cleared by the next one that works
        self._last_error: str | None = None
        self._consent_ok: bool | None = None
        self._counts: dict[str, int] = {"segments": 0, "judged": 0, "triggers": 0, "dropped": 0, "asr_errors": 0, "callback_errors": 0, "internal_errors": 0}
        self._skipped: dict[str, int] = {}
        self._on_state = on_state  # told (the persistent trouble reasons) whenever they change: the node's audio health follows the listener
        self._told: tuple[str, ...] = ()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # running, or still inside a recogniser call after stop(): never a second worker
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="help-listener", daemon=True)
        self._thread.start()
        self._tell()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=STOP_JOIN_S)
            if not self._thread.is_alive():
                self._thread = None  # a worker that is still in a recogniser call stays known: it ends by itself, and raises nothing (see _judge)
        self._clear()

    def _clear(self) -> None:
        with self._cond:
            pending, self._q = list(self._q), deque()
        for seg in pending:
            self._wipe(seg)

    # ---------------------------------------------------------------- input
    def submit(self, seg: "SpeechSegment") -> bool:
        """Queue one speech segment (stream thread: a length check and an enqueue, never ASR). False: not queued."""
        with self._lock:
            self._counts["segments"] += 1
        why = self._length_gate(seg.duration_s)
        if why is not None:
            self._skip(why)
            self._wipe(seg)
            return False
        dropped = None
        with self._cond:
            if len(self._q) >= self._max_queue:
                dropped = self._q.popleft()  # the newest speech matters most: keep it, count what was lost
            self._q.append(seg)
            self._cond.notify()
        if dropped is not None:
            self._wipe(dropped)
            with self._lock:
                self._counts["dropped"] += 1
                self._last_drop_ms = self.clock.mono_ms()
        return True

    def _length_gate(self, duration_s: float) -> str | None:
        longest = min(self.lcfg.max_utterance_s, self.cfg.audio.gate.speech_max_s)
        if not duration_s >= self.lcfg.min_utterance_s:
            return "too_short"
        if duration_s > longest:
            return "too_long"
        return None

    # --------------------------------------------------------------- worker
    def _warm(self) -> None:
        """One second of silence through the recogniser: the model is loaded before the first call, and a recogniser that cannot work
        shows as degraded from the start instead of at the moment someone cries for help."""
        try:
            engine = self._engine()
        except Exception as e:
            self._asr_failed("unavailable", e)
            return
        try:
            engine.transcribe(np.zeros(16000, dtype=np.float32), None)
        except Exception as e:
            self._asr_failed("error", e)

    def _run(self) -> None:
        if self._warm_up:
            self._warm()
        while not self._stop.is_set():
            with self._cond:
                if not self._q:
                    self._cond.wait(IDLE_WAIT_S)
                seg = self._q.popleft() if self._q and not self._stop.is_set() else None
            if seg is None:
                continue
            try:
                self.process(seg)
            except Exception:  # one bad segment must not end the listener
                with self._lock:
                    self._counts["internal_errors"] += 1
                    n = self._counts["internal_errors"]
                self._last_error = "internal_error"
                if n <= 3 or n % 100 == 0:
                    log.exception("help listener failed on a segment (%d so far)", n)

    # -------------------------------------------------------------- judging
    def process(self, seg: "SpeechSegment") -> str:
        try:
            return self._judge(seg)
        finally:
            self._wipe(seg)

    def _judge(self, seg: "SpeechSegment") -> str:
        lc = self.lcfg
        with self._lock:
            self._counts["judged"] += 1
        if not (self.cfg.audio.enabled and lc.enabled):
            return self._skip("disabled")
        if not self._check_consent():
            return self._skip("no_consent")
        since = self._inquiry_window()
        if since is not None and seg.t_start_mono_ms >= since:
            return self._skip("inquiry_listening")  # its reply path escalates; a second event would only duplicate it
        why = self._length_gate(seg.duration_s)
        if why is not None:
            return self._skip(why)
        now = self.clock.mono_ms()
        if self._last_trigger_ms is not None and now - self._last_trigger_ms < lc.cooldown_s * 1000.0:
            return self._skip("cooldown")  # judged before the ASR: a repeated call costs no recognition
        try:
            engine = self._engine()
        except Exception as e:
            return self._asr_failed("unavailable", e)
        try:
            res = engine.transcribe(seg.samples, None)
        except Exception as e:
            return self._asr_failed("error", e)
        if self._asr_state is not None:
            self._asr_state = None
            self._tell()
        # the recogniser takes seconds: the node may have been stopped, or the consent withdrawn, meanwhile. What was heard is dropped.
        if self._stop.is_set():
            return self._skip("stopped")
        if not self._check_consent():
            return self._skip("no_consent")
        text = str((res or {}).get("text") or "")
        try:
            conf = float((res or {}).get("avg_logprob"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            conf = float("nan")  # no confidence reported: not sure
        if not text.strip():
            return self._skip("no_text")
        if not conf >= lc.min_asr_logprob:  # also true for NaN
            return self._skip("low_confidence")
        verdict = self._read(text, conf)
        if verdict["intent"] != "HELP":
            return self._skip("not_help")
        seen = normalise_v2(text)
        screened = _NOT_A_CALL.sub(" zz ", seen)  # thanks, offers, "help is coming" ...: not a call; the rest of the utterance still is judged
        if screened != seen:
            verdict = self._read(screened, conf)
        calls = sorted(c for c in verdict["cues"] if c in CALL_CUES)
        if verdict["intent"] != "HELP" or not calls:
            return self._skip("not_a_call")
        if not (set(calls) - WEAK_CUES) and not _REQUEST.search(screened):
            return self._skip("not_a_call")  # only an ambulance / doctor was mentioned, nobody was asked for one
        if _units(text) > MAX_UNITS:
            return self._skip("too_many_words")
        details = {"intent": "HELP", "cue_class": calls[0], "cues": calls, "asr_avg_logprob": round(conf, 3), "duration_s": round(float(seg.duration_s), 2),
                   "intent_method": self.cfg.audio.intent_method}
        call = {"idem": new_id("sh_"), "at": self._utterance_time(seg), "details": details}
        try:
            self._on_help(call)
        except Exception as e:  # not delivered: no cooldown, the next call is judged again
            with self._lock:
                self._counts["callback_errors"] += 1
            self._call_failed = True
            self._last_error = type(e).__name__
            log.warning("spoken help could not be raised: %s", type(e).__name__)
            self._tell()
            return "callback_error"
        if self._call_failed:
            self._call_failed = False
            self._tell()
        with self._lock:
            self._counts["triggers"] += 1
        self._last_trigger_ms = now
        self._last_trigger_utc = iso(self.clock.utc_now())
        return "help"

    def _read(self, text: str, conf: float) -> dict[str, Any]:
        """The configured reply-intent model; the cue names come from the v2 lexicon either way (the v1 rule names none)."""
        model = self._model()
        res = dict(model.predict(text, conf))
        cues = res.get("cues")
        if cues is None and res.get("intent") == "HELP":
            from homeease.ml.audio.intent_v2 import RuleIntentV2

            cues = RuleIntentV2().cues(text)[0]
        return {"intent": res.get("intent"), "cues": list(cues or [])}

    def _model(self) -> Any:
        if self._intent is None:
            if self.cfg.audio.intent_method == "rule":
                from homeease.ml.audio.intent import RuleIntent

                self._intent = RuleIntent()
            else:
                from homeease.ml.audio.intent_v2 import RuleIntentV2

                self._intent = RuleIntentV2()
        return self._intent

    def _engine(self) -> Any:
        if self._asr is None:
            if self._asr_factory is None:
                raise RuntimeError("no speech recogniser configured")
            self._asr = self._asr_factory()
        return self._asr

    def _asr_failed(self, state: str, e: Exception) -> str:
        self._asr_state = state
        self._last_error = type(e).__name__
        with self._lock:
            self._counts["asr_errors"] += 1
            n = self._counts["asr_errors"]
        if n <= 3 or n % 100 == 0:
            log.warning("help listener ASR %s (%d so far): %s", state, n, type(e).__name__)
        self._tell()
        return f"asr_{state}"

    def _check_consent(self) -> bool:
        if self._consent is None:
            ok = False
        else:
            try:
                ok = bool(self._consent())
            except Exception:
                log.exception("consent check failed")
                ok = False  # unknown consent is no consent
        self._consent_ok = ok
        return ok

    def _inquiry_window(self) -> float | None:
        if self._inquiry_since is None:
            return None
        try:
            return self._inquiry_since()
        except Exception:
            return None

    @staticmethod
    def _utterance_time(seg: "SpeechSegment") -> datetime | None:
        try:
            t = datetime.fromisoformat(seg.t_start_utc)
        except (TypeError, ValueError):
            return None
        return t if t.tzinfo is not None else None

    def _skip(self, reason: str) -> str:
        with self._lock:
            self._skipped[reason] = self._skipped.get(reason, 0) + 1
        return reason

    @staticmethod
    def _wipe(seg: "SpeechSegment") -> None:
        try:
            seg.samples[:] = 0.0
        except Exception:  # a read-only buffer: the reference goes away with the segment
            pass

    # --------------------------------------------------------------- status
    def _trouble(self, transient: bool = True) -> list[str]:
        """Why the listener cannot be trusted right now. ``transient`` adds what passes by itself (a queue that overflowed lately)."""
        reasons: list[str] = []
        started = self._thread is not None
        if started and not self._thread.is_alive():  # type: ignore[union-attr]
            reasons.append("worker_stopped")
        if self._asr_state is not None:
            reasons.append(f"asr_{self._asr_state}")
        if transient and self._last_drop_ms is not None and self.clock.mono_ms() - self._last_drop_ms < OVERFLOW_MEMORY_S * 1000.0:
            reasons.append("queue_overflow")
        if self._call_failed:
            reasons.append("event_error")
        if not self.lcfg.min_utterance_s < min(self.lcfg.max_utterance_s, self.cfg.audio.gate.speech_max_s):
            reasons.append("length_gate_empty")  # no speech segment can be both long enough and short enough: this listener hears nothing
        return reasons

    def _tell(self) -> None:
        """Pass a change of the persistent trouble on to the node (its audio health), once per change."""
        if self._on_state is None:
            return
        reasons = tuple(self._trouble(transient=False))
        if reasons == self._told:
            return
        self._told = reasons
        try:
            self._on_state(list(reasons))
        except Exception:  # health reporting must not stop the listener
            log.exception("help listener state callback failed")

    def status(self) -> dict[str, Any]:
        """Counters and health. ``ok`` only means that the machinery is ready (worker alive, recogniser not failing, nothing lost
        lately); whether the microphone delivers is the live stream's status."""
        now = self.clock.mono_ms()
        started = self._thread is not None
        alive = started and self._thread.is_alive()  # type: ignore[union-attr]
        reasons = self._trouble()
        with self._lock:
            counts, skipped = dict(self._counts), dict(self._skipped)
        if self._stop.is_set():
            state, reasons = "off", ["stopped"]
        elif self._consent_ok is False:
            state, reasons = "off", ["no_consent"]  # nothing is listened to without consent, whatever else is wrong
        else:
            state = "degraded" if reasons else "ok"
        cooldown_left = 0.0
        if self._last_trigger_ms is not None:
            cooldown_left = max(0.0, self.lcfg.cooldown_s - (now - self._last_trigger_ms) / 1000.0)
        with self._cond:
            queued = len(self._q)
        return {
            "enabled": True,
            "status": state if state == "ok" else f"{state}:{reasons[0]}",
            "state": state,
            "reasons": reasons,
            "running": alive,
            "consent": self._consent_ok,
            "queued": queued,
            "counts": counts,
            "skipped": skipped,
            "cooldown_s": self.lcfg.cooldown_s,
            "cooldown_left_s": round(cooldown_left, 1),
            "last_trigger_at": self._last_trigger_utc,
            "last_error": self._last_error,
            "raw_audio_stored": False,
            "transcript_stored": False,
        }


__all__ = ["CALL_CUES", "HelpListener", "MAX_UNITS", "QUEUE_SEGMENTS", "WEAK_CUES"]
