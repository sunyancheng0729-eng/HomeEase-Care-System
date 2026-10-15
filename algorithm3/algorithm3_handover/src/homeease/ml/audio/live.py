"""Live microphone front of the sound monitor: input stream + worker thread + health reporting.

Responsibilities (everything else lives in ``pipeline.py`` / ``monitor.py``):

* Listen only while consent is on and there is something to listen for. The sound branch needs a loaded
  classifier: without a model there is nothing to analyse, so the microphone stays closed
  (``degraded:model_missing``) unless the help listener needs the stream too (``voice_required``), which
  opens it on its own. A monitor built with ``sound=False`` is the voice branch alone: no classifier, no
  sound events. The voice branch (``voice_sink``) is also how a voice inquiry reads its reply from an open
  stream instead of opening a second one; it never opens the microphone by itself. Withdrawing consent
  closes the stream and wipes the audio held in memory.
* The sounddevice callback only copies the block into a bounded queue; analysis runs in this
  object's own thread. If analysis falls behind, the oldest block is dropped and counted (the
  pipeline then sees a timestamp gap and restarts its timeline instead of stitching across it).
* A missing or failing microphone, a missing or corrupt model package, a stream that delivers nothing, a
  stream that delivers only digital silence (muted or privacy-blocked microphone), a stalled classifier: all
  become a ``degraded`` status with a reason (and a health interval via ``on_state``), retried with back-off.
  Nothing here raises into the runtime. Degraded never means "safe".
* "ok" is claimed only with evidence: audio blocks arrived since the stream was opened, the newest one is
  recent, the signal is not digital silence, and no stream or analysis fault happened during the last
  ``OK_HOLD_S`` seconds. Degrading is immediate, recovering is slow, so a flapping device writes one
  health interval per episode, not one per flip.
* Raw audio is never written anywhere: the queue, the ring buffer and the windows live in memory
  and are dropped or zeroed (``stop()``, consent withdrawal, stream loss: ``_drop_stream`` wipes all three).
* Without a consent provider there is no consent (fail closed).

Default OFF (``audio.sound_monitor.enabled=false``); started by ``Runtime.start`` in live mode only.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from homeease.core.clock import Clock
from homeease.core.config import AppConfig
from homeease.core.util import get_logger
from homeease.ml.audio.monitor import SoundClassifier, SoundEventStore, SoundMonitor
from homeease.ml.audio.pipeline import Function3Pipeline, SpeechSegment
from homeease.ml.registry import ModelLoadError

if TYPE_CHECKING:
    from homeease.backend.storage.db import Database

log = get_logger(__name__)

SR = 16000
BLOCK = 1600  # 100 ms per callback
QUEUE_BLOCKS = 100  # 10 s of audio between the callback and the worker
CONSENT_EVERY_S = 10.0
MODEL_RETRY_S = 60.0
HEALTH_EVERY_S = 5.0
SILENCE_WATCHDOG_S = 3.0  # no block for this long: the stream is dead even if the driver did not say so
OK_HOLD_S = 30.0  # after a stream / analysis fault, "ok" needs this long without a new one (no flapping)
SILENT_PEAK = 1e-4  # a block whose peak stays below this (-80 dBFS) carries no microphone signal (a real room is louder)
SILENT_MIC_S = 60.0  # this long without a non-silent block: muted / blocked microphone, degraded:microphone_silent
ANALYSIS_STALL_S = 30.0  # one window analysis taking longer than this is a hang, not a slow model
STATUS_WAIT_S = 0.25  # status() never waits longer than this for the worker (it returns the last snapshot, stale)
PRUNE_EVERY_S = 6 * 3600.0
POLL_S = 1.0  # idle wait of the worker loop
BACKOFF_START_S = 5.0  # microphone re-open back-off: doubles up to BACKOFF_MAX_S
BACKOFF_MAX_S = 300.0


def open_input_stream(callback: Callable[..., None], sr: int = SR, blocksize: int = BLOCK) -> Any:
    """Open and start the default microphone (module-level so tests can replace it)."""
    import sounddevice as sd

    stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=blocksize, callback=callback)
    stream.start()
    return stream


def _default_loader(model_ref: str, sha256: str | None = None) -> SoundClassifier:
    from homeease.ml.audio.sound_package import load_sound_classifier

    return load_sound_classifier(model_ref, expected_sha256=sha256)


class LiveSoundMonitor:
    def __init__(
        self,
        cfg: AppConfig,
        db: "Database",
        clock: Clock,
        subject_id: str,
        room_id: str | None = None,
        consent: Callable[[], bool] | None = None,  # None = no consent provider = never listen (fail closed)
        on_state: Callable[[str, list[str]], None] | None = None,
        loader: Callable[[str], SoundClassifier] | None = None,
        stream_factory: Callable[..., Any] | None = None,
        source_mode: str = "live",
        source: str = "microphone",
        voice_sink: Callable[[SpeechSegment], None] | None = None,
        sound: bool = True,
        voice_required: bool = False,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.clock = clock
        self.subject_id = subject_id
        self.room_id = room_id
        self._consent = consent
        self._on_state = on_state
        self._loader = loader or (lambda ref: _default_loader(ref, cfg.audio.sound_monitor.model_sha256))
        self._open_stream = stream_factory or open_input_stream
        self.source = source  # "microphone", or "replay_clips" in demonstrations (ml/audio/replay_clips.py)
        self._sound = bool(sound)  # the sound branch (classifier, sound events); off = the stream only feeds the voice branch
        self._voice_sink = voice_sink  # receives the speech segments of the stream (transient samples: never stored by the sink)
        self._voice_required = bool(voice_required)  # the help listener needs this stream: it opens without a sound model, and a failing speech detector degrades it
        self.store = SoundEventStore(db, subject_id, room_id, source_mode, clock)
        self._q: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=QUEUE_BLOCKS)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream: Any = None
        self._state, self._reasons = "off", ["starting"]
        self._consent_ok: bool | None = None
        self._last_audio = time.monotonic()
        self._last_signal = self._last_audio  # newest block that was not digital silence
        self._blocks_since_open = 0
        self._blocks_total = 0
        self._last_bad = float("-inf")  # newest stream / analysis fault (monotonic); "ok" needs OK_HOLD_S after it
        self._busy_since: float | None = None  # monotonic start of the window analysis in progress
        self._next_open = 0.0
        self._backoff = BACKOFF_START_S
        self._overruns = 0
        self._cb_flags = 0
        self._stream_opens = 0
        self._internal_errors = 0
        self._block_errors = 0
        self.classifier: SoundClassifier | None = None
        self.monitor: SoundMonitor | None
        self.pipeline: Function3Pipeline
        self._build(None)

    # ---------------------------------------------------------------- build
    def _build(self, clf: SoundClassifier | None) -> None:
        # build first, assign after: a classifier the monitor rejects must leave the model-less state (microphone closed) intact
        monitor = SoundMonitor(self.cfg.audio.sound_monitor, clf, self.clock, self.store) if self._sound else None
        pipeline = Function3Pipeline(self.cfg.audio, self.clock, monitor, voice_sink=self._voice_sink)
        old = getattr(self, "pipeline", None)
        self.classifier, self.monitor, self.pipeline = clf, monitor, pipeline
        if old is not None:
            old.close()

    def _set(self, state: str, reasons: list[str]) -> None:
        if (state, reasons) == (self._state, self._reasons):
            return
        self._state, self._reasons = state, list(reasons)
        (log.warning if state == "degraded" else log.info)("sound monitor %s%s", state, f" ({', '.join(reasons)})" if reasons else "")
        if self._on_state is not None:
            try:
                self._on_state(state, list(reasons))
            except Exception:  # health reporting must not stop the monitor
                log.exception("sound monitor state callback failed")

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sound-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._drop_stream()
        self._set("off", ["stopped"])

    # --------------------------------------------------------------- stream
    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Audio thread: copy and enqueue only. Timestamp = capture time of the block's first sample."""
        try:
            if status:
                self._cb_flags += 1
            x = np.array(indata[:, 0] if getattr(indata, "ndim", 1) == 2 else indata, dtype=np.float32)
            item = (x, self.clock.mono_ms() - len(x) * 1000.0 / SR)
            try:
                self._q.put_nowait(item)
            except queue.Full:
                try:
                    self._q.get_nowait()  # keep the newest audio, count what was lost
                except queue.Empty:
                    pass
                self._overruns += 1
                self._q.put_nowait(item)
        except Exception:  # an exception in a PortAudio callback would kill the stream
            self._block_errors += 1

    def _open(self) -> bool:
        try:
            self._stream = self._open_stream(self._callback)
        except Exception as e:  # no device, device busy, permission denied, driver error ...
            self._stream = None
            log.warning("microphone could not be opened: %s", type(e).__name__)
            self._set("degraded", ["microphone_unavailable"])
            return False
        self._stream_opens += 1
        self._last_audio = self._last_signal = time.monotonic()
        self._blocks_since_open = 0  # nothing is known about the new stream until its first block arrives
        return True

    def _drop_stream(self) -> None:
        """Close the stream and drop every trace of its audio: queued blocks and the pipeline's ring buffer.
        Idempotent; also the way to wipe memory when consent is withdrawn after the stream was already lost."""
        st, self._stream = self._stream, None
        if st is not None:
            try:
                st.stop()
                st.close()
            except Exception:
                pass
        while True:  # whatever is still queued is audio: drop it
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        try:
            self.pipeline.close(timeout=5.0)  # flushes open events, zeroes the ring buffer
        except Exception:
            log.exception("wiping the audio buffers failed")

    # --------------------------------------------------------------- worker
    def _check_consent(self) -> bool:
        if self._consent is None:
            return False  # no provider, no consent
        try:
            return bool(self._consent())
        except Exception:
            log.exception("consent check failed")
            return False  # unknown consent is no consent

    def _load_model(self) -> bool:
        ref = self.cfg.audio.sound_monitor.model_id
        try:
            clf = self._loader(ref)
        except ModelLoadError as e:
            self._set("degraded", ["model_missing"])
            log.debug("sound model %s not loaded: %s", ref, e)
            return False
        except Exception as e:
            log.warning("sound model %s failed to load: %s", ref, type(e).__name__)
            self._set("degraded", ["model_error"])
            return False
        try:
            self._build(clf)
        except Exception as e:  # e.g. the monitor rejects the model's class list: stay model-less, microphone closed
            log.warning("sound model %s rejected: %s", ref, type(e).__name__)
            self._set("degraded", ["model_error"])
            return False
        return True

    def _run(self) -> None:
        mono = time.monotonic
        next_consent = next_model = next_health = 0.0
        next_prune = 0.0
        consent_ok = False
        while not self._stop.is_set():
            now = mono()
            try:
                if now >= next_consent:
                    consent_ok = self._consent_ok = self._check_consent()
                    next_consent = now + CONSENT_EVERY_S
                if not consent_ok:
                    self._drop_stream()  # also wipes the ring buffer, whether or not a stream was still open
                    self._set("off", ["no_consent"])
                    self._stop.wait(POLL_S)
                    continue
                if self.classifier is None and self._sound and now >= next_model and not self._load_model():
                    next_model = now + MODEL_RETRY_S
                if self.classifier is None and not self._voice_required:  # no model and no listener: nothing to analyse, the microphone stays closed
                    self._stop.wait(POLL_S)
                    continue
                if now >= next_prune:
                    self._prune()
                    next_prune = now + PRUNE_EVERY_S
                if self._stream is None:
                    if now < self._next_open:
                        self._stop.wait(POLL_S)
                        continue
                    if not self._open():
                        self._next_open = now + self._backoff
                        self._backoff = min(self._backoff * 2.0, BACKOFF_MAX_S)
                        continue
                if now >= next_health:
                    next_health = now + HEALTH_EVERY_S
                    self._health_tick(now)
                self._pump()
            except Exception:
                self._internal_errors += 1
                self._last_bad = mono()
                if self._internal_errors <= 3 or self._internal_errors % 100 == 0:
                    log.exception("sound monitor loop error (%d so far)", self._internal_errors)
                self._set("degraded", ["internal_error"])
                self._stop.wait(2.0 * POLL_S)

    def _health_reasons(self, now: float) -> list[str] | None:
        """Why the monitor is not healthy right now; [] = healthy, None = not known yet (stream just opened)."""
        snap = self.pipeline.status()  # also refreshes the snapshot a timed-out status() call falls back to
        st = snap["sound"]["status"]
        if st.startswith("degraded:"):
            return [st.split(":", 1)[1]]
        vs = snap["voice"]["status"]
        if self._voice_required and vs.startswith("degraded:"):
            return [vs.split(":", 1)[1]]  # vad_unavailable / vad_error: a listener without a speech detector hears nothing
        if self._blocks_since_open == 0:
            return None
        if now - self._last_audio > SILENCE_WATCHDOG_S:
            return ["no_audio_data"]
        if now - self._last_signal > SILENT_MIC_S:
            return ["microphone_silent"]
        return []

    def _health_tick(self, now: float) -> None:
        """Degrade at once, recover slowly. "ok" needs audio evidence and OK_HOLD_S without a fault (a dead
        stream that is re-opened and dies again must stay degraded instead of flipping to ok in between)."""
        reasons = self._health_reasons(now)
        if reasons is None:
            return
        if reasons:
            self._last_bad = now
            self._set("degraded", reasons)
        elif self._state != "ok" and now - self._last_bad >= OK_HOLD_S:
            self._set("ok", [])

    def _pump(self) -> None:
        try:
            x, t = self._q.get(timeout=0.25)
        except queue.Empty:
            now = time.monotonic()
            if now - self._last_audio > SILENCE_WATCHDOG_S:
                self._last_bad = now
                self._drop_stream()  # wipes the audio buffers; reopened with back-off by the loop
                self._next_open = now + self._backoff
                self._backoff = min(self._backoff * 2.0, BACKOFF_MAX_S)
                self._set("degraded", ["no_audio_data"])
            return
        now = time.monotonic()
        if self._blocks_since_open == 0:
            self._backoff = BACKOFF_START_S  # the new stream delivers: the next failure starts the back-off again
        self._blocks_since_open += 1
        self._blocks_total += 1
        self._last_audio = now
        if len(x) and float(np.max(np.abs(x))) >= SILENT_PEAK:
            self._last_signal = now
        self._busy_since = now
        try:
            self.pipeline.process_chunk(x, t)
        finally:
            self._busy_since = None
        if time.monotonic() - now > ANALYSIS_STALL_S:  # it came back, but a status reader saw the stall while it lasted
            self._last_bad = time.monotonic()
            self._set("degraded", ["analysis_stalled"])

    def _prune(self) -> None:
        try:
            n = self.store.prune(self.cfg.privacy.retention_days_events)
            if n:
                log.info("sound events: %d older than %d days deleted", n, self.cfg.privacy.retention_days_events)
        except Exception as e:
            log.warning("sound event retention failed: %s", type(e).__name__)

    # ---------------------------------------------------------- voice branch
    def voice_ready(self) -> bool:
        """The stream is open, delivers audio now, and its speech branch works. A voice inquiry then takes its reply from the speech
        segments instead of opening a second input stream on the same device."""
        if self._voice_sink is None or self._stream is None or self._consent_ok is not True:
            return False
        if self._blocks_since_open == 0 or time.monotonic() - self._last_audio > SILENCE_WATCHDOG_S:
            return False
        return self.pipeline.voice_status()["status"] == "ok"

    def voice_speaking(self) -> bool:
        """Speech is going on in the stream right now (its end silence has not completed)."""
        return self._stream is not None and self.pipeline.speech_in_progress()

    # --------------------------------------------------------------- status
    def status(self) -> dict[str, Any]:
        """Never waits for the worker: while a window is being classified or written the last snapshot is returned
        (``stale: true``), and an analysis that has been running for ``ANALYSIS_STALL_S`` is reported as degraded."""
        now = time.monotonic()
        state, reasons = self._state, list(self._reasons)
        busy = self._busy_since
        stalled = busy is not None and now - busy > ANALYSIS_STALL_S
        if stalled:
            state, reasons = "degraded", ["analysis_stalled"]
        pipe = self.pipeline.status(STATUS_WAIT_S)
        label = state if state == "ok" else f"{state}:{reasons[0]}" if reasons else state
        open_ = self._stream is not None
        out = {
            "enabled": True,
            "branches": {"sound": self._sound, "voice": self._voice_sink is not None},
            "status": label,
            "state": state,
            "reasons": reasons,
            "source": self.source,
            "consent": self._consent_ok,
            "microphone": {
                "open": open_, "opens": self._stream_opens, "overruns": self._overruns, "callback_flags": self._cb_flags, "block_errors": self._block_errors,
                "blocks": self._blocks_total,
                "seconds_since_block": round(now - self._last_audio, 1) if open_ else None,
                "seconds_since_signal": round(now - self._last_signal, 1) if open_ and self._blocks_since_open else None,
            },
            "analysis": {"busy_s": None if busy is None else round(now - busy, 1), "stalled": stalled},
            "pipeline": pipe,
            "internal_errors": self._internal_errors,
            "retention_days": self.cfg.privacy.retention_days_events,
            "raw_audio_stored": False,
        }
        playing = getattr(self._stream, "now_playing", None)
        if self.source != "microphone":
            out["now_playing"] = playing
        return out


__all__ = ["LiveSoundMonitor", "open_input_stream"]
