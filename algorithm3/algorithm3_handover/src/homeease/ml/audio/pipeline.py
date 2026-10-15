"""Function 3 online pipeline: microphone audio -> shared front end -> two branches (proposal Figure 3-3).

    16 kHz mono chunks --> ring buffer (memory only) --> ActivityGate --+--> VOICE branch: speech segments for the
                                                                          |    inquiry / ASR / voiceprint steps (sink)
                                                                          +--> SOUND branch: SoundMonitor
                                                                               (window -> classifier -> label-only events)

Deliberate difference from the drawing. The figure puts one VAD in front of both branches. A speech
VAD would throw away exactly the sounds the sound branch exists for (a fall impact, breaking glass,
a thud), so here the speech VAD gates the VOICE branch only. The SOUND branch has its own non-speech
gate: an energy-onset trigger against a tracked noise floor, plus a slow periodic window that
re-samples sustained sounds the onset trigger has absorbed (a vacuum cleaner raises its own floor).
Speech is also a sound: a call for help or a shout is scored by the sound branch as well.

Contract
* Input: float32 mono chunks at 16 kHz with the monotonic timestamp (ms) of their first sample, on the
  same ``Clock`` the monitor uses (a ``VirtualClock`` in tests). Audio is analysed in memory and
  dropped; the ring buffer is zeroed on ``close()``. Nothing here writes a file or logs samples.
* Timeline: the pipeline counts samples and only compares the caller's timestamps with that count. A
  jump larger than ``audio.gate.gap_tolerance_ms`` is a discontinuity (dropped audio, device change):
  buffers and consecutive-window state restart, the event is not stitched across the hole. Smaller
  deviations are timestamp jitter and are slewed away slowly.
* Failures are never "safe": no model -> ``degraded:model_missing`` and no events; a VAD that cannot
  be created -> ``degraded:vad_unavailable`` and no speech segments; both surface in ``status()``.
* Nothing produced here may raise, close, downgrade or cancel a fall event; labels and speech
  segments are evidence for later steps only.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable

import numpy as np

from homeease.core.clock import Clock, iso
from homeease.core.config import ActivityGateCfg, AudioCfg
from homeease.core.util import get_logger
from homeease.ml.audio.features import SR, resample
from homeease.ml.audio.monitor import SoundEvent, SoundMonitor

log = get_logger(__name__)

FRAME = 160  # 10 ms at 16 kHz: the unit of the gate and of the VAD decisions
MIN_ACTIVE_FRAMES = 2  # a single 10 ms frame above the threshold is a click, not an activity span
MAX_PIECE_S = 0.5  # longer chunks are processed in pieces so the ring buffer never loses a pending window
SLEW = 0.02  # fraction of a sub-tolerance timestamp deviation corrected per chunk
SPEECH_PAD_S = 0.1  # context kept around a speech segment for the ASR / voiceprint steps
VAD_LOOKAHEAD_FRAMES = 30  # VAD decisions are final only 300 ms after the frame (hangover / min-speech filters need the future)


# ---------------------------------------------------------------- ring buffer
class AudioRing:
    """Circular in-memory buffer of the most recent samples, addressed by absolute sample index."""

    def __init__(self, seconds: float, sr: int = SR) -> None:
        self.cap = max(int(round(seconds * sr)), 1)
        self.buf = np.zeros(self.cap, dtype=np.float32)
        self.total = 0  # absolute index of the next sample to be written

    def append(self, x: np.ndarray) -> None:
        n = len(x)
        if n == 0:
            return
        if n >= self.cap:  # only the newest cap samples survive
            x = x[-self.cap:]
            start_abs = self.total + n - self.cap
        else:
            start_abs = self.total
        m = len(x)
        i = start_abs % self.cap
        first = min(m, self.cap - i)
        self.buf[i:i + first] = x[:first]
        if first < m:
            self.buf[:m - first] = x[first:]
        self.total += n

    def get(self, a: int, b: int) -> np.ndarray:
        """Copy of samples [a, b) clamped to what is still buffered (possibly shorter than requested)."""
        a = max(a, self.total - self.cap, 0)
        b = min(b, self.total)
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        i, n = a % self.cap, b - a
        if i + n <= self.cap:
            return self.buf[i:i + n].copy()
        return np.concatenate([self.buf[i:], self.buf[:i + n - self.cap]])

    @property
    def oldest(self) -> int:
        return max(self.total - self.cap, 0)

    def clear(self) -> None:
        self.buf[:] = 0.0
        self.total = 0


# ---------------------------------------------------------------- front end
class SoundTrigger:
    """Non-speech activity trigger of the SOUND branch: frame level against a tracked noise floor.

    The floor follows quiet quickly and loud only slowly (``floor_rise_db_per_s``), so a sound that
    stays on is absorbed after a while; the periodic window of ``SoundGate`` catches those."""

    def __init__(self, cfg: ActivityGateCfg) -> None:
        self.cfg = cfg
        self.rise_per_frame = cfg.floor_rise_db_per_s / 100.0
        self.hold_frames = int(round(cfg.hold_s * 100))
        self.reset()

    def reset(self) -> None:
        self.floor_db: float | None = None
        self.last_level_db: float | None = None
        self.spans: deque[tuple[int, int]] = deque()  # closed activity spans, frame units [start, end)
        self._open: list[int] | None = None
        self._quiet = 0
        self.active_frames = 0

    def push(self, levels_db: np.ndarray, first_frame: int) -> None:
        cfg = self.cfg
        for k, lv in enumerate(levels_db.tolist()):
            f = first_frame + k
            lv = max(lv, -90.0)
            self.last_level_db = lv
            if self.floor_db is None:
                self.floor_db = lv
            active = lv > max(self.floor_db + cfg.onset_db, cfg.min_level_dbfs)
            if active:
                self.active_frames += 1
                if self._open is None:
                    self._open = [f, f + 1]
                else:
                    self._open[1] = f + 1
                self._quiet = 0
            elif self._open is not None:
                self._quiet += 1
                if self._quiet > self.hold_frames:
                    self._close()
            # the decision above used the old floor: a sound cannot raise its own threshold instantly
            if lv < self.floor_db:
                self.floor_db += 0.05 * (lv - self.floor_db)
            else:
                self.floor_db += min(lv - self.floor_db, self.rise_per_frame)

    def _close(self) -> None:
        if self._open is not None and self._open[1] - self._open[0] >= MIN_ACTIVE_FRAMES:
            self.spans.append((self._open[0], self._open[1]))
        self._open = None

    def close_open(self) -> None:
        self._close()

    def span(self, a: int, b: int) -> tuple[int, int] | None:
        """Hull (in samples) of the activity inside the sample range [a, b), or None."""
        lo = hi = None
        open_span = [tuple(self._open)] if self._open is not None and self._open[1] - self._open[0] >= MIN_ACTIVE_FRAMES else []
        for s, e in chain(self.spans, open_span):
            s, e = s * FRAME, e * FRAME
            if e <= a or s >= b:
                continue
            s, e = max(s, a), min(e, b)
            lo = s if lo is None else min(lo, s)
            hi = e if hi is None else max(hi, e)
        return None if lo is None else (lo, hi)

    def prune(self, before_sample: int) -> None:
        while self.spans and self.spans[0][1] * FRAME <= before_sample:
            self.spans.popleft()


class SoundGate:
    """Decides which sliding windows the SOUND branch scores: onset trigger, else the periodic window."""

    def __init__(self, cfg: ActivityGateCfg, confirm_windows: int, window_samples: int, sr: int = SR) -> None:
        self.cfg = cfg
        self.sr = sr
        self.confirm = max(int(confirm_windows), 1)
        self.window_samples = window_samples
        self.trigger = SoundTrigger(cfg)
        self.periodic_samples = int(round(cfg.periodic_s * sr))
        self.reset()

    def reset(self) -> None:
        self.trigger.reset()
        self.due_at = self.periodic_samples  # first periodic check one period after the stream starts
        self.burst_left = 0
        self.n_onset = self.n_periodic = self.n_quiet_skips = self.n_idle = 0

    def decide(self, a: int, b: int, window: Callable[[], np.ndarray]) -> tuple[str, tuple[float, float]] | None:
        """Window [a, b) -> (trigger, active span in seconds from window start) or None (not scored).

        ``window`` fetches the samples lazily; only the periodic path needs them (level check)."""
        self.trigger.prune(a - self.window_samples)
        span = self.trigger.span(a, b)
        if span is not None:
            self.burst_left = 0
            self.due_at = b + self.periodic_samples
            self.n_onset += 1
            return "onset", ((span[0] - a) / self.sr, (span[1] - a) / self.sr)
        if self.burst_left == 0 and b >= self.due_at:
            self.burst_left = self.confirm  # enough consecutive windows to satisfy N-window confirmation
        if self.burst_left > 0:
            w = window()
            level = 10.0 * np.log10(float(np.mean(w.astype(np.float64) ** 2)) + 1e-12) if len(w) else -120.0
            if level >= self.cfg.min_level_dbfs:  # a quiet room is not sampled: nothing below the absolute floor triggers
                self.burst_left -= 1
                self.due_at = b + self.periodic_samples
                self.n_periodic += 1
                return "periodic", (0.0, self.window_samples / self.sr)
            self.burst_left = 0
            self.due_at = b + self.periodic_samples
            self.n_quiet_skips += 1
            return None
        self.n_idle += 1
        return None


@dataclass
class _SpeechRun:
    """Speech state machine over committed 10 ms VAD decisions."""

    min_frames: int
    end_silence_frames: int
    max_frames: int
    start: int | None = None
    last_speech: int = -1

    def push(self, f: int, v: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        if v:
            if self.start is None:
                self.start = f
            self.last_speech = f
        elif self.start is not None and f - self.last_speech >= self.end_silence_frames:
            out += self._close(self.start, self.last_speech + 1)
        if self.start is not None and f + 1 - self.start >= self.max_frames:  # long speech is cut into bounded segments
            out += self._close(self.start, f + 1)
        return out

    def flush(self) -> list[tuple[int, int]]:
        return self._close(self.start, self.last_speech + 1) if self.start is not None else []

    def _close(self, s: int, e: int) -> list[tuple[int, int]]:
        self.start = None
        return [(s, e)] if e - s >= self.min_frames else []  # shorter bursts are clicks, not speech


class SpeechGate:
    """Speech VAD of the VOICE branch: re-runs the VAD on the last ``vad_window_s`` of audio every
    ``vad_hop_s`` and turns settled frame decisions into speech segments (spans in frames)."""

    def __init__(self, vad: Any, cfg: ActivityGateCfg) -> None:
        self.vad = vad
        self.cfg = cfg
        self.win_frames = int(round(cfg.vad_window_s * 100))
        self.hop_frames = max(int(round(cfg.vad_hop_s * 100)), 1)
        self.errors = 0  # kept across timeline restarts: a broken VAD stays visible
        self.reset()

    def reset(self) -> None:
        self.committed = 0
        self.next_step = self.hop_frames
        self.run = _SpeechRun(max(int(round(self.cfg.speech_min_s * 100)), 1), max(int(round(self.cfg.speech_end_silence_s * 100)), 1), int(round(self.cfg.speech_max_s * 100)))
        self.speech_frames = 0

    def step(self, ring: AudioRing, frames_done: int, final: bool = False) -> list[tuple[int, int]]:
        if not final and frames_done < self.next_step:
            return []
        while self.next_step <= frames_done:
            self.next_step += self.hop_frames
        f1 = frames_done
        f0 = max(0, f1 - self.win_frames)
        commit_to = f1 if final else f1 - VAD_LOOKAHEAD_FRAMES
        start = max(self.committed, f0)
        if commit_to <= start:
            return []
        x = ring.get(f0 * FRAME, f1 * FRAME)
        try:
            d = np.asarray(self.vad.frames(x, SR)).reshape(-1)
        except Exception as e:  # a failing VAD must not take the sound branch down
            self.errors += 1
            if self.errors <= 3 or self.errors % 100 == 0:
                log.warning("speech VAD failed (%d so far): %s", self.errors, type(e).__name__)
            self.committed = max(self.committed, commit_to)
            return []
        out: list[tuple[int, int]] = []
        for f in range(start, commit_to):
            i = f - f0
            v = int(d[i]) if 0 <= i < len(d) else 0
            self.speech_frames += v
            out += self.run.push(f, v)
        self.committed = commit_to
        return out

    def flush(self) -> list[tuple[int, int]]:
        return self.run.flush()


class ActivityGate:
    """Shared front end: speech VAD for the VOICE branch, onset/periodic gate for the SOUND branch."""

    def __init__(self, cfg: ActivityGateCfg, confirm_windows: int, window_samples: int, vad: Any | None) -> None:
        self.sound = SoundGate(cfg, confirm_windows, window_samples)
        self.speech = SpeechGate(vad, cfg) if vad is not None else None

    def reset(self) -> None:
        self.sound.reset()
        if self.speech is not None:
            self.speech.reset()


# ----------------------------------------------------------------- products
@dataclass
class SpeechSegment:
    """Speech cut out of the stream for the voice branch. ``samples`` is a transient in-memory hand-off
    to the consumer (ASR / voiceprint): it must not be stored, logged or sent anywhere. It is excluded
    from ``repr``; ``meta()`` is the storable, audio-free description."""

    t_start_mono_ms: float
    t_end_mono_ms: float
    t_start_utc: str
    t_end_utc: str
    duration_s: float
    sr: int
    truncated: bool  # the start of the segment had already left the ring buffer
    samples: np.ndarray = field(repr=False, compare=False)

    def meta(self) -> dict[str, Any]:
        return {"t_start_utc": self.t_start_utc, "t_end_utc": self.t_end_utc, "t_start_mono_ms": self.t_start_mono_ms, "t_end_mono_ms": self.t_end_mono_ms,
                "duration_s": self.duration_s, "truncated": self.truncated}


@dataclass
class ReplayReport:
    """Result of ``process_array``: labels, speech metadata (no samples) and the final status."""

    events: list[SoundEvent]
    speech: list[dict[str, Any]]
    status: dict[str, Any]
    duration_s: float


class Function3Pipeline:
    """Ring buffer + ActivityGate + the two branches. One thread feeds it; ``status()`` is thread-safe."""

    def __init__(
        self,
        cfg: AudioCfg,
        clock: Clock,
        monitor: SoundMonitor | None = None,
        voice_sink: Callable[[SpeechSegment], None] | None = None,
        vad: Any | None = None,
        sr: int = SR,
    ) -> None:
        if sr != SR:
            raise ValueError(f"the pipeline works on {SR} Hz audio; resample before feeding it")
        self.cfg = cfg
        self.clock = clock
        self.monitor = monitor
        self.voice_sink = voice_sink
        self.sr = sr
        self.gate_cfg = cfg.gate
        self._lock = threading.RLock()
        if monitor is not None:
            self.win_samples = monitor.window_samples
            self.hop_samples = max(int(round(monitor.hop_s * 100)), 1) * FRAME  # boundaries on the 10 ms grid
            ring_s = max(cfg.buffer_seconds, monitor.window_s + monitor.hop_s + 1.0)
            confirm = monitor.cfg.confirm_windows
        else:
            self.win_samples = int(round(cfg.sound_monitor.window_s * sr))
            self.hop_samples = max(int(round(cfg.sound_monitor.hop_s * 100)), 1) * FRAME
            ring_s = cfg.buffer_seconds
            confirm = cfg.sound_monitor.confirm_windows
        self._voice_error: str | None = None
        if voice_sink is not None and vad is None:
            try:
                from homeease.ml.audio import vad as vadmod

                vad = vadmod.create(cfg.vad)
            except Exception as e:  # missing optional dependency / model: the voice branch stays off, loudly
                self._voice_error = f"{cfg.vad}:{type(e).__name__}"
                log.warning("speech VAD %s unavailable (%s): voice branch disabled", cfg.vad, type(e).__name__)
        self.vad_name = getattr(vad, "name", type(vad).__name__) if vad is not None else None
        self.ring = AudioRing(ring_s, sr)
        self.gate = ActivityGate(cfg.gate, confirm, self.win_samples, vad if voice_sink is not None else None)
        self._t0_ms: float | None = None
        self._carry = np.zeros(0, dtype=np.float32)
        self._frames_done = 0
        self._next_boundary = self.hop_samples
        self._chunks = 0
        self._samples_total = 0
        self._discontinuities = 0
        self._bad_samples = 0
        self._speech_segments = 0
        self._speech_s = 0.0
        self._sink_errors = 0
        self._ring_misses = 0
        self._last_end_ms: float | None = None  # stream time at the end of the newest chunk
        self._speech_log: deque[dict[str, Any]] = deque(maxlen=200)
        self._status_cache: dict[str, Any] | None = None  # what a timed-out status() call returns (stale: true)

    # ------------------------------------------------------------- timeline
    def _t_ms(self, sample: int) -> float:
        assert self._t0_ms is not None
        return self._t0_ms + sample * 1000.0 / self.sr

    def _restart(self, t_mono_ms: float) -> None:
        self._t0_ms = float(t_mono_ms)
        self.ring.clear()
        self._carry = np.zeros(0, dtype=np.float32)
        self._frames_done = 0
        self._next_boundary = self.hop_samples
        self.gate.reset()

    def _discontinuity(self, t_mono_ms: float) -> None:
        self._discontinuities += 1
        for s, e in (self.gate.speech.flush() if self.gate.speech is not None else []):
            self._emit_speech(s, e)  # speech heard before the hole is still valid
        if self.monitor is not None:
            self.monitor.on_discontinuity()
        self._restart(t_mono_ms)

    # ---------------------------------------------------------------- input
    def process_chunk(self, x: np.ndarray, t_mono_ms: float | None = None) -> list[SoundEvent]:
        """Feed one chunk (16 kHz mono) whose first sample was captured at ``t_mono_ms`` (None: contiguous).

        Returns the sound events newly confirmed by this chunk (usually none)."""
        x = self._as_float32(x)
        if len(x) == 0:
            return []
        with self._lock:
            now = self.clock.mono_ms()
            if self._t0_ms is None:
                self._restart(t_mono_ms if t_mono_ms is not None else now - len(x) * 1000.0 / self.sr)
            elif t_mono_ms is not None:
                diff = t_mono_ms - self._t_ms(self.ring.total)
                if abs(diff) > self.gate_cfg.gap_tolerance_ms:
                    self._discontinuity(t_mono_ms)
                else:
                    self._t0_ms += SLEW * diff  # jitter of the caller's timestamps, not a gap
            self._chunks += 1
            out: list[SoundEvent] = []
            step = int(MAX_PIECE_S * self.sr)
            for i in range(0, len(x), step):
                out += self._push(x[i:i + step])
            self._last_end_ms = self._t_ms(self.ring.total)
            return out

    def process_array(self, x: np.ndarray, sr: int = SR, t0_mono_ms: float | None = None, chunk_s: float = 0.1, flush: bool = True) -> ReplayReport:
        """Offline replay of a whole recording through the live code path (tests, CLI demo).

        ``t0_mono_ms`` is the monotonic time of the first sample (default: now on the pipeline clock)."""
        x = np.asarray(x)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = self._as_float32(x)
        if sr != SR:
            x = resample(x, sr, SR)
        t0 = float(self.clock.mono_ms() if t0_mono_ms is None else t0_mono_ms)
        n = max(int(round(chunk_s * SR)), 1)
        for i in range(0, len(x), n):
            self.process_chunk(x[i:i + n], t0 + i * 1000.0 / SR)
        if flush:
            self.flush()
        events = self.monitor.recent if self.monitor is not None else []
        with self._lock:
            speech = list(self._speech_log)
        return ReplayReport(events=sorted(events, key=lambda e: e.t_start_mono_ms), speech=speech, status=self.status(), duration_s=len(x) / SR)

    def _as_float32(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1:
            raise ValueError("the pipeline takes mono audio (1-D array)")
        if a.dtype.kind == "i":
            a = a.astype(np.float32) / float(np.iinfo(a.dtype).max + 1)
        a = a.astype(np.float32, copy=False)
        bad = ~np.isfinite(a)
        if bad.any():  # a corrupt driver buffer must neither poison the gate nor look like sound
            self._bad_samples += int(bad.sum())
            a = np.where(bad, 0.0, a).astype(np.float32)
        return a

    # ---------------------------------------------------------- processing
    def _push(self, piece: np.ndarray) -> list[SoundEvent]:
        self.ring.append(piece)
        self._samples_total += len(piece)
        data = np.concatenate([self._carry, piece]) if len(self._carry) else piece
        n = len(data) // FRAME
        if n:
            fr = data[:n * FRAME].reshape(n, FRAME).astype(np.float64)
            levels = 10.0 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
            if self.monitor is not None:
                self.gate.sound.trigger.push(levels, self._frames_done)
            self._frames_done += n
        self._carry = data[n * FRAME:].copy()
        out: list[SoundEvent] = []
        done = self._frames_done * FRAME
        while self._next_boundary <= done:
            out += self._sound_boundary(self._next_boundary)
            self._next_boundary += self.hop_samples
        if self.gate.speech is not None:
            for s, e in self.gate.speech.step(self.ring, self._frames_done):
                self._emit_speech(s, e)
        return out

    def _sound_boundary(self, b: int) -> list[SoundEvent]:
        mon = self.monitor
        if mon is None:
            return []
        mon.advance(self._t_ms(b))  # events whose cooldown has passed are closed even when this window is not scored
        a = b - self.win_samples
        if a < 0:
            return []
        verdict = self.gate.sound.decide(a, b, lambda: self.ring.get(a, b))
        if verdict is None:
            return []
        trigger, (s0, s1) = verdict
        w = self.ring.get(a, b)
        if len(w) != self.win_samples:
            self._ring_misses += 1
            return []
        return mon.submit_window(w, self._t_ms(a), self._t_ms(b), (s0, s1), trigger)

    def _emit_speech(self, f0: int, f1: int) -> None:
        pad = int(SPEECH_PAD_S * self.sr)
        s, e = f0 * FRAME, f1 * FRAME
        seg_audio = self.ring.get(max(s - pad, 0), e + pad)
        truncated = max(s - pad, 0) < self.ring.oldest
        t0, t1 = self._t_ms(s), self._t_ms(e)
        seg = SpeechSegment(t_start_mono_ms=t0, t_end_mono_ms=t1, t_start_utc=iso(self.clock.mono_to_utc(t0)) or "", t_end_utc=iso(self.clock.mono_to_utc(t1)) or "",
                            duration_s=round((e - s) / self.sr, 3), sr=self.sr, truncated=truncated, samples=seg_audio)
        self._speech_segments += 1
        self._speech_s += seg.duration_s
        self._speech_log.append(seg.meta())
        if self.voice_sink is None:
            return
        try:
            self.voice_sink(seg)
        except Exception as e:  # the consumer must not stop the pipeline; never log its payload
            self._sink_errors += 1
            log.warning("speech segment consumer failed (%d so far): %s", self._sink_errors, type(e).__name__)

    # ------------------------------------------------------------ lifecycle
    def flush(self) -> None:
        """End of stream: settle the VAD, emit the open speech segment, write the open sound events."""
        with self._lock:
            if self._t0_ms is None:
                return
            if self.monitor is not None:
                self.gate.sound.trigger.close_open()
            if self.gate.speech is not None:
                for s, e in self.gate.speech.step(self.ring, self._frames_done, final=True) + self.gate.speech.flush():
                    self._emit_speech(s, e)
            if self.monitor is not None:
                self.monitor.flush()

    def close(self, timeout: float | None = None) -> bool:
        """Stop: flush, then wipe the audio held in memory. Safe to call again and again (consent withdrawn,
        stream lost, shutdown); the next chunk starts a fresh timeline. With ``timeout`` it gives up waiting for a
        worker that is stuck inside a classifier call and returns False (the audio is then wiped by the next close)."""
        if not self._lock.acquire(timeout=-1 if timeout is None else timeout):
            log.warning("pipeline busy: audio buffers not wiped yet")
            return False
        try:
            self.flush()
            if self.ring.total or len(self._carry):
                self.ring.clear()
                self._carry = np.zeros(0, dtype=np.float32)
            self._t0_ms = None  # no stitching across a stop: the next chunk restarts the timeline
            self._frames_done = 0
            self._next_boundary = self.hop_samples
            self.gate.reset()
            return True
        finally:
            self._lock.release()

    # --------------------------------------------------------------- status
    def voice_status(self) -> dict[str, Any]:
        if self.voice_sink is None:
            return {"status": "off", "reason": "no consumer attached"}
        sg = self.gate.speech
        if sg is None:
            return {"status": "degraded:vad_unavailable", "vad": self.cfg.vad, "error": self._voice_error, "segments": 0}
        state = "degraded:vad_error" if sg.errors else "ok"
        return {"status": state, "vad": self.vad_name, "segments": self._speech_segments, "speech_s": round(self._speech_s, 2), "vad_errors": sg.errors, "sink_errors": self._sink_errors}

    def speech_in_progress(self) -> bool:
        """A speech run is open: speech was heard and its end silence is not complete yet. Lock-free (one attribute read, a stale
        answer costs a caller one more poll): a voice inquiry asks it at the end of its response window to let a reply finish."""
        sg = self.gate.speech
        return sg is not None and sg.run.start is not None

    def status(self, timeout: float | None = None) -> dict[str, Any]:
        """Health and counters. The worker holds the lock while it classifies a window and writes to the database;
        with ``timeout`` a caller (an HTTP status endpoint) waits at most that long and otherwise gets the last
        snapshot marked ``stale: true`` instead of hanging behind a slow model."""
        if not self._lock.acquire(timeout=-1 if timeout is None else timeout):
            return {**(self._status_cache or {"status": "degraded:busy", "sound": {"status": "degraded:busy"}, "voice": {"status": "off"}}), "stale": True}
        try:
            out = self._status_locked()
            self._status_cache = out
            return dict(out)
        finally:
            self._lock.release()

    def _status_locked(self) -> dict[str, Any]:
        mon = self.monitor.status() if self.monitor is not None else {"status": "off"}
        voice = self.voice_status()
        degraded = [s for s in (mon["status"], voice["status"]) if s.startswith("degraded")]
        sg = self.gate.sound
        return {
            "status": degraded[0] if degraded else "ok",
            "sound": mon,
            "voice": voice,
            "gate": {
                "noise_floor_db": None if sg.trigger.floor_db is None else round(sg.trigger.floor_db, 1),
                "last_level_db": None if sg.trigger.last_level_db is None else round(sg.trigger.last_level_db, 1),
                "onset_windows": sg.n_onset, "periodic_windows": sg.n_periodic, "quiet_periodic_skips": sg.n_quiet_skips, "idle_windows": sg.n_idle,
                "onset_db": self.gate_cfg.onset_db, "min_level_dbfs": self.gate_cfg.min_level_dbfs, "periodic_s": self.gate_cfg.periodic_s,
            },
            "stream": {
                "chunks": self._chunks, "seconds": round(self._samples_total / self.sr, 2), "buffered_s": round(min(self.ring.total, self.ring.cap) / self.sr, 2),
                "discontinuities": self._discontinuities, "bad_samples": self._bad_samples, "ring_misses": self._ring_misses,
                "last_chunk_utc": iso(self.clock.mono_to_utc(self._last_end_ms)) if self._last_end_ms is not None else None,
            },
            "raw_audio_stored": False,
        }


__all__ = ["ActivityGate", "AudioRing", "Function3Pipeline", "ReplayReport", "SoundGate", "SoundTrigger", "SpeechGate", "SpeechSegment"]
