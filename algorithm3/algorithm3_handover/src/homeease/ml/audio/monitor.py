"""Daily sound monitoring (proposal 1.5.2, Figure 3-3 branch B): audio windows -> sound-event
classification -> event labelling that keeps only category, timestamps and summary statistics.

Privacy by construction: ``SoundEvent`` has no audio field of any kind, the database table has no
such column, and this module never writes a file. A window is analysed in memory and dropped.

Decision logic (all of it deterministic and testable on a virtual clock):

* a classifier object (``predict_proba(window)`` + ``classes``) scores each window handed over by
  the pipeline; the top class counts only if its probability reaches ``min_conf`` (or the class's own
  cut-off when the model package carries dev-chosen per-class cut-offs), otherwise the outcome is
  UNKNOWN (never silently mapped to DAILY or to "safe");
* ``confirm_windows`` consecutive windows with the same label are needed before an event exists
  (hysteresis against single-window flicker);
* detections of the same category closer than ``cooldown_s`` are merged into one event, so a
  siren or a vacuum cleaner is one event, not hundreds;
* without a classifier the monitor reports ``degraded:model_missing`` and emits nothing. It never
  invents labels;
* an event whose database write failed is kept in memory and written again (every ``RETRY_UNSAVED_MS``
  and at ``flush()``); the status stays ``degraded:storage_error`` until nothing is left unsaved.

Labels are assistive hints; they never raise, close, downgrade or cancel a fall event.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Iterable, Protocol, Sequence

import numpy as np

from homeease.core.clock import UTC, Clock, ensure_utc, iso
from homeease.core.config import SoundMonitorCfg
from homeease.core.util import get_logger, new_id

if TYPE_CHECKING:  # the database layer is only needed by SoundEventStore
    from homeease.backend.storage.db import Database

log = get_logger(__name__)

SR = 16000
CATEGORIES = ("HELP", "IMPACT", "ABNORMAL", "DAILY", "UNKNOWN")
CLASS_NAMES = CATEGORIES[:4]  # classifier classes; UNKNOWN is a decision outcome, never a class
DEFAULT_MIN_CONF = 0.5
FLUSH_EVERY_MS = 10_000.0  # a long event is re-saved at most this often while it keeps growing
RETRY_UNSAVED_MS = 5_000.0  # an event whose database write failed is written again this often (stream time)
MAX_UNSAVED = 500  # bound of the in-memory retry queue; beyond it the oldest unsaved event is dropped and counted


class SoundClassifier(Protocol):
    """What the monitor needs from a model (a trained package or a test stub).

    ``predict_proba(window)`` receives one mono float32 window at 16 kHz of exactly
    ``round(window_s * 16000)`` samples (the classifier's own ``window_s`` if it declares one,
    otherwise ``audio.sound_monitor.window_s``) and returns one probability per entry of
    ``classes`` (finite, in [0, 1]). It must not keep a reference to the window.
    Optional attributes: ``window_s`` (float), ``min_conf`` (dev-chosen threshold), ``min_conf_by_class``
    (dev-chosen per-class cut-offs, ``{"HELP": 0.7, ...}``; classes without an entry use ``min_conf``),
    ``model_version``.
    """

    model_id: str
    classes: Sequence[str]

    def predict_proba(self, window: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class SoundEvent:
    """One labelled sound event. Scalars only: no audio, spectrogram, embedding or transcript."""

    event_id: str
    category: str  # HELP | IMPACT | ABNORMAL | DAILY | UNKNOWN
    confidence: float  # mean top-class probability over the windows that confirmed it
    t_start_utc: str  # ISO 8601 UTC, start of the acoustic activity
    t_end_utc: str
    t_start_mono_ms: float  # same instants on the node's monotonic clock
    t_end_mono_ms: float
    duration_s: float
    rms_db: float  # dBFS over the active span (mean power over the merged windows)
    peak_db: float  # dBFS, maximum over the merged windows
    n_windows: int  # how many scored windows were merged into this event
    model_id: str  # package id@version, or the stub's id

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class _Win:
    """Per-window scalars kept for a run (no samples)."""

    t0: float  # activity span in mono ms
    t1: float
    t_end: float  # window end in mono ms
    conf: float
    power: float  # mean square of the active span
    peak: float


@dataclass
class _Run:
    category: str
    wins: list[_Win] = field(default_factory=list)


@dataclass
class _Open:
    event_id: str
    category: str
    t0: float
    t1: float
    n: int = 0
    conf_sum: float = 0.0
    power_sum: float = 0.0
    peak: float = 0.0
    last_det: float = 0.0
    persisted_at: float = 0.0
    dirty: bool = False


def _db(p: float) -> float:
    return float(10.0 * math.log10(max(p, 1e-12)))


def _validate_classes(classes: Sequence[str]) -> list[str]:
    cl = [str(c) for c in classes]
    if "UNKNOWN" in cl:
        raise ValueError("UNKNOWN is a decision outcome (low confidence), not a class the classifier may emit")
    bad = [c for c in cl if c not in CLASS_NAMES]
    if bad or not cl or len(set(cl)) != len(cl):
        raise ValueError(f"classifier classes must be distinct members of {CLASS_NAMES}, got {cl}")
    return cl


class SoundMonitor:
    """Sliding-window sound-event monitor. One thread feeds it; ``status()`` may be read from any."""

    def __init__(
        self,
        cfg: SoundMonitorCfg,
        classifier: SoundClassifier | None,
        clock: Clock,
        store: "SoundEventStore | None" = None,
        listeners: Iterable[Callable[[SoundEvent, bool], None]] = (),
        sr: int = SR,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self.store = store
        self.listeners = list(listeners)
        self.sr = sr
        self.classifier = classifier
        self.classes: list[str] = _validate_classes(classifier.classes) if classifier is not None else []
        declared = getattr(classifier, "window_s", None)
        self.window_s = float(declared) if declared else float(cfg.window_s)
        if declared and abs(float(declared) - cfg.window_s) > 1e-6:
            log.warning("sound classifier was trained with window_s=%.2f; audio.sound_monitor.window_s=%.2f is ignored", float(declared), cfg.window_s)
        self.hop_s = float(min(cfg.hop_s, self.window_s))
        pkg_min = getattr(classifier, "min_conf", None)
        self.min_conf = float(cfg.min_conf) if cfg.min_conf is not None else float(pkg_min) if pkg_min is not None else DEFAULT_MIN_CONF
        # per-class cut-offs from the package; an explicit config value is a global override and wins over them
        pkg_cuts = getattr(classifier, "min_conf_by_class", None) if cfg.min_conf is None else None
        self.min_conf_by_class: dict[str, float] = {str(c): float(v) for c, v in (pkg_cuts or {}).items() if str(c) in self.classes and 0.0 <= float(v) <= 1.0}
        self.model_id = self._model_label(classifier)
        self._lock = threading.RLock()
        self._run: _Run | None = None
        self._open: dict[str, _Open] = {}
        self._recent: deque[SoundEvent] = deque(maxlen=200)
        self._n_scored = 0
        self._n_skipped = 0
        self._n_events = 0
        self._classifier_errors = 0
        self._consecutive_errors = 0
        self._storage_errors = 0
        self._storage_ok = True
        self._unsaved: dict[str, SoundEvent] = {}  # newest snapshot of every event the database has not accepted yet
        self._events_lost = 0
        self._retry_at_ms = 0.0
        self._status_cache: dict[str, Any] | None = None
        self._last_window_ms: float | None = None
        self._by_trigger: dict[str, int] = {}
        self._by_label: dict[str, int] = {}
        self.status()  # seeds the snapshot a timed-out status() call falls back to

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _model_label(clf: SoundClassifier | None) -> str:
        if clf is None:
            return ""
        ver = getattr(clf, "model_version", None)
        return f"{clf.model_id}@{ver}" if ver else str(clf.model_id)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_s * self.sr))

    @property
    def recent(self) -> list[SoundEvent]:
        with self._lock:
            return list(self._recent)

    def status(self, timeout: float | None = None) -> dict[str, Any]:
        """Counters and health. ``timeout`` bounds the wait for the monitor lock (held while a window is being
        classified and written): on expiry the last snapshot is returned with ``stale: true``."""
        if not self._lock.acquire(timeout=-1 if timeout is None else timeout):
            return {**(self._status_cache or {"status": "degraded:busy"}), "stale": True}
        try:
            if self.classifier is None:
                state = "degraded:model_missing"
            elif self._consecutive_errors > 0:
                state = "degraded:classifier_error"
            elif not self._storage_ok or self._unsaved:
                state = "degraded:storage_error"
            else:
                state = "ok"
            self._status_cache = {
                "status": state,
                "model_id": self.model_id or None,
                "classes": list(self.classes),
                "window_s": self.window_s,
                "hop_s": self.hop_s,
                "min_conf": self.min_conf,
                "min_conf_by_class": dict(self.min_conf_by_class),
                "confirm_windows": self.cfg.confirm_windows,
                "cooldown_s": self.cfg.cooldown_s,
                "windows_scored": self._n_scored,
                "windows_skipped_no_model": self._n_skipped,
                "windows_by_trigger": dict(self._by_trigger),
                "windows_by_label": dict(self._by_label),
                "events_emitted": self._n_events,
                "open_events": len(self._open),
                "classifier_errors": self._classifier_errors,
                "storage_errors": self._storage_errors,
                "events_unsaved": len(self._unsaved),
                "events_lost": self._events_lost,
                "last_window_utc": iso(self.clock.mono_to_utc(self._last_window_ms)) if self._last_window_ms is not None else None,
                "raw_audio_stored": False,
            }
            return dict(self._status_cache)
        finally:
            self._lock.release()

    # ------------------------------------------------------------ scoring
    def submit_window(
        self,
        window: np.ndarray,
        t_start_ms: float,
        t_end_ms: float,
        active_span_s: tuple[float, float] | None = None,
        trigger: str = "onset",
    ) -> list[SoundEvent]:
        """Score one window; returns events that were newly confirmed by it (possibly none).

        ``active_span_s`` is the acoustic activity inside the window (seconds from its start);
        it bounds the event time span and the level statistics. The window is not retained."""
        with self._lock:
            self._advance(t_end_ms)
            self._last_window_ms = t_end_ms
            if self.classifier is None:
                self._n_skipped += 1
                return []
            try:
                probs = self._predict(window)
            except Exception as e:  # a failing model must never take the pipeline down
                self._classifier_errors += 1
                self._consecutive_errors += 1
                self._run = None
                if self._classifier_errors <= 3 or self._classifier_errors % 100 == 0:
                    log.warning("sound classifier failed (%d so far): %s", self._classifier_errors, type(e).__name__)
                return []
            self._consecutive_errors = 0
            self._n_scored += 1
            self._by_trigger[trigger] = self._by_trigger.get(trigger, 0) + 1
            k = int(np.argmax(probs))
            label = self.classes[k] if probs[k] >= self.min_conf_by_class.get(self.classes[k], self.min_conf) else "UNKNOWN"
            self._by_label[label] = self._by_label.get(label, 0) + 1
            t0, t1, seg = self._active_part(window, t_start_ms, t_end_ms, active_span_s)
            win = _Win(t0=t0, t1=t1, t_end=t_end_ms, conf=float(probs[k]), power=float(np.mean(seg.astype(np.float64) ** 2)) if len(seg) else 0.0,
                       peak=float(np.max(np.abs(seg))) if len(seg) else 0.0)
            return self._track(label, win)

    def _predict(self, window: np.ndarray) -> np.ndarray:
        x = np.asarray(window, dtype=np.float32).reshape(-1)
        if len(x) != self.window_samples:
            raise ValueError(f"window has {len(x)} samples, expected {self.window_samples}")
        p = np.asarray(self.classifier.predict_proba(x), dtype=np.float64).reshape(-1)  # type: ignore[union-attr]
        if p.shape != (len(self.classes),) or not np.all(np.isfinite(p)) or p.min() < -1e-6 or p.max() > 1.0 + 1e-6:
            raise ValueError("classifier returned an invalid probability vector")
        return np.clip(p, 0.0, 1.0)

    def _active_part(self, window: np.ndarray, t_start_ms: float, t_end_ms: float, span: tuple[float, float] | None) -> tuple[float, float, np.ndarray]:
        if span is None:
            return t_start_ms, t_end_ms, np.asarray(window, dtype=np.float32)
        a = max(0.0, min(span[0], span[1]))
        b = min((t_end_ms - t_start_ms) / 1000.0, max(span[0], span[1]))
        i0, i1 = int(a * self.sr), max(int(a * self.sr) + 1, int(b * self.sr))
        return t_start_ms + a * 1000.0, t_start_ms + b * 1000.0, np.asarray(window[i0:i1], dtype=np.float32)

    def _track(self, label: str, win: _Win) -> list[SoundEvent]:
        n = self.cfg.confirm_windows
        run = self._run
        consecutive = run is not None and bool(run.wins) and win.t_end - run.wins[-1].t_end <= 1.5 * self.hop_s * 1000.0
        if run is None or run.category != label or not consecutive:
            run = self._run = _Run(label)
        run.wins.append(win)
        if len(run.wins) < n:
            return []
        # just confirmed: all n windows of the run count; afterwards every further window counts
        batch = list(run.wins) if len(run.wins) == n else [win]
        new_open: _Open | None = None
        for w in batch:
            o, is_new = self._absorb(label, w)
            if is_new:
                new_open = o
        if new_open is None:
            return []
        created = self._snapshot(new_open)  # taken after the whole batch so the first stored row is complete
        self._n_events += 1
        self._recent.append(created)
        self._persist(created, update=False)  # a failed first write is kept in self._unsaved and retried (never lost silently)
        new_open.persisted_at, new_open.dirty = win.t_end, False
        return [created]

    def _absorb(self, label: str, w: _Win) -> tuple[_Open, bool]:
        """Add a confirmed window to the open event of its category (new event after the cooldown)."""
        o = self._open.get(label)
        if o is not None and w.t0 - o.last_det > self.cfg.cooldown_s * 1000.0:
            self._finalize(o)
            o = None
        is_new = o is None
        if o is None:
            o = _Open(event_id=new_id("snd_"), category=label, t0=w.t0, t1=w.t1, last_det=w.t_end)
            self._open[label] = o
        o.t0, o.t1 = min(o.t0, w.t0), max(o.t1, w.t1)
        o.n += 1
        o.conf_sum += w.conf
        o.power_sum += w.power
        o.peak = max(o.peak, w.peak)
        o.last_det = max(o.last_det, w.t_end)
        o.dirty = True
        return o, is_new

    def _snapshot(self, o: _Open) -> SoundEvent:
        return SoundEvent(
            event_id=o.event_id, category=o.category, confidence=round(o.conf_sum / max(o.n, 1), 4),
            t_start_utc=iso(self.clock.mono_to_utc(o.t0)) or "", t_end_utc=iso(self.clock.mono_to_utc(o.t1)) or "",
            t_start_mono_ms=o.t0, t_end_mono_ms=o.t1, duration_s=round(max(o.t1 - o.t0, 0.0) / 1000.0, 3),
            rms_db=round(_db(o.power_sum / max(o.n, 1)), 2), peak_db=round(20.0 * math.log10(max(o.peak, 1e-6)), 2),
            n_windows=o.n, model_id=self.model_id,
        )

    def _write(self, ev: SoundEvent) -> bool:
        """One database write. A failure keeps the newest snapshot of the event for the retry (``_retry_unsaved``)."""
        if self.store is None:
            return True
        try:
            self.store.save(ev)
        except Exception as e:
            self._storage_errors += 1
            self._storage_ok = False
            if ev.event_id not in self._unsaved and len(self._unsaved) >= MAX_UNSAVED:
                self._unsaved.pop(next(iter(self._unsaved)))  # bounded memory: the oldest unsaved event is given up and counted
                self._events_lost += 1
            self._unsaved[ev.event_id] = ev
            log.warning("sound event not stored (%d so far, %d waiting): %s", self._storage_errors, len(self._unsaved), type(e).__name__)
            return False
        self._unsaved.pop(ev.event_id, None)
        self._storage_ok = True
        return True

    def _persist(self, ev: SoundEvent, update: bool) -> bool:
        ok = self._write(ev)
        for fn in self.listeners:
            try:
                fn(ev, update)
            except Exception:  # a listener must not stop the monitor
                log.exception("sound event listener failed")
        return ok

    def _retry_unsaved(self, now_ms: float, force: bool = False) -> None:
        """Write events again that the database refused. A snapshot that is not the latest of its (still open) event
        is replaced by the open event's own snapshot first, so a retry never writes older numbers over newer ones."""
        if not self._unsaved or (not force and now_ms < self._retry_at_ms):
            return
        self._retry_at_ms = now_ms + RETRY_UNSAVED_MS
        for eid, ev in list(self._unsaved.items()):
            o = next((x for x in self._open.values() if x.event_id == eid), None)
            snap = self._snapshot(o) if o is not None else ev
            if self._write(snap) and o is not None:
                o.persisted_at, o.dirty = now_ms, False
            if not self._storage_ok:
                break  # the database is still down: one failure per retry round is enough

    def _finalize(self, o: _Open) -> None:
        if o.dirty:
            snap = self._snapshot(o)
            self._replace_recent(snap)
            self._persist(snap, update=True)
            o.dirty = False
        self._open.pop(o.category, None)

    def _replace_recent(self, snap: SoundEvent) -> None:
        for i, e in enumerate(self._recent):
            if e.event_id == snap.event_id:
                self._recent[i] = snap
                return

    def advance(self, now_ms: float) -> None:
        """Close events whose cooldown has passed and flush long ones (call once per hop)."""
        with self._lock:
            self._advance(now_ms)

    def _advance(self, now_ms: float) -> None:
        cooldown_ms = self.cfg.cooldown_s * 1000.0
        for o in list(self._open.values()):
            if now_ms - o.last_det > cooldown_ms:
                self._finalize(o)
            elif o.dirty and now_ms - o.persisted_at >= FLUSH_EVERY_MS:
                snap = self._snapshot(o)
                self._replace_recent(snap)
                self._persist(snap, update=True)  # a failed write stays in self._unsaved and is retried below
                o.persisted_at, o.dirty = now_ms, False
        self._retry_unsaved(now_ms)

    def on_discontinuity(self) -> None:
        """The audio timeline restarted (gap, device change): consecutive windows are no longer consecutive."""
        with self._lock:
            self._run = None

    def flush(self) -> None:
        """End of stream: write every open event (and retry the ones the database refused) and forget the current run."""
        with self._lock:
            self._run = None
            for o in list(self._open.values()):
                self._finalize(o)
            self._retry_unsaved(0.0, force=True)


# ---------------------------------------------------------------- storage
class SoundEventStore:
    """Persists labels only (``sound_events`` table). Upsert by event id; rows are scalars."""

    def __init__(self, db: "Database", subject_id: str, room_id: str | None = None, source_mode: str = "live", clock: Clock | None = None) -> None:
        self.db = db
        self.subject_id = subject_id
        self.room_id = room_id
        self.source_mode = source_mode
        self.clock = clock

    def save(self, ev: SoundEvent) -> None:
        from homeease.backend.storage.models import SoundEventRecord

        now = self.clock.utc_now() if self.clock is not None else datetime.now(UTC)
        with self.db.session() as s:
            row = s.get(SoundEventRecord, ev.event_id)
            if row is None:
                row = SoundEventRecord(id=ev.event_id, subject_id=self.subject_id, room_id=self.room_id, source_mode=self.source_mode, created_at=now)
                s.add(row)
            row.category = ev.category
            row.confidence = float(ev.confidence)
            row.t_start = _parse_iso(ev.t_start_utc)
            row.t_end = _parse_iso(ev.t_end_utc)
            row.t_start_mono_ms = float(ev.t_start_mono_ms)
            row.t_end_mono_ms = float(ev.t_end_mono_ms)
            row.duration_s = float(ev.duration_s)
            row.rms_db = float(ev.rms_db)
            row.peak_db = float(ev.peak_db)
            row.n_windows = int(ev.n_windows)
            row.model_id = ev.model_id[:128]
            row.updated_at = now

    @staticmethod
    def _to_event(r: Any) -> SoundEvent:
        return SoundEvent(
            event_id=r.id, category=r.category, confidence=r.confidence, t_start_utc=iso(r.t_start) or "", t_end_utc=iso(r.t_end) or "",
            t_start_mono_ms=r.t_start_mono_ms, t_end_mono_ms=r.t_end_mono_ms, duration_s=r.duration_s, rms_db=r.rms_db, peak_db=r.peak_db,
            n_windows=r.n_windows, model_id=r.model_id,
        )

    def recent(self, limit: int = 50) -> list[SoundEvent]:
        from sqlalchemy import select

        from homeease.backend.storage.models import SoundEventRecord

        with self.db.session() as s:
            rows = s.execute(select(SoundEventRecord).where(SoundEventRecord.subject_id == self.subject_id).order_by(SoundEventRecord.t_start.desc()).limit(limit)).scalars().all()
            return [self._to_event(r) for r in rows]

    def between(self, start: datetime, end: datetime) -> list[SoundEvent]:
        from sqlalchemy import select

        from homeease.backend.storage.models import SoundEventRecord

        with self.db.session() as s:
            q = select(SoundEventRecord).where(SoundEventRecord.subject_id == self.subject_id, SoundEventRecord.t_start >= ensure_utc(start), SoundEventRecord.t_start < ensure_utc(end))
            return [self._to_event(r) for r in s.execute(q.order_by(SoundEventRecord.t_start)).scalars()]

    def daily_summary(self, day: date | None = None, tz: str = "UTC", now: datetime | None = None) -> dict[str, Any]:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(tz)
        local_day = day or (now or (self.clock.utc_now() if self.clock is not None else datetime.now(UTC))).astimezone(zone).date()
        start = datetime(local_day.year, local_day.month, local_day.day, tzinfo=zone)
        end = (datetime(local_day.year, local_day.month, local_day.day) + timedelta(days=1)).replace(tzinfo=zone)
        return daily_summary(self.between(start, end), local_day, tz)

    def prune(self, retention_days: int, now: datetime | None = None) -> int:
        """Retention (privacy.retention_days_events): delete events that ended before the cutoff."""
        from sqlalchemy import delete

        from homeease.backend.storage.models import SoundEventRecord

        cutoff = (now or (self.clock.utc_now() if self.clock is not None else datetime.now(UTC))) - timedelta(days=retention_days)
        with self.db.session() as s:
            res = s.execute(delete(SoundEventRecord).where(SoundEventRecord.t_end < cutoff))
            return int(res.rowcount or 0)


def _parse_iso(v: str) -> datetime:
    dt = ensure_utc(datetime.fromisoformat(v.replace("Z", "+00:00")))
    assert dt is not None
    return dt


def daily_summary(events: Iterable[SoundEvent], day: date, tz: str = "UTC") -> dict[str, Any]:
    """Counts per category per local hour for one day: the "necessary statistical information"
    that replaces any audio storage. UNKNOWN is reported as its own bucket, never folded into DAILY."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)
    hours = [{"hour": h, **{c: 0 for c in CATEGORIES}} for h in range(24)]
    totals = {c: 0 for c in CATEGORIES}
    for e in events:
        t = _parse_iso(e.t_start_utc).astimezone(zone)
        if t.date() != day or e.category not in totals:
            continue
        hours[t.hour][e.category] += 1
        totals[e.category] += 1
    return {
        "date": day.isoformat(), "timezone": tz, "categories": list(CATEGORIES), "total": sum(totals.values()), "by_category": totals, "hourly": hours,
        "note": "counts of labelled events only; no audio is stored. Labels are assistive hints and never change an alert.",
    }
