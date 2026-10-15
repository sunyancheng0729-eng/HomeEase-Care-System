"""Clock abstractions.

Durations and timeouts use a monotonic millisecond clock (``mono_ms``) that is
unaffected by manual wall-clock changes. Wall time is only used for storage and
display, always as timezone-aware UTC. Replay and tests use ``VirtualClock`` so
that detection latency is measured on the recording time axis, not on an
accelerated wall clock.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol


UTC = timezone.utc


class Clock(Protocol):
    def mono_ms(self) -> float: ...

    def utc_now(self) -> datetime: ...

    def mono_to_utc(self, mono_ms: float) -> datetime: ...

    def utc_to_mono(self, utc: datetime) -> float: ...


class SystemClock:
    """Real node clock. The mono->UTC mapping is re-estimated on demand."""

    kind = "system"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._refresh()

    def _refresh(self) -> None:
        # Bracket the wall-clock read with two monotonic reads to bound the error.
        m0 = time.monotonic_ns()
        wall = time.time_ns()
        m1 = time.monotonic_ns()
        mono_ms = (m0 + m1) / 2e6
        with self._lock:
            self._utc_epoch_minus_mono_ms = wall / 1e6 - mono_ms
            self._refreshed_mono = mono_ms

    def mono_ms(self) -> float:
        return time.monotonic_ns() / 1e6

    def utc_now(self) -> datetime:
        return datetime.now(UTC)

    def mono_to_utc(self, mono_ms: float) -> datetime:
        if self.mono_ms() - self._refreshed_mono > 60_000:
            self._refresh()
        with self._lock:
            epoch_ms = mono_ms + self._utc_epoch_minus_mono_ms
        return datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC)

    def utc_to_mono(self, utc: datetime) -> float:
        with self._lock:
            return utc.timestamp() * 1000.0 - self._utc_epoch_minus_mono_ms


class VirtualClock:
    """Deterministic clock for replay, synthetic runs and tests."""

    kind = "virtual"

    def __init__(self, start_mono_ms: float = 0.0, start_utc: datetime | None = None) -> None:
        self._mono = float(start_mono_ms)
        self._base_mono = float(start_mono_ms)
        self._base_utc = start_utc or datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        self._lock = threading.Lock()

    def mono_ms(self) -> float:
        with self._lock:
            return self._mono

    def set(self, mono_ms: float) -> None:
        with self._lock:
            if mono_ms < self._mono:
                raise ValueError("virtual clock cannot go backwards")
            self._mono = float(mono_ms)

    def advance(self, delta_ms: float) -> float:
        with self._lock:
            if delta_ms < 0:
                raise ValueError("negative advance")
            self._mono += float(delta_ms)
            return self._mono

    def utc_now(self) -> datetime:
        return self.mono_to_utc(self.mono_ms())

    def mono_to_utc(self, mono_ms: float) -> datetime:
        return self._base_utc + timedelta(milliseconds=mono_ms - self._base_mono)

    def utc_to_mono(self, utc: datetime) -> float:
        return self._base_mono + (utc - self._base_utc).total_seconds() * 1000.0


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_utc(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo; values written by this app are always UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def iso(dt: datetime | None) -> str | None:
    dt = ensure_utc(dt)
    return dt.isoformat().replace("+00:00", "Z") if dt else None
