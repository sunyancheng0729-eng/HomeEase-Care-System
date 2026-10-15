"""Alarm markers of an event and its single / dual / triple channel tag. Pure functions, no database.

A marker says "this channel raised an alarm for the event at this time": {channel, at, kind, counted}. The manager fixed the
statistics window at 10 s. The window starts at the earliest marker of a counted channel (accel, vision, audio); the markers
of counted channels inside [t0, t0 + 10 s] are counted and the tag is the number of DISTINCT counted channels in it
(1 single_channel, 2 dual_channel, 3 triple_channel). A later marker is kept with counted=false and never changes the tag.
The window is the one of the event's first marker also when a later burst of the same event has channels that agree with each other:
those markers are recorded (counted=false), the tag stays the one of the first window.
The help button (channel "manual") is recorded and never counted; no counted marker means no tag.

The tag depends on the marker times only, not on the order they arrive in (an earlier marker that arrives late moves the
window start), so the stored ``counted`` flags are recomputed on every change and the list is kept in time order.

Chinese display names for the tag values, kept by the web UI in its own i18n tables:
single_channel = \u5355\u901a\u9053, dual_channel = \u53cc\u901a\u9053, triple_channel = \u4e09\u901a\u9053.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

WINDOW_S = 10.0  # fixed by the manager; deliberately not a configuration value
COUNTED_CHANNELS = ("accel", "vision", "audio")
CHANNELS = COUNTED_CHANNELS + ("manual",)
TAG_BY_COUNT = {1: "single_channel", 2: "dual_channel", 3: "triple_channel"}
TAGS = tuple(TAG_BY_COUNT.values())


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def stamp(dt: datetime) -> str:
    """UTC ISO-8601 with milliseconds (``2026-01-01T00:00:03.250Z``): the precision a marker keeps, so a boundary comparison is exact."""
    d = _utc(dt)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + f".{d.microsecond // 1000:03d}Z"


def parse(ts: str) -> datetime:
    return _utc(datetime.fromisoformat(ts.replace("Z", "+00:00")))


def _normalise(markers: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Time-ordered copy with ``counted`` recomputed from the times alone."""
    rows = sorted(((parse(m["at"]), CHANNELS.index(m["channel"]), str(m["kind"])) for m in markers))
    starts = [t for t, ch, _ in rows if CHANNELS[ch] in COUNTED_CHANNELS]
    if not starts:
        return [{"channel": CHANNELS[ch], "at": stamp(t), "kind": kind, "counted": False} for t, ch, kind in rows]
    end = min(starts) + timedelta(seconds=WINDOW_S)
    return [{"channel": CHANNELS[ch], "at": stamp(t), "kind": kind, "counted": CHANNELS[ch] in COUNTED_CHANNELS and t <= end} for t, ch, kind in rows]


def add_marker(markers: Iterable[dict[str, Any]] | None, channel: str, at: datetime | str, kind: str) -> list[dict[str, Any]]:
    """The marker list with one more marker. The same channel and kind again within WINDOW_S of a stored marker is the same alarm (the
    decisions of one fall repeat their evidence): it is not stored again and the earlier time is kept. A repeat farther away is another
    burst of the same event: stored for the record, and outside the window of the first marker it is never counted."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown alarm channel {channel!r}")
    when = parse(at) if isinstance(at, str) else _utc(at)
    when = when.replace(microsecond=(when.microsecond // 1000) * 1000)
    cur = [dict(m) for m in markers or []]
    same = [m for m in cur if m["channel"] == channel and m["kind"] == kind and abs((parse(m["at"]) - when).total_seconds()) < WINDOW_S]
    if same:
        m = min(same, key=lambda x: parse(x["at"]))
        if when < parse(m["at"]):
            m["at"] = stamp(when)
        return _normalise(cur)
    cur.append({"channel": channel, "at": stamp(when), "kind": kind, "counted": False})
    return _normalise(cur)


def counted_channels(markers: Iterable[dict[str, Any]] | None) -> list[str]:
    """The distinct counted channels inside the window, in the order accel, vision, audio."""
    inside = {m["channel"] for m in _normalise(markers or []) if m["counted"]}
    return [c for c in COUNTED_CHANNELS if c in inside]


def compute_tag(markers: Iterable[dict[str, Any]] | None) -> str | None:
    """single_channel / dual_channel / triple_channel, or None when no counted channel has a marker."""
    return TAG_BY_COUNT.get(len(counted_channels(markers)))
