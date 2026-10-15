"""Alarm markers and the single / dual / triple channel tag (services/alarm_tags.py): pure functions, the fixed 10 s statistics window."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pytest

from homeease.backend.services import alarm_tags as at

T0 = datetime(2026, 1, 1, 8, 0, 0, tzinfo=timezone.utc)


def build(*items: tuple[str, float, str]) -> list[dict]:
    """Markers added one after the other: (channel, seconds after T0, kind)."""
    out: list[dict] = []
    for channel, dt, kind in items:
        out = at.add_marker(out, channel, T0 + timedelta(seconds=dt), kind)
    return out


def by_channel(markers: list[dict]) -> dict[str, dict]:
    return {m["channel"]: m for m in markers}


def test_the_window_is_fixed_at_ten_seconds() -> None:
    assert at.WINDOW_S == 10.0


def test_one_two_and_three_distinct_channels_make_single_dual_and_triple() -> None:
    assert at.compute_tag(build(("vision", 0.0, "fall_candidate"))) == "single_channel"
    assert at.compute_tag(build(("accel", 0.0, "fall_candidate"), ("vision", 2.5, "fall_candidate"))) == "dual_channel"
    assert at.compute_tag(build(("accel", 0.0, "fall_candidate"), ("vision", 2.5, "fall_candidate"), ("audio", 4.0, "inquiry_no_cancel"))) == "triple_channel"
    assert at.compute_tag(build(("vision", 1.0, "fall_candidate"), ("audio", 3.0, "spoken_help"))) == "dual_channel"


def test_a_marker_exactly_ten_seconds_after_the_first_is_inside_the_window() -> None:
    m = build(("accel", 0.0, "fall_candidate"), ("vision", 10.0, "fall_candidate"))
    assert at.compute_tag(m) == "dual_channel" and by_channel(m)["vision"]["counted"] is True


def test_a_marker_one_millisecond_later_is_stored_but_not_counted() -> None:
    m = build(("accel", 0.0, "fall_candidate"), ("vision", 10.001, "fall_candidate"))
    assert at.compute_tag(m) == "single_channel"
    assert by_channel(m)["accel"]["counted"] is True and by_channel(m)["vision"]["counted"] is False
    m = build(("accel", 0.0, "fall_candidate"), ("vision", 3.0, "fall_candidate"), ("audio", 10.001, "inquiry_no_cancel"))
    assert at.compute_tag(m) == "dual_channel" and by_channel(m)["audio"]["counted"] is False  # the late third channel does not upgrade the tag
    assert len(m) == 3  # and it is kept for the record


def test_the_window_starts_at_the_earliest_counted_marker_not_at_the_first_arrival() -> None:
    m = build(("vision", 5.0, "fall_candidate"), ("accel", 12.0, "fall_candidate"))
    assert at.compute_tag(m) == "dual_channel"  # 12 s is 7 s after the first marker, not after T0
    m = at.add_marker(m, "audio", T0 + timedelta(seconds=1.0), "inquiry_no_cancel")  # an earlier marker arrives late: the window moves back
    assert by_channel(m)["audio"]["counted"] is True and by_channel(m)["vision"]["counted"] is True and by_channel(m)["accel"]["counted"] is False
    assert at.compute_tag(m) == "dual_channel"


def test_the_same_channel_twice_counts_once() -> None:
    m = build(("vision", 0.0, "fall_candidate"), ("vision", 4.0, "fall_candidate"))
    assert len(m) == 1 and m[0]["at"] == "2026-01-01T08:00:00.000Z"  # same channel and kind: not stored again, the earlier time is kept
    assert at.compute_tag(m) == "single_channel"
    m = build(("audio", 0.0, "spoken_help"), ("audio", 4.0, "inquiry_no_cancel"))  # two kinds of one channel are two markers but one channel
    assert len(m) == 2 and at.compute_tag(m) == "single_channel" and at.counted_channels(m) == ["audio"]


def test_the_earlier_time_wins_when_a_duplicate_arrives_with_an_earlier_time() -> None:
    m = build(("vision", 6.0, "fall_candidate"), ("vision", 2.0, "fall_candidate"))
    assert len(m) == 1 and m[0]["at"] == "2026-01-01T08:00:02.000Z"


def test_the_help_button_is_recorded_and_never_counted() -> None:
    m = build(("manual", 0.0, "help_button"))
    assert at.compute_tag(m) is None and m[0]["counted"] is False and len(m) == 1
    m = build(("manual", 0.0, "help_button"), ("audio", 1.0, "inquiry_no_cancel"))
    assert at.compute_tag(m) == "single_channel" and by_channel(m)["manual"]["counted"] is False
    m = build(("accel", 0.0, "fall_candidate"), ("manual", 1.0, "help_button"), ("vision", 2.0, "fall_candidate"))
    assert at.compute_tag(m) == "dual_channel"  # a manual marker never makes a third channel
    m = build(("manual", 0.0, "help_button"), ("vision", 20.0, "fall_candidate"))
    assert at.compute_tag(m) == "single_channel" and by_channel(m)["vision"]["counted"] is True  # the window is the counted channels' own, not the button's


def test_no_counted_marker_means_no_tag() -> None:
    assert at.compute_tag([]) is None and at.compute_tag(None) is None
    assert at.counted_channels(None) == []


def test_the_order_of_arrival_does_not_matter() -> None:
    items = [("accel", 1.0, "fall_candidate"), ("vision", 8.0, "fall_candidate"), ("audio", 11.5, "inquiry_no_cancel"), ("manual", 0.5, "help_button")]
    results = {(at.compute_tag(m), tuple((x["channel"], x["at"], x["counted"]) for x in m)) for m in (build(*p) for p in itertools.permutations(items))}
    assert len(results) == 1
    tag, rows = results.pop()
    assert tag == "dual_channel"  # the audio marker is 10.5 s after the accel marker
    assert [r[0] for r in rows] == ["manual", "accel", "vision", "audio"] and [r[2] for r in rows] == [False, True, True, False]  # time order, counted recomputed


def test_marker_shape_and_time_format() -> None:
    m = build(("audio", 3.25, "spoken_help"))
    assert m == [{"channel": "audio", "at": "2026-01-01T08:00:03.250Z", "kind": "spoken_help", "counted": True}]
    # a string time, a naive datetime (read as UTC) and another time zone all end up as UTC with milliseconds
    sg = timezone(timedelta(hours=8))
    got = at.add_marker([], "vision", datetime(2026, 1, 1, 16, 0, 0, 123999, tzinfo=sg), "fall_candidate")
    assert got[0]["at"] == "2026-01-01T08:00:00.123Z"
    assert at.add_marker([], "vision", "2026-01-01T08:00:00+00:00", "fall_candidate")[0]["at"] == "2026-01-01T08:00:00.000Z"
    assert at.add_marker([], "vision", datetime(2026, 1, 1, 8, 0, 0), "fall_candidate")[0]["at"] == "2026-01-01T08:00:00.000Z"
    assert at.parse(got[0]["at"]) == datetime(2026, 1, 1, 8, 0, 0, 123000, tzinfo=timezone.utc)


def test_the_input_list_is_not_changed_and_unknown_channels_are_refused() -> None:
    before = build(("accel", 0.0, "fall_candidate"))
    snapshot = [dict(m) for m in before]
    after = at.add_marker(before, "vision", T0 + timedelta(seconds=1), "fall_candidate")
    assert before == snapshot and len(after) == 2
    with pytest.raises(ValueError):
        at.add_marker(before, "scream", T0, "sound_label")  # a sound-event label is never an alarm marker channel


def test_tag_values_are_the_agreed_vocabulary() -> None:
    assert at.TAGS == ("single_channel", "dual_channel", "triple_channel")
    assert at.COUNTED_CHANNELS == ("accel", "vision", "audio")


def test_a_repeat_of_the_same_alarm_farther_than_the_window_is_recorded_and_never_counted() -> None:
    # the decisions of one fall repeat their evidence: the same channel and kind within the window is the same alarm
    m = build(("accel", 0.0, "fall_candidate"), ("accel", 9.999, "fall_candidate"))
    assert len(m) == 1 and m[0]["at"] == "2026-01-01T08:00:00.000Z"
    # a later burst of the same event is another alarm: kept for the record, outside the first window it counts for nothing
    m = build(("accel", 0.0, "fall_candidate"), ("accel", 45.0, "fall_candidate"), ("vision", 45.0, "fall_candidate"))
    assert [(x["channel"], x["at"], x["counted"]) for x in m] == [("accel", "2026-01-01T08:00:00.000Z", True), ("accel", "2026-01-01T08:00:45.000Z", False), ("vision", "2026-01-01T08:00:45.000Z", False)]
    assert at.compute_tag(m) == "single_channel" and at.counted_channels(m) == ["accel"]  # the event's first window decides the tag
    # exactly one window apart is already another burst (and, being inside the first window, it is counted without adding a channel)
    m = build(("accel", 0.0, "fall_candidate"), ("accel", 10.0, "fall_candidate"))
    assert len(m) == 2 and [x["counted"] for x in m] == [True, True] and at.compute_tag(m) == "single_channel"


def test_a_repeat_marker_is_bounded_by_the_length_of_the_event() -> None:
    items = [("vision", float(i), "fall_candidate") for i in range(0, 61)]  # a decision every second for a minute (the merge gap of one event)
    m = build(*items)
    assert len(m) == 7 and m[0]["at"] == "2026-01-01T08:00:00.000Z"  # one per window, the first of each; not 61
