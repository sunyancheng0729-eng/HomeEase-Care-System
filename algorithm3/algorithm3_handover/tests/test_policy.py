"""The decision rules around the voice check, as plain values. The scenarios mirror the node's scenario tests (tests/test_alarm_events.py and
tests/test_alarm_ui.py of the HomeEase repository): a reply never closes or downgrades an event."""

from __future__ import annotations

import pytest

from algorithm3.policy import alarm_tag, inquiry_outcome, voice_cancel_verdict

FINE = {"status": "RESPONDED", "intent": "REASSURANCE", "asr_avg_logprob": -0.3, "prompt_played": True}
SILENCE = {"status": "NO_RESPONSE", "intent": None, "prompt_played": True}
HELP = {"status": "RESPONDED", "intent": "HELP", "asr_avg_logprob": -0.4, "prompt_played": True}


def test_a_trusted_im_fine_only_stops_the_escalation():
    o = inquiry_outcome(FINE)
    assert o["state"] == "cancelled" and o["by"] == "voice" and o["cancelled"] is True
    assert o["escalate"] is False and o["marker"] is None  # no audio channel, the event stays open for a caregiver


@pytest.mark.parametrize("rec", [
    SILENCE,
    {**FINE, "intent": "OTHER"},
    {**FINE, "intent": "UNCERTAIN"},
    {**FINE, "asr_avg_logprob": -1.6},                       # recogniser not sure
    {**FINE, "speaker_verified": False, "voiceprint_enrolled": True},  # a voice that does not match the enrolment
    {**FINE, "speaker_verified": None, "voiceprint_enrolled": True},   # an enrolment exists but the voices could not be compared
])
def test_every_other_outcome_is_no_cancel_and_escalates_a_pending_event(rec):
    o = inquiry_outcome(rec, decision_level="PENDING")
    assert o["state"] == "no_cancel" and o["marker"] == "inquiry_no_cancel" and o["escalate"] is True and o["cancelled"] is False


def test_the_older_adult_button_cancels_like_a_trusted_reply():
    o = inquiry_outcome(SILENCE, button_ok=True)
    assert o["state"] == "cancelled" and o["by"] == "button" and o["escalate"] is False


def test_a_spoken_call_for_help_is_never_cancelled_by_the_button():
    o = inquiry_outcome(HELP, button_ok=True)
    assert o["state"] == "no_cancel" and o["marker"] == "spoken_help" and o["escalate"] is True


def test_a_call_for_help_on_an_emergency_event_notifies_and_does_not_escalate_again():
    o = inquiry_outcome(HELP, decision_level="EMERGENCY")
    assert o["escalate"] is False and o["notify"] is True


@pytest.mark.parametrize("status", ["not_run", "audio_unavailable", "garbage", None])
def test_a_check_that_could_not_be_made_changes_nothing(status):
    o = inquiry_outcome({"status": status, "prompt_played": True})
    assert o["marker"] is None and o["escalate"] is False and o["cancelled"] is False
    assert o["state"] in ("not_run", "audio_unavailable")


def test_a_question_that_reached_nobody_is_neither_cancel_nor_no_cancel_unless_it_was_a_call_for_help():
    assert inquiry_outcome({**SILENCE, "prompt_played": False})["state"] == "not_run"
    assert inquiry_outcome({**FINE, "prompt_played": False})["state"] == "not_run"
    assert inquiry_outcome({**HELP, "prompt_played": False})["escalate"] is True


def test_the_escalation_option_switches_only_the_escalation_off():
    o = inquiry_outcome(SILENCE, escalate_on_no_cancel=False)
    assert o["marker"] == "inquiry_no_cancel" and o["escalate"] is False
    assert inquiry_outcome(HELP, escalate_on_no_cancel=False)["escalate"] is True  # a call for help always escalates


def test_a_closed_event_is_judged_for_the_record_and_not_touched():
    o = inquiry_outcome(SILENCE, resolved=True)
    assert o["state"] == "no_cancel" and o["escalate"] is False and o["marker"] is None
    assert inquiry_outcome(FINE, resolved=True)["cancelled"] is False
    assert inquiry_outcome(HELP, resolved=True)["call"] is True


def test_verdict_edge_cases():
    assert voice_cancel_verdict({"intent": "REASSURANCE", "asr_avg_logprob": float("nan")}) is False
    assert voice_cancel_verdict({"intent": "REASSURANCE"}) is True       # no confidence reported, no enrolment
    assert voice_cancel_verdict({"intent": "HELP"}) is False


def test_one_two_three_channels_inside_ten_seconds():
    one = alarm_tag([{"channel": "vision", "at_s": 0}])
    two = alarm_tag([{"channel": "vision", "at_s": 0}, {"channel": "audio", "at_s": 4}])
    three = alarm_tag([{"channel": "accel", "at_s": 0}, {"channel": "vision", "at_s": 2}, {"channel": "audio", "at_s": 6.5}])
    assert (one["tag"], two["tag"], three["tag"]) == ("single_channel", "dual_channel", "triple_channel")


def test_a_marker_after_the_window_is_recorded_and_not_counted():
    t = alarm_tag([{"channel": "vision", "at_s": 0}, {"channel": "accel", "at_s": 10.0}, {"channel": "audio", "at_s": 10.001}])
    assert t["tag"] == "dual_channel" and [m["counted"] for m in t["markers"]] == [True, True, False]


def test_the_help_button_is_recorded_and_never_counted():
    t = alarm_tag([{"channel": "manual", "at_s": 0}])
    assert t["tag"] is None and t["markers"][0]["counted"] is False
    assert alarm_tag([{"channel": "manual", "at_s": 0}, {"channel": "vision", "at_s": 1}])["tag"] == "single_channel"
