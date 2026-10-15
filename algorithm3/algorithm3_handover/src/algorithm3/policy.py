"""Decision rules around the voice check and the alarm tag, as pure functions (no database, no clock, no HTTP).

In the HomeEase node these rules live in ``backend/services/events.py`` (``_voice_cancel_verdict`` and ``_voice_outcome``) next to the
event store. A new system has no ``EventService``, so the same rules are given here as functions of plain values. The functions below
mirror the node's code line by line; the scenarios in ``tests/test_policy.py`` are the node's scenario tests written as plain
values. ``alarm_tag`` and ``add_marker`` are the node's own module (``backend/services/alarm_tags.py``), unchanged.

Rules, in one place (a reply never closes or downgrades an event):

* a valid cancel is a trusted spoken "I'm fine" (intent REASSURANCE, acceptable ASR confidence and, when a voiceprint enrolment
  exists, a matching voice) or the older adult's "I'm OK" button pressed after the question was asked. It only stops the escalation.
* anything else that was judged (silence, unclear or unrelated speech, a call for help, an untrusted "I'm fine") is no cancel: an
  audio alarm marker, and a PENDING event is raised to EMERGENCY.
* a check that could not be made (audio off, no consent, microphone or recogniser failure, a question that reached nobody) is neither:
  no marker, no escalation, shown as degraded. A spoken call for help is always an alarm.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from homeease.backend.services import alarm_tags

INQUIRY_JUDGED = ("RESPONDED", "NO_RESPONSE", "UNCERTAIN")  # records that say how the person answered
DEFAULT_MIN_ASR_LOGPROB = -1.0  # audio.voiceprint.min_asr_logprob


def voice_cancel_verdict(rec: dict[str, Any], min_asr_logprob: float = DEFAULT_MIN_ASR_LOGPROB) -> bool:
    """Is this spoken reply a trusted "I'm fine"? Mirror of ``EventService._voice_cancel_verdict``.

    Needs the REASSURANCE intent, an acceptable ASR confidence when there is one and, when the household has a voiceprint enrolment, a
    matching voice (``speaker_verified`` True). A verdict of False is a mismatch. No verdict passes only without an enrolment;
    ``voiceprint_enrolled`` says there is one even when the voices could not be compared (reply too short)."""
    if rec.get("intent") != "REASSURANCE":
        return False
    conf = rec.get("asr_avg_logprob")
    if conf is not None:
        try:
            if not float(conf) >= min_asr_logprob:  # also catches NaN
                return False
        except (TypeError, ValueError):
            return False
    verdict = rec.get("speaker_verified")
    if verdict is None:
        return not rec.get("voiceprint_enrolled")
    return verdict is True


def inquiry_outcome(
    rec: dict[str, Any],
    *,
    decision_level: str = "PENDING",
    resolved: bool = False,
    button_ok: bool = False,
    escalate_on_no_cancel: bool = True,
    min_asr_logprob: float = DEFAULT_MIN_ASR_LOGPROB,
) -> dict[str, Any]:
    """What a voice-inquiry result means for the event. Mirror of ``EventService._voice_outcome`` with the database lookup replaced by
    ``button_ok`` (the older adult pressed "I'm OK" after the question was asked).

    ``rec`` is a reply record (``classify_reply`` returns one) plus ``prompt_played`` (False: the question never reached anybody).
    Returns ``state`` (cancelled | no_cancel | not_run | audio_unavailable), ``cancelled``, ``by`` (voice | button), ``marker`` (kind of
    the audio alarm marker to add, or None), ``escalate`` (raise PENDING to EMERGENCY), ``notify`` (tell the caregivers about a call for
    help on an event that is already EMERGENCY) and ``call`` (a call for help that no open event can take: an alarm of its own)."""
    none = {"cancelled": False, "by": None, "marker": None, "escalate": False, "notify": False, "call": False}
    status = rec.get("status")
    if status in ("not_run", "audio_unavailable"):
        return {"state": str(status), **none}
    if status not in INQUIRY_JUDGED:  # a record that does not say how the person answered is no evidence either way
        return {"state": "not_run", **none}
    spoken = rec.get("intent") == "HELP"
    if rec.get("prompt_played") is False and not spoken:
        # the question never reached anybody: what the room did afterwards answers nothing; only a spoken call for help is evidence
        return {"state": "not_run", **none}
    by = None
    if not spoken:  # a call for help is not cancelled by the button: the safe direction wins
        if voice_cancel_verdict(rec, min_asr_logprob):
            by = "voice"
        elif button_ok:
            by = "button"
    live = not resolved
    if by is not None:
        return {**none, "state": "cancelled", "cancelled": live, "by": by}
    if not live:
        return {**none, "state": "no_cancel", "call": spoken}
    raised = spoken or escalate_on_no_cancel
    return {**none, "state": "no_cancel", "marker": "spoken_help" if spoken else "inquiry_no_cancel",
            "escalate": raised and decision_level != "EMERGENCY", "notify": spoken and decision_level == "EMERGENCY"}


_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def alarm_tag(markers: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Tag of an event from its alarm markers. ``markers`` are dicts ``{"channel": accel|vision|audio|manual, "at_s": seconds, "kind": str}``
    (``at_s`` relative to any common zero) or the node's own marker dicts with an ISO ``at``.

    Returns ``tag`` (single_channel | dual_channel | triple_channel | None), the ``channels`` that were counted, and the normalised
    ``markers`` with their ``counted`` flags. The window is fixed at 10 s from the earliest counted marker."""
    store: list[dict[str, Any]] = []
    for m in markers:
        at = m["at"] if "at" in m else _BASE + timedelta(seconds=float(m["at_s"]))
        store = alarm_tags.add_marker(store, str(m["channel"]), at, str(m.get("kind") or m["channel"]))
    return {"tag": alarm_tags.compute_tag(store), "channels": alarm_tags.counted_channels(store), "markers": store,
            "window_s": alarm_tags.WINDOW_S}
