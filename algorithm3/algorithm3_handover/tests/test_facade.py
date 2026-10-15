"""The facade and the demo page with the bundled samples (real models: a few seconds each)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithm3 import Algorithm3
from homeease.core.paths import models_dir, project_root
from homeease.ml.audio.features import load_audio

SAMPLES = project_root() / "samples"
HAVE_MODELS = (models_dir() / "audio" / "faster-whisper-tiny" / "model.bin").is_file() and (models_dir() / "audio" / "yamnet" / "yamnet.onnx").is_file()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="weights are not in models/")


@pytest.fixture(scope="module")
def a3() -> Algorithm3:
    return Algorithm3()


def test_text_replies_need_no_model(a3):
    assert a3.classify_reply(text="I'm fine, don't worry")["intent"] == "REASSURANCE"
    assert a3.classify_reply(text="救命")["intent"] == "HELP"          # Chinese "help!"
    assert a3.classify_reply(text="I'm fine but I can't get up")["intent"] == "HELP"  # a help cue wins over a reassurance cue
    assert a3.classify_reply(text="")["intent"] == "UNCERTAIN"
    r = a3.classify_reply(text="I'm fine, don't worry")
    assert r["trusted_reassurance"] is True and r["cancel_eligible"] is False  # no enrolment: evidence only


@needs_models
def test_silence_is_no_response(a3):
    rec = a3.classify_reply(audio=load_audio(str(SAMPLES / "reply_silence.wav")), sr=16000)
    assert rec["status"] == "NO_RESPONSE" and rec["trusted_reassurance"] is False


@needs_models
def test_spoken_help_alarm_and_the_things_that_must_not_alarm(a3):
    def verdicts(name: str) -> list[str]:
        return [s["verdict"] for s in a3.detect_spoken_help(load_audio(str(SAMPLES / name)), 16000)["segments"]]

    assert "help" in verdicts("reply_help_en.wav")
    assert "help" in verdicts("reply_help_zh.wav")
    assert "help" not in verdicts("reply_tv_long.wav")        # a long broadcast sentence
    assert "help" not in verdicts("reply_thanks_en.wav")      # "thank you for your help"
    assert "help" not in verdicts("reply_fine_en.wav")


@needs_models
def test_scene_labels_carry_no_audio(a3):
    r = a3.label_sounds(load_audio(str(SAMPLES / "scene_glass_break.wav")), 16000)
    assert r["raw_audio_stored"] is False and r["duration_s"] > 19
    assert all(set(e) >= {"category", "confidence", "start_s", "end_s"} for e in r["events"])
    assert {e["category"] for e in r["events"]} <= {"HELP", "IMPACT", "ABNORMAL", "DAILY", "UNKNOWN"}
    assert "samples" not in json.dumps(r)


@needs_models
def test_the_demo_page_endpoints():
    from fastapi.testclient import TestClient

    from algorithm3.webdemo import create_app

    c = TestClient(create_app())
    assert c.get("/").status_code == 200
    info = c.get("/api/info").json()
    assert len(info["samples"]["scenes"]) == 6 and len(info["samples"]["replies"]) >= 9
    r = c.post("/api/reply", data={"text": "help me"}).json()
    assert r["intent"] == "HELP"
    r = c.post("/api/alarm", json={"markers": [{"channel": "vision", "at_s": 0}], "reply": {"status": "NO_RESPONSE"}}).json()
    assert r["final_level"] == "EMERGENCY" and r["tag"] == "dual_channel"
    assert c.post("/api/reply", data={"sample": "../../etc/passwd"}).status_code == 404
