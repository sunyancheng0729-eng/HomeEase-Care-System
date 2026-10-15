import sqlite3
import json
from datetime import datetime, timezone

import pytest

from care.events import Events
from care.storage import Store
from scripts.clear_records import clear_history
from care.workers import protocol_default


def test_original_audio_timestamps_can_cross_worker_protocol():
    stamp = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
    payload = {"events": [{"started_at": stamp}], "calls": [{"at": stamp}]}
    result = json.loads(json.dumps(payload, default=protocol_default, allow_nan=False))
    assert result["events"][0]["started_at"] == "2026-10-15T12:00:00+00:00"
    assert result["calls"][0]["at"] == stamp.isoformat()
    with pytest.raises(TypeError):
        protocol_default(object())


def test_clear_history_removes_falls_sessions_and_evidence_only(tmp_path):
    store = Store(tmp_path / "homeease.sqlite3")
    settings = store.settings()
    service = Events(store)
    for channel in ("vision", "accel", "audio"):
        service.create(channel, {}, "replay")
    store.save_session({"session_id": "training", "created_at": "2026-10-15", "mode": "live", "status": "ended"})
    store.save_sound({"task": "sounds", "result": {}}, "replay")
    records = tmp_path / "video_bridge" / "systems" / "records"
    (records / "clips").mkdir(parents=True)
    (records / "clips" / "fall.mp4").write_bytes(b"evidence")
    (tmp_path / "playback").mkdir()
    (tmp_path / "playback" / "fall.mp4").write_bytes(b"playback")
    with sqlite3.connect(records / "falls.db") as conn:
        conn.execute("CREATE TABLE falls(id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO falls VALUES (1)")
    result = clear_history(tmp_path)
    assert result == {"notifications": 2, "fall_events": 2, "sessions": 1, "engine_falls": 1, "evidence_files": 2}
    assert store.sessions() == []
    assert [e["channel"] for e in store.events()] == ["audio"]
    assert len(store.notifications()) == len(store.sounds()) == 1
    assert store.settings() == settings
    assert not list((records / "clips").iterdir())
    assert not list((tmp_path / "playback").iterdir())
    assert clear_history(tmp_path)["fall_events"] == 0
