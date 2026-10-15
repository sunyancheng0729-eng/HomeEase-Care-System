"""Exercise the real FastAPI routes and original models in an isolated database."""
import base64
import json
import os
import sys
import time
import psutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runtime" / ("acceptance-" + time.strftime("%Y%m%d-%H%M%S"))
os.environ["HOMEEASE_SYSTEM_DATA"] = str(DATA)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from care import main


def checked(response):
    assert response.status_code < 400, (response.status_code, response.text[:500])
    return response.json()


if __name__ == "__main__":
    results = {}
    with TestClient(main.app) as client:
        deadline = time.time() + 90
        while time.time() < deadline:
            state = checked(client.get("/api/health"))
            if all(m["state"] == "ready" for m in state["modules"]):
                break
            time.sleep(.3)
        else:
            raise AssertionError(state)
        results["models"] = {m["id"]: m["details"].get("model") or m["details"].get("model_id") or m["details"].get("sound_model") for m in state["modules"]}
        profile = checked(client.get("/api/settings"))
        profile.update(audio_consent=True)
        profile["resident"].update(name="集成验收", room="测试空间")
        profile["contacts"] = [{"name": "测试联系人", "relation": "测试", "phone": "000000000"}]
        checked(client.put("/api/settings", json=profile))
        notice = checked(client.post("/api/notifications/test"))
        notices = checked(client.get("/api/notifications"))
        assert next(n for n in notices if n["id"] == notice["id"])["delivered_at"]
        results["notifications"] = True

        signal = checked(client.post("/api/signal/sample"))
        assert len(signal["records"]) == 1 and signal["records"][0]["mode"] == "replay"
        eid = signal["records"][0]["id"]
        checked(client.post(f"/api/events/{eid}/inquiry", json={"prompt_played": True}))
        checked(client.post(f"/api/events/{eid}/inquiry/processing"))
        audio = (ROOT / "algorithm3" / "algorithm3_handover" / "samples" / "reply_help_en.wav").read_bytes()
        reply = checked(client.post("/api/audio/analyze", data={"task": "reply", "mode": "replay", "event_id": eid}, files={"file": ("reply.wav", audio, "audio/wav")}))
        assert reply["event"]["level"] == "EMERGENCY"
        assert reply["event"]["state"] == "open"
        assert "transcript" not in reply["result"] and "transcript" not in reply["event"]["inquiry"]["record"]
        resolved = checked(client.post(f"/api/events/{eid}/action", json={"action": "resolve", "outcome": "confirmed_fall", "note": "自动验收中的原始回放"}))
        assert resolved["response_s"] >= 0
        assert client.get(f"/reports/events/{eid}").status_code == 200
        results["signal_and_voice_workflow"] = {"windows": signal["info"]["n_windows"], "events": 1, "intent": reply["result"]["intent"], "level": reply["event"]["level"], "tag": reply["event"]["tag"]}

        exercise_counts = {}
        originals = json.loads((ROOT / "algorithm2" / "reports" / "replay_sessions.json").read_text())
        references = {s["exercise"]: s["count"] for s in originals}
        for exercise in references:
            frames = json.loads((ROOT / "algorithm2" / "examples" / f"{exercise}_replay.json").read_text())["frames"]
            created = main.system.create_session(exercise, 10, "replay")
            sid = created["session_id"]
            first = {"request_id": "batch-0", "frames": frames[:100]}
            accepted = checked(client.post(f"/api/sessions/{sid}/frames", json=first))
            retried = checked(client.post(f"/api/sessions/{sid}/frames", json=first))
            assert accepted["count"] == retried["count"]
            conflict = {**first, "frames": frames[1:101]}
            assert client.post(f"/api/sessions/{sid}/frames", json=conflict).status_code == 409
            for index in range(100, len(frames), 100):
                checked(client.post(f"/api/sessions/{sid}/frames", json={"request_id": f"batch-{index}", "frames": frames[index:index+100]}))
            ended = checked(client.post(f"/api/sessions/{sid}/control", json={"action": "end"}))
            assert ended["count"] == references[exercise]
            assert checked(client.get(f"/api/sessions/{sid}/export.json"))["count"] == ended["count"]
            exercise_counts[exercise] = ended["count"]
        results["four_exercise_http_counts"] = exercise_counts

        labels = checked(client.post("/api/audio/sample", json={"sample": "scene_quiet_home", "task": "sounds"}))
        help_result = checked(client.post("/api/audio/sample", json={"sample": "reply_help_en", "task": "help"}))
        assert help_result["result"]["help"]
        assert not any(name.endswith(".wav") or name.endswith(".webm") for name in os.listdir(DATA / "uploads"))
        results["audio"] = {"sound_events": len(labels["result"]["events"]), "spoken_help": True, "raw_audio_stored": False}

        with client.websocket_connect("/ws/audio") as ws:
            assert ws.receive_json()["state"] == "listening"
            import numpy as np
            pcm = base64.b64encode(np.zeros(8000, dtype="<f4").tobytes()).decode()
            for index in range(4):
                ws.send_json({"pcm": pcm, "timestamp_ms": index * 500.0})
                result = ws.receive_json()
                assert "pipeline" in result, result
            ws.send_json({"type": "stop"})
            assert ws.receive_json()["state"] == "stopped"
        assert not main.system.audio_lock.locked()
        results["continuous_microphone_pipeline"] = True

        data = np.loadtxt(ROOT / "samples" / "fall-01-acc.csv", delimiter=",")
        samples = [[float((r[0]-data[0,0])/1000), *[float(v) for v in r[2:5]]] for r in data]
        with client.websocket_connect("/ws/sensor") as ws:
            assert ws.receive_json()["state"] == "connected"
            for index in range(0,len(samples),50):
                ws.send_json({"samples":samples[index:index+50]})
                result = ws.receive_json()
                assert "windows" in result, result
            ws.send_json({"type":"flush"})
            assert ws.receive_json()["state"] == "stopped"
        live_sensor_events = main.system.store.events(mode="live")
        assert any(e["channel"] == "accel" for e in live_sensor_events)
        results["sensor_websocket"] = True

        checked(client.post("/api/video/start", json={"source": "sample"}))
        deadline = time.time()+60
        while time.time()<deadline:
            video = checked(client.get("/api/status"))["video"]
            if video["state"] != "running":
                break
            time.sleep(.3)
        else:
            raise AssertionError("Video never finished")
        main.system.poll_video(force=True)
        video_events = [e for e in main.system.store.events(mode="replay") if e["channel"] == "vision"]
        assert video["frames"] >= 30
        assert video_events, video
        clip = client.get(f"/api/video/clip/{video_events[0]['id']}")
        assert clip.status_code == 200 and len(clip.content)>1000
        frame = client.get("/api/video/frame")
        assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"
        results["video"] = {"frames": video["frames"], "peak": video["peak"], "events": len(video_events), "playable_clip_bytes":len(clip.content)}

        report = checked(client.get("/api/reports?mode=replay"))
        assert report["summary"]["sessions"] >= 4
        assert client.get("/reports/print?mode=replay").status_code == 200
        assert client.get("/api/reports/export.csv?mode=replay").status_code == 200
        results["reports"] = True
        original_pid = main.system.modules["signal"].process.pid
        original_worker = main.system.modules["signal"].worker_process
        main.system.modules["signal"].process.kill()
        main.system.modules["signal"].process.wait()
        assert client.get("/api/reports").status_code == 200
        checked(client.post("/api/signal/sample"))
        assert main.system.modules["signal"].process.pid != original_pid
        assert not original_worker.is_running()
        results["isolated_failure_and_recovery"] = True
        profile["audio_consent"] = False
        checked(client.put("/api/settings", json=profile))
        assert client.post("/api/audio/sample", json={"sample":"reply_help_en","task":"help"}).status_code == 409
        results["consent_revocation"] = True
        owned_workers = [m.worker_process for m in main.system.modules.values() if m.worker_process]
    assert all(not p.is_running() for p in owned_workers)
    results["shutdown"] = "all owned worker processes released"
    (ROOT / "artifacts" / "acceptance.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))
