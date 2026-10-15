"""Real desktop camera/concurrent sensor acceptance, on an isolated server."""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
assert URL.endswith(":8011"), "Use the isolated acceptance server on port 8011"


async def main():
    result = {}
    sid = None
    with httpx.Client(base_url=URL, timeout=120) as client:
        def checked(response):
            response.raise_for_status()
            return response.json()
        try:
            if checked(client.get("/api/status"))["video"]["state"] != "running":
                checked(client.post("/api/video/start", json={"source": "camera"}))
            fps = []
            observations = {}
            for _ in range(10):
                await asyncio.sleep(.5)
                fps.append(checked(client.get("/api/status"))["video"]["fps"])
            result["video_alone_fps"] = {"median": statistics.median(fps), "samples": fps}
            created = checked(client.post("/api/sessions", json={"exercise": "chair_sit_to_stand", "source": "camera", "target_reps": 10}))
            sid = created["session_id"]
            assert created["shared_camera"]
            samples = []
            for line in (ROOT / "samples" / "fall-01-acc.csv").read_text().splitlines():
                try:
                    values = list(map(float, line.split(",")))
                except ValueError:
                    continue
                samples.append([values[0] / 1000, *values[2:5]])
            fps = []
            async with websockets.connect(URL.replace("http", "ws") + "/ws/sensor", origin=URL) as sensor:
                assert json.loads(await sensor.recv())["state"] == "connected"
                for index in range(0, min(500, len(samples)), 10):
                    await sensor.send(json.dumps({"samples": samples[index:index + 10]}))
                    batch = json.loads(await sensor.recv())
                    assert "error" not in batch, batch
                    await asyncio.sleep(.2)
                    status = checked(client.get("/api/status"))
                    training = checked(client.get(f"/api/sessions/{sid}"))
                    diagnostics = training.get("capture_diagnostics", {})
                    if diagnostics.get("timestamp_ms") is not None:
                        observations[diagnostics["timestamp_ms"]] = diagnostics
                    fps.append(status["video"]["fps"])
                    assert status["sensor"]["state"] == "connected"
                    assert len(status["cameras"]) == 1
                    assert status["cameras"][0]["users"] == sorted(["video", sid])
                result["joint_video_sensor"] = {"windows": batch["windows"], "fps_median": statistics.median(fps), "fps_samples": fps}
                await sensor.send(json.dumps({"type": "flush"}))
                assert json.loads(await sensor.recv())["state"] == "stopped"
            state = checked(client.get(f"/api/sessions/{sid}"))
            assert state["status"] == "running", state
            result["shared_exercise"] = {"status": state["status"], "quality": state["quality"], "elapsed_s": state["elapsed_s"], "frame_shape": state["frame_shape"]}
            diagnostics = list(observations.values())
            assert diagnostics, state
            result["shared_exercise"]["runtime"] = {
                "observation_fps_median": statistics.median(d["observation_fps"] for d in diagnostics),
                "pose_ms_median": statistics.median(d["pose_ms"] for d in diagnostics),
                "processing_ms_median": statistics.median(d["processing_ms"] for d in diagnostics),
                "max_sampled_gap_ms": max(d.get("gap_ms") or 0 for d in diagnostics),
                "timestamp_gap_interruptions": sum(i["reason"] == "timestamp_gap" for i in state["interruptions"])}
            checked(client.post(f"/api/sessions/{sid}/control", json={"action": "pause"}))
            await asyncio.sleep(.4)
            paused = checked(client.get(f"/api/sessions/{sid}"))
            await asyncio.sleep(.4)
            assert checked(client.get(f"/api/sessions/{sid}"))["updated_at"] == paused["updated_at"]
            checked(client.post(f"/api/sessions/{sid}/control", json={"action": "resume"}))
            checked(client.post("/api/video/stop"))
            await asyncio.sleep(1)
            assert checked(client.get("/api/status"))["cameras"][0]["users"] == [sid]
            assert checked(client.get(f"/api/sessions/{sid}"))["updated_at"] != paused["updated_at"]
            checked(client.post("/api/video/start", json={"source": "camera"}))
            checked(client.post(f"/api/sessions/{sid}/control", json={"action": "end"}))
            assert checked(client.get("/api/status"))["cameras"][0]["users"] == ["video"]
            result["independent_stop_restart_pause_resume"] = True
            (ROOT / "artifacts" / "camera-runtime.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, ensure_ascii=True))
        finally:
            if sid:
                client.post(f"/api/sessions/{sid}/control", json={"action": "end"})
            client.post("/api/video/stop")


if __name__ == "__main__":
    asyncio.run(main())
