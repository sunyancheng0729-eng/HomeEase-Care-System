"""Actual PCM recording through the live WebSocket, on the isolated server."""
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8011"


async def main():
    with httpx.Client(base_url=URL, timeout=120) as client:
        settings = client.get("/api/settings").json()
        settings["audio_consent"] = True
        client.put("/api/settings", json=settings).raise_for_status()
        before = {e["id"] for e in client.get("/api/events").json()}
        samples, rate = sf.read(ROOT / "algorithm3" / "algorithm3_handover" / "samples" / "scene_fall_help_call.wav", dtype="float32")
        assert rate == 16000 and samples.ndim == 1
        samples = np.concatenate([samples, np.zeros(16000 * 5, dtype=np.float32)])
        calls, sound_ids, latencies = [], set(), []
        async with websockets.connect(URL.replace("http", "ws") + "/ws/audio", origin=URL) as ws:
            assert json.loads(await ws.recv())["state"] == "listening"
            for index in range(0, len(samples), 8000):
                chunk = samples[index:index + 8000]
                started = time.perf_counter()
                await ws.send(json.dumps({"pcm": base64.b64encode(chunk.astype("<f4").tobytes()).decode(), "timestamp_ms": index / 16}))
                result = json.loads(await ws.recv())
                assert "error" not in result, result
                calls.extend(result["calls"])
                sound_ids.update(e["event_id"] for e in result["events"])
                latencies.append(time.perf_counter() - started)
                assert result["pipeline"]["stream"]["discontinuities"] == 0, result["pipeline"]
                await asyncio.sleep(max(0, len(chunk) / rate - (time.perf_counter() - started)))
            await ws.send(json.dumps({"type": "stop"}))
            assert json.loads(await ws.recv())["state"] == "stopped"
        events = [e for e in client.get("/api/events").json() if e["id"] not in before and e["channel"] == "audio"]
        assert calls and events, {"calls": calls, "pipeline": result["pipeline"], "listener": result["listener"]}
        assert all(e["level"] == "EMERGENCY" and e["state"] == "open" for e in events)
        assert all("transcript" not in json.dumps(e) for e in events)
        summary = {"sample": "scene_fall_help_call.wav", "seconds": len(samples) / rate,
                   "chunks": len(latencies), "discontinuities": 0, "calls": len(calls),
                   "live_emergency_events": len(events), "sound_events": len(sound_ids),
                   "max_request_ms": round(max(latencies) * 1000, 1), "raw_audio_stored": False}
        (ROOT / "artifacts" / "live-audio.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
