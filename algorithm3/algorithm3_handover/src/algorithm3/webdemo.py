"""Local demo page for Algorithm 3: FastAPI + one static HTML page. Everything runs on this machine; audio is processed in memory and
nothing is stored. Start it with ``start.bat``, or ``python -m algorithm3.webdemo``."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from homeease.core.paths import project_root
from homeease.ml.audio.features import SR, load_audio

from algorithm3 import policy
from algorithm3.api import Algorithm3
from algorithm3.decode import decode_bytes

ROOT = project_root()
SAMPLES = ROOT / "samples"
STATIC = Path(__file__).with_name("static")
MAX_UPLOAD = 25 * 1024 * 1024


class Marker(BaseModel):
    channel: str = Field(pattern="^(accel|vision|audio|manual)$")
    at_s: float = Field(ge=-3600, le=3600)
    kind: str = "candidate"


class Reply(BaseModel):
    status: str = "RESPONDED"
    intent: str | None = None
    asr_avg_logprob: float | None = None
    speaker_verified: bool | None = None
    voiceprint_enrolled: bool = False
    prompt_played: bool = True


class AlarmRequest(BaseModel):
    markers: list[Marker] = []
    reply: Reply = Reply()
    question_at_s: float = 1.0
    level: str = Field(default="PENDING", pattern="^(PENDING|EMERGENCY)$")
    button_ok: bool = False
    escalate_on_no_cancel: bool = True
    resolved: bool = False


def create_app(vad: str = "auto") -> FastAPI:
    app = FastAPI(title="Algorithm 3 demo", docs_url="/docs")
    a3 = Algorithm3(vad=vad)
    index = json.loads((SAMPLES / "index.json").read_text(encoding="utf-8")) if (SAMPLES / "index.json").is_file() else {"scenes": [], "replies": []}
    known = {s["id"]: s["file"] for s in index["scenes"] + index["replies"]}

    def audio_from(sample: str | None, file: UploadFile | None) -> np.ndarray:
        if file is not None:
            data = file.file.read(MAX_UPLOAD + 1)
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "file too large (limit 25 MB)")
            try:
                return decode_bytes(data)
            except Exception as e:
                raise HTTPException(400, f"cannot read this audio file: {type(e).__name__}") from e
        if sample:
            if sample not in known:
                raise HTTPException(404, "unknown sample")
            return load_audio(str(SAMPLES / known[sample]), SR)
        raise HTTPException(400, "give a sample or a file")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {**a3.info(), "samples": index}

    @app.post("/api/warmup")
    def warmup() -> dict[str, Any]:
        return a3.warm_up()

    @app.post("/api/sounds")
    def sounds(sample: str | None = Form(None), file: UploadFile | None = File(None)) -> dict[str, Any]:
        x = audio_from(sample, file)
        try:
            return a3.label_sounds(x, SR)
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e

    @app.post("/api/reply")
    def reply(sample: str | None = Form(None), text: str | None = Form(None), file: UploadFile | None = File(None)) -> dict[str, Any]:
        if text is not None and text.strip():
            return a3.classify_reply(text=text.strip()[:300])
        x = audio_from(sample, file)
        return a3.classify_reply(audio=x, sr=SR)

    @app.post("/api/help")
    def help_call(sample: str | None = Form(None), file: UploadFile | None = File(None)) -> dict[str, Any]:
        x = audio_from(sample, file)
        try:
            return a3.detect_spoken_help(x, SR)
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e

    @app.post("/api/alarm")
    def alarm(req: AlarmRequest) -> dict[str, Any]:
        rec = req.reply.model_dump()
        out = policy.inquiry_outcome(rec, decision_level=req.level, resolved=req.resolved, button_ok=req.button_ok,
                                     escalate_on_no_cancel=req.escalate_on_no_cancel)
        markers = [m.model_dump() for m in req.markers]
        if out["marker"]:  # the audio marker is stamped when the question was asked
            markers.append({"channel": "audio", "at_s": req.question_at_s, "kind": out["marker"]})
        tagged = policy.alarm_tag(markers)
        final = "EMERGENCY" if (out["escalate"] or req.level == "EMERGENCY") else req.level
        return {"outcome": out, "level": req.level, "final_level": final, **tagged}

    app.mount("/samples", StaticFiles(directory=str(SAMPLES)), name="samples")
    return app


def _free_port(start: int) -> int:
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Algorithm 3 local demo page")
    ap.add_argument("--port", type=int, default=8730)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--vad", default="auto", choices=["auto", "silero", "energy"])
    args = ap.parse_args()
    port = _free_port(args.port)
    url = f"http://127.0.0.1:{port}/"
    if not args.no_browser and not os.environ.get("ALGORITHM3_NO_BROWSER"):
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    print(f"Algorithm 3 demo: {url}   (close this window or press Ctrl+C to stop)")
    uvicorn.run(create_app(args.vad), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
