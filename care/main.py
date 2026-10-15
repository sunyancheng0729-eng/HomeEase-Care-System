import asyncio
import base64
import binascii
import csv
import hashlib
import hmac
import io
import json
import logging
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from care import schemas
from care.adapters.process import ModuleError
from care.config import ROOT, RUNTIME, UPLOADS, LOGS, VIDEO, SIGNAL, AUDIO, EXERCISE, EXERCISES, prepare
from care.services import System, canonical_csv
from care.storage import now

prepare()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", handlers=[
    logging.FileHandler(LOGS / "system.log", encoding="utf-8"), logging.StreamHandler()])
system = None


@asynccontextmanager
async def lifespan(app):
    global system
    system = System()
    system.start()
    yield
    await asyncio.to_thread(system.close)


app = FastAPI(title="HomeEase 居家照护", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "web" / "templates")


def check_access(headers, client_host):
    password = os.environ.get("HOMEEASE_PASSWORD")
    if password:
        value = headers.get("authorization", "")
        try:
            decoded = base64.b64decode(value.removeprefix("Basic ")).decode()
            username, provided = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return False
        return hmac.compare_digest(provided.encode(), password.encode()) and username == "caregiver"
    return client_host in {"127.0.0.1", "::1", "testclient"}


@app.middleware("http")
async def guard(request, call_next):
    if not check_access(request.headers, request.client.host if request.client else ""):
        return Response("需要身份验证", status_code=401, headers={"WWW-Authenticate": 'Basic realm="HomeEase", charset="UTF-8"'})
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"error": "请求来源不匹配"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ModuleError)
async def module_error(request, exc):
    return JSONResponse({"error": str(exc), "code": "MODULE_ERROR"}, status_code=exc.status)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"error": str(exc)}, status_code=409)


@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse({"error": "记录不存在"}, status_code=404)


@app.exception_handler(Exception)
async def unexpected(request, exc):
    logging.getLogger(__name__).exception("Request failed: %s", request.url.path)
    return JSONResponse({"error": "操作失败，详情已记录。请在设置页检查模块状态。"}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/elder", response_class=HTMLResponse)
def elder(request: Request):
    return templates.TemplateResponse(request=request, name="elder.html", context={})


@app.get("/sensor", response_class=HTMLResponse)
def sensor_page(request: Request):
    return templates.TemplateResponse(request=request, name="sensor.html", context={})


@app.get("/api/health")
def health():
    return {"status": "running", "storage": "sqlite", "modules": [m.health() for m in system.modules.values()]}


@app.get("/api/status")
def status():
    return system.status()


@app.get("/api/settings")
def settings():
    return system.store.settings()


@app.put("/api/settings")
def update_settings(body: schemas.Settings):
    old = system.store.settings()
    value = body.model_dump()
    value["notification_test_at"] = old.get("notification_test_at")
    saved = system.store.save_settings(value)
    if old["audio_consent"] and not value["audio_consent"]:
        system.audio_last_received = None
        if system.modules["audio"].state == "ready":
            system.modules["audio"].call("live_stop", timeout=15)
        for event in system.store.events(state="open"):
            if event["inquiry"].get("status") in {"listening", "processing"}:
                system.events.inquiry_reply(event["id"], {"status": "audio_unavailable"})
    return saved


@app.post("/api/modules/{name}/initialize", status_code=202)
def initialize(name: str):
    if name not in system.modules:
        raise HTTPException(404, "模块不存在")
    if system.modules[name].state == "loading":
        return {"state": "loading"}
    if name == "video" and system.video.get("state") == "running":
        raise ValueError("请先停止视频检测")
    if name == "signal" and system.sensor["state"] == "connected":
        raise ValueError("请先断开传感器")
    if name == "exercise" and any(meta["active"] for meta in system.session_metadata.values()):
        raise ValueError("请先结束运动训练")
    if name == "audio" and system.audio_lock.locked():
        raise ValueError("请先停止麦克风监听")
    system.modules[name].close()
    system.modules[name].state = "loading"
    system.submit(system.initialize, name)
    return {"state": "loading"}


@app.get("/api/diagnostics")
def diagnostics():
    return {"logs": system.store.diagnostics(), "modules": [m.health() for m in system.modules.values()]}


@app.post("/api/notifications/test")
def test_notice():
    settings = system.store.settings()
    notice = system.store.notify(None, {"title": "照护通知测试", "message": "站内通知测试", "mode": "test", "contact": None})
    if settings.get("webhook_url"):
        system.store.notify(None, {"title": "HomeEase 通知测试", "message": "通知通道测试", "mode": "test"}, kind="webhook")
    settings["notification_test_at"] = now()
    system.store.save_settings(settings)
    return notice


@app.get("/api/notifications")
def notices():
    return system.store.notifications(deliver=True)


@app.post("/api/notifications/{notice_id}/seen")
def seen(notice_id: str):
    system.store.acknowledge_notification(notice_id)
    return {"status": "seen"}


async def read_upload(file, maximum):
    data = bytearray()
    try:
        while chunk := await file.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > maximum:
                raise HTTPException(413, f"文件超过 {maximum // 1024 // 1024} MB")
        if not data:
            raise HTTPException(422, "文件为空")
        return bytes(data)
    finally:
        await file.close()


@app.post("/api/uploads/video")
async def upload_video(file: UploadFile = File()):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(422, "请选择 MP4、AVI、MOV、MKV 或 WebM 视频")
    upload_id = uuid.uuid4().hex + suffix
    path = UPLOADS / upload_id
    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise HTTPException(413, "视频不能超过 512 MB")
                output.write(chunk)
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            if not ok:
                raise HTTPException(422, "视频没有可读取的画面")
        finally:
            cap.release()
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"upload_id": upload_id, "filename": file.filename, "bytes": size}


def uploaded(name):
    if not name or Path(name).name != name:
        raise ValueError("请先上传视频")
    path = UPLOADS / name
    if not path.is_file():
        raise ValueError("上传文件不存在，请重新上传")
    return path


@app.post("/api/video/start")
def start_video(body: schemas.VideoStart):
    if body.source == "camera":
        source, mode = {"type": "camera", "index": body.camera_index}, "live"
    elif body.source == "sample":
        source, mode = {"type": "file", "path": str(ROOT / "samples" / "fall-01-cam0.mp4")}, "replay"
    else:
        source, mode = {"type": "file", "path": str(uploaded(body.upload_id))}, "replay"
    return system.video_start(source, mode)


@app.post("/api/video/stop")
def stop_video():
    return system.video_stop()


@app.get("/api/video/frame")
def video_frame():
    if system.modules["video"].state != "ready":
        raise HTTPException(503, "视频模块尚未就绪")
    data = base64.b64decode(system.modules["video"].call("frame", timeout=10))
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/video/stream")
async def video_stream(request: Request):
    if system.modules["video"].state != "ready":
        raise HTTPException(503, "视频模块尚未就绪")

    async def frames():
        last = None
        while not await request.is_disconnected():
            if system.video.get("state") not in {"running", "idle", "finished"}:
                break
            try:
                data = base64.b64decode(await asyncio.to_thread(system.modules["video"].call, "frame", timeout=10))
            except ModuleError:
                break
            if data and data != last:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                last = data
            await asyncio.sleep(1 / 15)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/video/clip/{event_id}")
def video_clip(event_id: str):
    event = system.store.event(event_id)
    clip = event["evidence"].get("clip")
    if not clip:
        raise HTTPException(404, "该事件没有视频片段")
    path = RUNTIME / "video_bridge" / "systems" / "records" / "clips" / Path(clip).name
    if not path.is_file():
        raise HTTPException(404, "视频片段尚未保存或已移除")
    from care.media import browser_clip
    return FileResponse(browser_clip(path), media_type="video/mp4")


@app.post("/api/signal/sample")
def signal_sample():
    return system.signal_offline(ROOT / "samples" / "fall-01-acc.csv", "URFD fall-01 加速度样例")


@app.post("/api/signal/analyze")
async def signal_upload(file: UploadFile = File()):
    data = await read_upload(file, 20 * 1024 * 1024)
    canonical = canonical_csv(data)
    path = UPLOADS / (uuid.uuid4().hex + ".csv")
    try:
        path.write_bytes(canonical)
        return await asyncio.to_thread(system.signal_offline, path, file.filename or "上传 CSV")
    finally:
        path.unlink(missing_ok=True)


def websocket_allowed(ws):
    origin = ws.headers.get("origin")
    return check_access(ws.headers, ws.client.host if ws.client else "") and (not origin or urlparse(origin).netloc == ws.headers.get("host"))


@app.websocket("/ws/sensor")
async def sensor_stream(ws: WebSocket):
    if not websocket_allowed(ws):
        await ws.close(1008)
        return
    await ws.accept()
    if not system.sensor_lock.acquire(blocking=False):
        await ws.send_json({"error": "已有传感器连接"})
        await ws.close(1008)
        return
    stream_id = uuid.uuid4().hex
    last_timestamp = None
    initialized = False
    stop_requested = False
    try:
        await asyncio.to_thread(system.modules["signal"].call, "stream_start")
        initialized = True
        system.sensor = {"state": "connected", "last_received": None, "stream_id": stream_id}
        await ws.send_json({"state": "connected", "units": "seconds,g,g,g"})
        while True:
            message = await asyncio.wait_for(ws.receive_json(), timeout=15)
            if message.get("type") == "flush":
                stop_requested = True
                break
            samples = message.get("samples")
            if not isinstance(samples, list) or not 1 <= len(samples) <= 200:
                raise ValueError("每批需要 1 至 200 个 [t,ax,ay,az] 样本")
            for sample in samples:
                if not isinstance(sample, list) or len(sample) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in sample):
                    raise ValueError("样本必须包含 4 个有限数值")
                if last_timestamp is not None and sample[0] <= last_timestamp:
                    raise ValueError("传感器时间戳必须严格递增")
                last_timestamp = sample[0]
            result = await asyncio.to_thread(system.modules["signal"].call, "samples", samples=samples)
            system.signal_results(result, stream_id)
            history = (system.sensor.get("samples", []) + samples)[-300:]
            system.sensor.update(last_received=time.time(), prob=result["prob"], windows=result["windows"],
                                 recent=result.get("recent", []), samples=history)
            await ws.send_json(result)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as exc:
        try:
            await ws.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        if initialized:
            try:
                result = await asyncio.to_thread(system.modules["signal"].call, "flush", timeout=15)
                system.signal_results(result, stream_id)
            except Exception:
                logging.exception("Sensor flush failed")
        system.sensor = {"state": "disconnected", "last_received": None}
        system.sensor_lock.release()
        try:
            if stop_requested:
                await ws.send_json({"state": "stopped"})
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.get("/api/exercises")
def exercises():
    return EXERCISES


@app.post("/api/sessions", status_code=201)
def create_session(body: schemas.SessionStart):
    path = uploaded(body.upload_id) if body.source == "upload" else None
    if body.source == "camera" and body.capture == "shared":
        return system.start_camera_session(body.exercise, body.target_reps, body.camera_index)
    if body.source == "camera":
        system.modules["exercise"].call("pose_initialize")
    result = system.create_session(body.exercise, body.target_reps, "live" if body.source == "camera" else "replay")
    sid = result["session_id"]
    if body.source == "replay":
        system.submit(system.replay, sid, body.exercise)
    elif path:
        system.submit(system.video_exercise, sid, path)
    return result


@app.get("/api/sessions")
def sessions(mode: str | None = None):
    return system.store.sessions(mode=mode)


@app.get("/api/sessions/{sid}")
def session(sid: str):
    result = next((s for s in system.store.sessions() if s["session_id"] == sid), None)
    if result is None:
        raise KeyError(sid)
    meta = system.session_metadata.get(sid, {})
    if meta.get("shared_camera"):
        meta["last_viewer"] = time.monotonic()
    return {**result, "keypoints": system.latest_poses.get(sid),
            "shared_camera": system.session_metadata.get(sid, {}).get("shared_camera", False),
            "frame_shape": system.camera_shapes.get(sid, result.get("frame_shape"))}


@app.get("/api/sessions/{sid}/camera/stream")
async def exercise_camera_stream(sid: str, request: Request):
    if not system.cameras.preview(sid):
        raise HTTPException(409, "训练摄像头未连接")

    async def frames():
        last = None
        while not await request.is_disconnected():
            data = system.cameras.preview(sid)
            if data is None:
                break
            if data != last:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                last = data
            await asyncio.sleep(1 / 30)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/sessions/{sid}/control")
def control_session(sid: str, body: schemas.SessionControl):
    return system.session_control(sid, body.action)


@app.post("/api/sessions/{sid}/frames")
async def session_frames(sid: str, request: Request):
    data = await request.body()
    if len(data) > 2_000_000:
        raise HTTPException(413, "骨架批次过大")
    body = json.loads(data)
    return await asyncio.to_thread(system.session_frames, sid, body)


@app.post("/api/sessions/{sid}/image")
async def session_image(sid: str, timestamp_ms: float = Query(ge=0, le=1e15), request_id: str = Query(min_length=1, max_length=128), file: UploadFile = File()):
    data = await read_upload(file, 5 * 1024 * 1024)
    return await asyncio.to_thread(system.session_image, sid, data, timestamp_ms, request_id)


@app.get("/api/sessions/{sid}/export.json")
def export_session(sid: str):
    result = session(sid)
    result.pop("keypoints", None)
    return Response(json.dumps(result, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="exercise-{sid}.json"'})


@app.get("/api/audio/samples")
def audio_samples():
    data = json.loads((AUDIO / "samples" / "index.json").read_text(encoding="utf-8"))
    for group in data.values():
        if isinstance(group, list):
            for item in group:
                for key in ("said", "note"):
                    item.pop(key, None)
    return data


@app.get("/api/audio/samples/{sample}")
def audio_sample_file(sample: str):
    path = AUDIO / "samples" / (sample + ".wav")
    if Path(sample).name != sample or not path.is_file():
        raise HTTPException(404, "音频样例不存在")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/audio/sample")
def analyze_audio_sample(body: schemas.SampleAudio):
    path = AUDIO / "samples" / (body.sample + ".wav")
    if not path.is_file():
        raise HTTPException(404, "音频样例不存在")
    return system.audio_analyze(path.read_bytes(), body.task, "replay")


@app.post("/api/audio/analyze")
async def analyze_audio(task: str = Form(), mode: str = Form(default="replay"), event_id: str | None = Form(default=None), file: UploadFile = File()):
    if task not in {"monitor", "sounds", "help", "reply"} or mode not in {"live", "replay"}:
        raise HTTPException(422, "音频任务或输入来源不合法")
    data = await read_upload(file, 20 * 1024 * 1024)
    return await asyncio.to_thread(system.audio_analyze, data, task, mode, event_id)


@app.get("/api/sounds")
def sounds():
    return system.store.sounds()


@app.websocket("/ws/audio")
async def live_audio(ws: WebSocket):
    if not websocket_allowed(ws) or not system.store.settings()["audio_consent"]:
        await ws.close(1008)
        return
    await ws.accept()
    if not system.audio_lock.acquire(blocking=False):
        await ws.send_json({"error": "已有麦克风输入"})
        await ws.close(1008)
        return
    started = False
    stop_requested = False
    seen_events = {}
    last_time = None

    def persist_result(result):
        for call in result.get("calls", []):
            system.events.create("audio", call.get("details", {}), "live", "连续求救监听",
                f"audio:{call.get('idem', uuid.uuid4().hex)}", at=call.get("at"))
        for sound in result.get("events", []):
            key = sound["event_id"]
            if seen_events.get(key) != sound.get("n_windows"):
                system.store.save_sound({"task": "sounds", "result": {"events": [sound], "raw_audio_stored": False}}, "live")
                seen_events[key] = sound.get("n_windows")
        if len(seen_events) > 250:
            for key in list(seen_events)[:-100]:
                del seen_events[key]

    try:
        await asyncio.to_thread(system.modules["audio"].call, "live_start")
        started = True
        await ws.send_json({"state": "listening"})
        while True:
            message = await asyncio.wait_for(ws.receive_json(), timeout=15)
            if not system.store.settings()["audio_consent"]:
                break
            if message.get("type") == "stop":
                stop_requested = True
                break
            pcm, stamp = message.get("pcm"), message.get("timestamp_ms")
            if not isinstance(pcm, str) or len(pcm) > 175000 or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError("无效的音频块")
            if last_time is not None and stamp <= last_time:
                raise ValueError("音频时间戳必须递增")
            last_time = stamp
            listening = any(e["inquiry"].get("status") in {"listening", "processing"} for e in system.store.events(state="open"))
            result = await asyncio.to_thread(system.modules["audio"].call, "live_chunk", pcm=pcm, timestamp_ms=stamp, inquiry_active=listening)
            if not system.store.settings()["audio_consent"]:
                break
            system.audio_last_received = time.time()
            degraded = any(str(result.get(k, {}).get("status", "")).startswith("degraded") for k in ("pipeline", "listener"))
            system.audio_quality = {"state": "reduced" if degraded else "active", "reason": "原音频流水线覆盖受限" if degraded else "连续麦克风输入有效"}
            persist_result(result)
            await ws.send_json(result)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as exc:
        try:
            await ws.send_json({"error": str(exc)})
        except Exception:
            pass
    finally:
        if started:
            try:
                result = await asyncio.to_thread(system.modules["audio"].call, "live_stop", timeout=15)
                if system.store.settings()["audio_consent"]:
                    persist_result(result)
            except Exception:
                logging.exception("Audio shutdown failed")
        system.audio_last_received = None
        system.audio_lock.release()
        try:
            if stop_requested:
                await ws.send_json({"state": "stopped"})
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.get("/api/events")
def events(mode: str | None = None, state: str | None = None):
    return system.store.events(mode=mode, state=state)


@app.get("/api/events/{event_id}")
def event(event_id: str):
    return system.store.event(event_id)


@app.post("/api/events/{event_id}/action")
def event_action(event_id: str, body: schemas.EventAction):
    return system.events.action(event_id, body.action, body.outcome, body.note, revision=body.revision)


@app.post("/api/events/{event_id}/inquiry")
def inquiry_start(event_id: str, body: schemas.InquiryStart):
    if not system.store.settings()["audio_consent"]:
        raise ValueError("未获得音频处理授权")
    return system.events.inquiry_start(event_id, body.prompt_played, body.reason)


@app.post("/api/events/{event_id}/inquiry/processing")
def inquiry_processing(event_id: str):
    return system.events.inquiry_processing(event_id)


@app.post("/api/events/{event_id}/inquiry/unavailable")
def inquiry_unavailable(event_id: str):
    return system.events.inquiry_reply(event_id, {"status": "audio_unavailable"})


@app.post("/api/help", status_code=201)
def request_help():
    existing = system.store.events(mode="live", state="open", limit=20)
    for old in existing:
        if old["channel"] == "manual" and time.time() - __import__("datetime").datetime.fromisoformat(old["created_at"]).timestamp() < 30:
            return old
    return system.events.create("manual", {"action": "help_button"}, "live", "长者求助按钮")


@app.get("/api/reports")
def reports(days: int = Query(default=7, ge=1, le=90), mode: str = Query(default="live", pattern="^(live|replay)$")):
    return system.report(days, mode)


@app.get("/api/reports/export.csv")
def export_csv(days: int = Query(default=7, ge=1, le=90), mode: str = Query(default="live", pattern="^(live|replay)$")):
    report = system.report(days, mode)
    data = io.StringIO(newline="")
    writer = csv.writer(data)
    writer.writerow(["类型", "记录编号", "时间", "来源", "级别/动作", "处置/次数", "状态"])
    safe = lambda value: "'" + str(value) if str(value).startswith(("=", "+", "-", "@", "\t", "\r")) else value
    for item in report["events"]:
        writer.writerow([safe(v) for v in ["事件", item["id"], item["created_at"], item["mode"], item["level"], item["outcome"] or "", item["state"]]])
    for item in report["sessions"]:
        writer.writerow([safe(v) for v in ["训练", item["session_id"], item["created_at"], item["mode"], item["exercise"], item.get("count", 0), item["status"]]])
    return Response("\ufeff" + data.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="homeease-report.csv"'})


@app.get("/reports/print", response_class=HTMLResponse)
def print_report(request: Request, days: int = Query(default=7, ge=1, le=90), mode: str = Query(default="live", pattern="^(live|replay)$")):
    return templates.TemplateResponse(request=request, name="report.html", context={"report": system.report(days, mode), "event": None, "exercises": EXERCISES})


@app.get("/reports/events/{event_id}", response_class=HTMLResponse)
def print_event(event_id: str, request: Request):
    return templates.TemplateResponse(request=request, name="report.html", context={"event": system.store.event(event_id), "report": None, "exercises": EXERCISES,
        "notifications": system.store.notifications(event_id=event_id, limit=10000)})
