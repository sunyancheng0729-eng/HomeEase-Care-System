import base64
import csv
import io
import json
import logging
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from care.adapters.process import AlgorithmProcess, ModuleError
from care.camera import CameraHub
from care.config import EXERCISE, SIGNAL, VIDEO, RUNTIME, EXERCISES
from care.events import Events
from care.storage import Store, now

log = logging.getLogger(__name__)


def canonical_csv(data):
    text = data.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    numeric = []
    header = None
    for row in rows:
        if not row or not any(cell.strip() for cell in row):
            continue
        try:
            values = [float(cell) for cell in row]
        except ValueError:
            if header is None and not numeric:
                header = [v.strip().lower() for v in row]
                continue
            raise ValueError("CSV 含有无法识别的数据行")
        if len(values) not in {4, 5} or not all(math.isfinite(v) for v in values):
            raise ValueError("CSV 需要 4 或 5 列有限数值")
        numeric.append(values)
    if len(numeric) < 12:
        raise ValueError("至少需要 12 个有效加速度样本")
    widths = {len(r) for r in numeric}
    if len(widths) != 1:
        raise ValueError("CSV 列数不一致")
    if len(numeric[0]) == 4:
        if not header or header[0] not in {"t", "t_s", "t_ms"}:
            raise ValueError("4 列 CSV 需要表头 t,ax,ay,az（秒）或 t_ms,ax,ay,az（毫秒）；加速度单位 g")
        factor = 1 if header[0] == "t_ms" else 1000
        numeric = [[r[0] * factor, math.sqrt(sum(v * v for v in r[1:])), *r[1:]] for r in numeric]
    if any(b[0] <= a[0] for a, b in zip(numeric, numeric[1:])):
        raise ValueError("样本时间戳必须严格递增")
    output = io.StringIO(newline="")
    csv.writer(output).writerows(numeric)
    return output.getvalue().encode("utf-8")


class System:
    def __init__(self, store=None):
        self.store = store or Store()
        self.events = Events(self.store)
        self.modules = {name: AlgorithmProcess(name) for name in ("video", "signal", "exercise", "audio")}
        self.pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="homeease")
        self.stop_event = threading.Event()
        self.video = {"state": "idle", "prob": 0, "frames": 0}
        self.video_mode = "replay"
        self.video_run = None
        self.video_finalized = True
        self.video_last_frame_at = 0
        self.video_event_keys = {}
        self.video_lock = threading.RLock()
        self.cameras = CameraHub()
        self.camera_shapes = {}
        self.sensor_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.sensor = {"state": "disconnected", "last_received": None}
        self.sessions_lock = threading.RLock()
        self.session_metadata = {}
        self.session_commands = {}
        self.session_cancel = {}
        self.latest_poses = {}
        self.audio_last_received = None
        self.audio_quality = {"state": "active", "reason": "麦克风输入有效"}
        self.jobs = set()
        self.jobs_lock = threading.RLock()
        self.store.recover_coverage()
        for channel in ("vision", "accel", "audio"):
            self.store.set_coverage(channel, "unavailable", "尚未连接实时输入")
        for old in self.store.sessions():
            if old["status"] in {"running", "paused"}:
                old.update(status="interrupted", quality="service_restarted", ended_at=now())
                self.store.save_session(old)

    def start(self):
        for name in self.modules:
            self.submit(self.initialize, name)
        self.scheduler = threading.Thread(target=self._loop, daemon=True, name="homeease-scheduler")
        self.scheduler.start()

    def submit(self, fn, *args):
        future = self.pool.submit(fn, *args)
        with self.jobs_lock:
            self.jobs.add(future)
        def done(job):
            with self.jobs_lock:
                self.jobs.discard(job)
            if not job.cancelled() and job.exception():
                log.error("Background task failed: %s", job.exception())
        future.add_done_callback(done)
        return future

    def initialize(self, name):
        module = self.modules[name]
        try:
            result = module.call("initialize", timeout=150)
            self.store.audit(name, "initialized", {"model": result.get("model") or result.get("model_id") or result.get("sound_model")})
            return result
        except Exception as exc:
            module.state = "unavailable"
            module.error = str(exc)
            self.store.audit(name, "initialization_failed", {"error": str(exc)})
            raise

    def _loop(self):
        while not self.stop_event.wait(1):
            for name, operation in (("events", self.events.tick), ("video", self.poll_video),
                                    ("coverage", self.check_coverage), ("sessions", self.check_sessions),
                                    ("notifications", self.deliver_webhooks), ("heartbeat", self.store.heartbeat)):
                try:
                    operation()
                except Exception as exc:
                    log.exception("%s scheduler operation failed", name)

    def video_start(self, source, mode):
        with self.video_lock:
            if self.video.get("state") == "running":
                raise ValueError("视频检测正在运行")
            self.video_mode = mode
            self.video_run = uuid.uuid4().hex
            self.video_finalized = False
            self.video_event_keys = {}
            if source.get("type") == "camera":
                source = {**source, "shared_camera": self.cameras.acquire(source.get("index", 0), "video")}
            try:
                self.video = self.modules["video"].call("start", source=source)
            except Exception:
                self.cameras.release("video")
                raise
            self.video_last_frame_at = time.time()
            self.store.audit("video", "started", {"mode": mode, "source": source.get("type")})
            return self.video

    def video_stop(self):
        with self.video_lock:
            try:
                self.video = self.modules["video"].call("stop", timeout=20)
            finally:
                self.cameras.release("video")
            self.poll_video(force=True)
            self.store.set_coverage("vision", "unavailable", "实时视频已停止")
            return self.video

    def poll_video(self, force=False):
        if not self.video_run or (self.video_finalized and not force):
            return
        with self.video_lock:
            previous = self.video.get("frames", 0)
            try:
                status = self.modules["video"].call("status", timeout=12)
            except Exception as exc:
                self.video.update(state="error", error=str(exc))
                self.cameras.release("video")
                raise
            self.video = {k: v for k, v in status.items() if k != "records"}
            self.video_finalized = status["state"] != "running" and not status.get("processing", False)
            if self.video_finalized:
                self.cameras.release("video")
            if status["frames"] != previous:
                self.video_last_frame_at = time.time()
            alarm_at = status.get("alarm_at")
            if alarm_at and alarm_at not in self.video_event_keys:
                event = self.events.create("vision", {"prob": status["prob"], "threshold": status["threshold"]}, self.video_mode,
                    source=status["source"], dedup_key=f"video:{self.video_run}:{alarm_at}",
                    at=datetime.fromtimestamp(alarm_at, timezone.utc).isoformat(timespec="milliseconds"))
                self.video_event_keys[alarm_at] = event["id"]
            for record in status.get("records", []):
                if not self.video_event_keys:
                    continue
                alarm = min(self.video_event_keys, key=lambda stamp: abs(stamp - record["epoch"]))
                if abs(alarm - record["epoch"]) > 2:
                    continue
                eid = self.video_event_keys[alarm]
                old = self.store.event(eid)
                if old["evidence"].get("clip") == record.get("clip"):
                    continue
                def add_clip(event, record=record):
                    event["evidence"].update(record)
                    Events.timeline(event, "evidence_saved", "原视频引擎已保存事件片段")
                self.store.update_event(eid, add_clip)

    def check_coverage(self):
        if self.video_mode == "live" and self.video.get("state") == "running":
            fresh = time.time() - self.video_last_frame_at < 8
            visible = self.video.get("person_visible", False)
            self.store.set_coverage("vision", "active" if fresh and visible else "reduced",
                "实时画面有效" if fresh and visible else ("画面中未检测到人体" if fresh else "视频数据已超时"))
        else:
            self.store.set_coverage("vision", "unavailable", "正在样例回放" if self.video_mode == "replay" and self.video.get("state") == "running" else "未开启实时视频")
        if self.sensor["state"] == "connected":
            fresh = time.time() - (self.sensor.get("last_received") or 0) < 5
            self.store.set_coverage("accel", "active" if fresh else "reduced", "已接收实时传感器数据" if fresh else "传感器数据已超时")
        else:
            self.store.set_coverage("accel", "unavailable", "传感器未连接")
        consent = self.store.settings()["audio_consent"]
        fresh_audio = consent and self.audio_last_received and time.time() - self.audio_last_received < 20
        self.store.set_coverage("audio", self.audio_quality["state"] if fresh_audio else "unavailable",
            self.audio_quality["reason"] if fresh_audio else ("未开启音频授权" if not consent else "未连接连续麦克风输入"))

    def status(self):
        coverage = [r for r in self.store.coverage() if r["ended_at"] is None]
        active = sum(r["state"] == "active" for r in coverage if r["channel"] in {"vision", "accel"})
        return {"modules": [m.health() for m in self.modules.values()], "video": self.video,
                "video_mode": self.video_mode, "sensor": self.sensor, "coverage": coverage,
                "cameras": self.cameras.status(),
                "coverage_state": "active" if active == 2 else "reduced" if active else "unavailable",
                "server_time": now(), "audio_consent": self.store.settings()["audio_consent"]}

    def signal_offline(self, path, source):
        result = self.modules["signal"].call("offline", path=str(path))
        run = uuid.uuid4().hex
        records = [self.events.create("accel", evidence, "replay", source, f"signal:{run}:{index}")
                   for index, evidence in enumerate(result["events"])]
        self.store.audit("signal", "offline_completed", {"windows": result["info"]["n_windows"], "events": len(records)})
        return {**result, "records": records, "mode": "replay"}

    def signal_results(self, result, stream_id):
        for alarm in result.get("alarms", []):
            self.events.create("accel", alarm, "live", "实时传感器", f"sensor:{stream_id}:{alarm['t_alarm']}",
                               at=datetime.fromtimestamp(alarm["wall_alarm"], timezone.utc).isoformat(timespec="milliseconds"))
        for evidence in result.get("events", []):
            event = self.events.create("accel", evidence, "live", "实时传感器", f"sensor:{stream_id}:{evidence['t_alarm']}",
                at=datetime.fromtimestamp(evidence.get("wall_alarm") or time.time(), timezone.utc).isoformat(timespec="milliseconds"))
            if "snippet" not in event["evidence"]:
                self.store.update_event(event["id"], lambda item, e=evidence: item["evidence"].update(e))

    def create_session(self, exercise, target, mode):
        with self.sessions_lock:
            active = [m for m in self.session_metadata.values() if m.get("active")]
            if active:
                raise ValueError("请先结束当前运动训练")
            inactive = [sid for sid, meta in self.session_metadata.items() if not meta["active"]]
            for old_sid in inactive[:-99]:
                self.session_metadata.pop(old_sid, None)
                self.session_commands.pop(old_sid, None)
                self.session_cancel.pop(old_sid, None)
                self.latest_poses.pop(old_sid, None)
                self.camera_shapes.pop(old_sid, None)
            result = self.modules["exercise"].call("create", exercise=exercise, target_reps=target)
            sid = result["session_id"]
            self.session_metadata[sid] = {"created_at": now(), "mode": mode, "active": True}
            self.session_commands[sid] = threading.RLock()
            self.session_cancel[sid] = threading.Event()
            return self.save_session(result)

    def save_session(self, result):
        sid = result["session_id"]
        meta = self.session_metadata[sid]
        meta["status"] = result["status"]
        payload = {**result, "created_at": meta["created_at"], "updated_at": now(), "mode": meta["mode"]}
        if result["status"] == "ended":
            payload["ended_at"] = now()
            meta["active"] = False
        return self.store.save_session(payload)

    def session_control(self, sid, action):
        if sid not in self.session_metadata:
            raise KeyError(sid)
        with self.session_commands[sid]:
            if not self.session_metadata[sid]["active"]:
                if action == "end":
                    return next(s for s in self.store.sessions() if s["session_id"] == sid)
                raise ValueError("训练已经结束")
            if action == "end":
                self.session_cancel[sid].set()
            result = self.save_session(self.modules["exercise"].call("control", session_id=sid, action=action))
            if action == "end":
                self.modules["exercise"].call("release", session_id=sid)
                self.cameras.release(sid)
            return result

    def start_camera_session(self, exercise, target, index):
        with self.sessions_lock:
            if any(meta["active"] for meta in self.session_metadata.values()):
                raise ValueError("请先结束当前运动训练")
            owner = "exercise-opening"
            descriptor = self.cameras.acquire(index, owner)
            try:
                self.modules["exercise"].call("pose_initialize")
                result = self.create_session(exercise, target, "live")
                sid = result["session_id"]
                self.cameras.acquire(index, sid)
                self.session_metadata[sid]["shared_camera"] = True
                self.session_metadata[sid]["last_viewer"] = time.monotonic()
                self.camera_shapes[sid] = descriptor["shape"]
                self.submit(self.camera_exercise, sid, descriptor)
                return {**result, "shared_camera": True, "frame_shape": descriptor["shape"]}
            finally:
                self.cameras.release(owner)

    def camera_exercise(self, sid, descriptor):
        sequence = 0
        base = time.perf_counter()
        last_stamp = None
        stamps = []
        try:
            while not self.session_cancel[sid].is_set() and not self.stop_event.is_set():
                with self.session_commands[sid]:
                    if self.session_cancel[sid].is_set() or self.stop_event.is_set():
                        break
                    if self.session_metadata[sid].get("status") == "running":
                        result = self.modules["exercise"].call("camera_frame", session_id=sid,
                            descriptor=descriptor, base_stamp=base, last_sequence=sequence)
                        if not result.get("duplicate"):
                            sequence = result.pop("capture_sequence")
                            self.latest_poses[sid] = result.pop("keypoints")
                            diagnostics = result["capture_diagnostics"]
                            stamp = diagnostics["timestamp_ms"]
                            diagnostics["gap_ms"] = round(stamp - last_stamp, 2) if last_stamp is not None else None
                            last_stamp = stamp
                            stamps.append(stamp)
                            stamps = stamps[-60:]
                            diagnostics["observation_fps"] = round((len(stamps) - 1) * 1000 / (stamps[-1] - stamps[0]), 1) if len(stamps) > 1 else 0
                            self.save_session(result)
                if self.session_cancel[sid].wait(.005):
                    break
        except Exception as exc:
            self.interrupt_session(sid, exc)
        finally:
            # Release the worker's memory handle before the last capture owner.
            with self.session_commands[sid]:
                if self.modules["exercise"].state == "ready":
                    try:
                        self.modules["exercise"].call("camera_detach", session_id=sid)
                    except Exception:
                        pass
            self.cameras.release(sid)

    def check_sessions(self):
        active = {sid for sid, meta in self.session_metadata.items() if meta["active"] and meta["mode"] == "live"}
        if not active:
            return
        for item in self.store.sessions(mode="live"):
            meta = self.session_metadata.get(item["session_id"], {})
            if item["session_id"] in active and meta.get("shared_camera") and time.monotonic() - meta["last_viewer"] > 15:
                self.session_control(item["session_id"], "end")
                self.store.audit("exercise", "client_disconnected", {"session_id": item["session_id"]})
                continue
            if item["session_id"] in active and item["status"] == "running":
                stamp = datetime.fromisoformat(item.get("updated_at", item["created_at"])).timestamp()
                if time.time() - stamp > 10:
                    self.session_control(item["session_id"], "pause")
                    self.store.audit("exercise", "capture_timeout", {"session_id": item["session_id"]})

    def session_frames(self, sid, body):
        if sid not in self.session_metadata:
            raise KeyError(sid)
        with self.session_commands[sid]:
            result = self.modules["exercise"].call("frames", session_id=sid, body=body)
            self.latest_poses[sid] = body["frames"][-1]["keypoints"]
            return self.save_session(result)

    def session_image(self, sid, data, timestamp_ms, request_id):
        if sid not in self.session_metadata:
            raise KeyError(sid)
        with self.session_commands[sid]:
            result = self.modules["exercise"].call("image", session_id=sid, image=base64.b64encode(data).decode(),
                        timestamp_ms=timestamp_ms, request_id=request_id)
            self.latest_poses[sid] = result.pop("keypoints")
            self.save_session(result)
            return {**result, "keypoints": self.latest_poses[sid]}

    def replay(self, sid, exercise):
        record = json.loads((EXERCISE / "examples" / f"{exercise}_replay.json").read_text())
        try:
            for index in range(0, len(record["frames"]), 20):
                if self.session_cancel[sid].is_set() or self.stop_event.is_set():
                    return
                while self.session_metadata[sid]["active"]:
                    with self.session_commands[sid]:
                        snapshot = self.modules["exercise"].call("snapshot", session_id=sid)
                    if snapshot["status"] != "paused":
                        break
                    if self.session_cancel[sid].wait(.2):
                        return
                self.session_frames(sid, {"request_id": f"replay-{index}", "frames": record["frames"][index:index+20]})
                if self.session_cancel[sid].wait(.18):
                    return
            self.session_control(sid, "end")
        except Exception as exc:
            self.interrupt_session(sid, exc)

    def video_exercise(self, sid, path):
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError("无法打开运动视频")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or not math.isfinite(fps) or not 1 <= fps <= 240:
                raise ValueError("无法确定视频帧率")
            self.modules["exercise"].call("pose_initialize")
            index = 0
            while not self.session_cancel[sid].is_set() and not self.stop_event.is_set():
                with self.session_commands[sid]:
                    snapshot = self.modules["exercise"].call("snapshot", session_id=sid)
                    if snapshot["status"] == "paused":
                        paused = True
                    else:
                        paused = False
                        ok, frame = cap.read()
                        if not ok:
                            break
                        encoded, jpeg = cv2.imencode(".jpg", frame)
                        if not encoded:
                            raise ValueError("视频帧编码失败")
                        self.session_image(sid, jpeg.tobytes(), index * 1000 / fps, f"video-{index}")
                        index += 1
                if paused and self.session_cancel[sid].wait(.2):
                    return
            if not self.session_cancel[sid].is_set():
                self.session_control(sid, "end")
        except Exception as exc:
            self.interrupt_session(sid, exc)
        finally:
            cap.release()

    def interrupt_session(self, sid, error):
        log.exception("Exercise session %s interrupted", sid)
        with self.session_commands[sid]:
            sessions = self.store.sessions()
            current = next((s for s in sessions if s["session_id"] == sid), None)
            if current and current["status"] != "ended":
                current.update(status="interrupted", quality="module_error", error=str(error), ended_at=now())
                self.store.save_session(current)
            self.session_metadata[sid]["active"] = False
            self.session_cancel[sid].set()
            self.latest_poses.pop(sid, None)
            try:
                self.modules["exercise"].call("release", session_id=sid, timeout=10)
            except Exception:
                pass
            self.cameras.release(sid)

    def audio_analyze(self, data, task, mode, event_id=None):
        if not self.store.settings()["audio_consent"]:
            raise ValueError("需先在设置中获得长者的音频处理授权")
        result = self.modules["audio"].call("analyze", timeout=120, data=base64.b64encode(data).decode(), task=task)
        if not self.store.settings()["audio_consent"]:
            raise ValueError("音频授权已撤回，结果已丢弃")
        result.pop("transcript", None)
        if task == "reply":
            if not event_id:
                raise ValueError("语音回复需要关联一个事件")
            event = self.events.inquiry_reply(event_id, result)
            return {"result": result, "event": event}
        help_result = result.get("help_result", {}) if task == "monitor" else result if task == "help" else {}
        if help_result.get("help"):
            event = self.events.create("audio", help_result, mode, "麦克风" if mode == "live" else "音频回放")
            help_result["event_id"] = event["id"]
        stored = self.store.save_sound({"task": task, "result": result}, mode)
        return {"result": result, "id": stored["id"]}

    def deliver_webhooks(self):
        destination = self.store.settings().get("webhook_url")
        if not destination:
            return
        with self.store.connection() as conn:
            rows = conn.execute("SELECT * FROM notifications WHERE kind='webhook' AND status IN ('queued','retry') AND next_at<=? ORDER BY created_at LIMIT 1", (time.time(),)).fetchall()
        for row in rows:
            try:
                with httpx.Client(timeout=5, follow_redirects=False) as client:
                    response = client.post(destination, json=json.loads(row["payload"]), headers={"Idempotency-Key": row["id"]})
                    response.raise_for_status()
                state = "delivered"
            except Exception:
                state = "failed" if row["attempts"] >= 4 else "retry"
            with self.store.connection() as conn:
                payload = json.loads(row["payload"])
                payload["last_attempt_at"] = now()
                if state == "delivered":
                    payload["delivered_at"] = payload["last_attempt_at"]
                conn.execute("UPDATE notifications SET status=?,attempts=attempts+1,next_at=?,payload=? WHERE id=?", (
                    state, time.time() + min(300, 10 * 2 ** row["attempts"]), json.dumps(payload, ensure_ascii=False), row["id"]))

    def report(self, days=7, mode="live"):
        tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(tz).date()
        start = datetime.combine(today - timedelta(days=days-1), datetime.min.time(), tzinfo=tz).astimezone(timezone.utc)
        events = self.store.events(mode=mode, limit=100000, since=start.isoformat())
        sessions = self.store.sessions(mode=mode, since=start.isoformat(), limit=100000)
        coverage = self.store.coverage(start.isoformat()) if mode == "live" else []
        series = []
        for offset in range(days):
            date = today - timedelta(days=days-1-offset)
            same_day = lambda item: datetime.fromisoformat(item["created_at"]).astimezone(tz).date() == date
            daily_events = [e for e in events if same_day(e)]
            daily_sessions = [s for s in sessions if same_day(s)]
            series.append({"date": date.isoformat(), "events": len(daily_events), "reps": sum(s.get("count", 0) for s in daily_sessions),
                           "confirmed_falls": sum(e["outcome"] == "confirmed_fall" for e in daily_events)})
        unavailable = {}
        for row in coverage:
            end = datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else datetime.now(timezone.utc)
            begin = max(start, datetime.fromisoformat(row["started_at"]))
            if row["state"] != "active":
                unavailable[row["channel"]] = unavailable.get(row["channel"], 0) + max(0, (end - begin).total_seconds())
        completed = [s for s in sessions if s["status"] == "ended"]
        return {"generated_at": now(), "days": days, "mode": mode, "resident": self.store.settings()["resident"],
                "events": events, "sessions": sessions, "series": series, "coverage": coverage,
                "notifications": [n for n in self.store.notifications(since=start.isoformat(), limit=100000) if n.get("mode") == mode],
                "summary": {"confirmed_falls": sum(e["outcome"] == "confirmed_fall" for e in events),
                    "false_alarms": sum(e["outcome"] == "false_alarm" for e in events), "open_events": sum(e["state"] == "open" for e in events),
                    "verification_events": sum(e["level"] == "PENDING" for e in events),
                    "unresolved_outcomes": sum(e["outcome"] == "unresolved" for e in events),
                    "sessions": len(completed), "reps": sum(s.get("count", 0) for s in completed),
                    "participation_days": len({datetime.fromisoformat(s["created_at"]).astimezone(tz).date() for s in completed}),
                    "reps_by_exercise": {key: sum(s.get("count", 0) for s in completed if s["exercise"] == key) for key in EXERCISES},
                    "coverage_unavailable_s": unavailable}}

    def close(self):
        self.stop_event.set()
        for cancel in self.session_cancel.values():
            cancel.set()
        if hasattr(self, "scheduler"):
            self.scheduler.join(timeout=8)
        self.pool.shutdown(wait=True, cancel_futures=True)
        for sid, meta in list(self.session_metadata.items()):
            if meta["active"]:
                try:
                    self.session_control(sid, "end")
                except Exception:
                    pass
        for module in self.modules.values():
            module.close()
        self.cameras.close()
        for channel in ("vision", "accel", "audio"):
            self.store.set_coverage(channel, "unavailable", "服务已停止")
