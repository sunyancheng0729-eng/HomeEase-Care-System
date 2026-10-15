"""One original algorithm import namespace per process; JSON over local pipes."""
import base64
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import traceback
import time
from datetime import date, datetime
from pathlib import Path

from care.config import VIDEO, SIGNAL, EXERCISE, AUDIO, RUNTIME, prepare


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class VideoWorker:
    def __init__(self):
        # The delivered engine expects systems/ beside standalone/. Mirror only
        # its integration wrapper byte-for-byte; all inference comes from originals.
        bridge = RUNTIME / "video_bridge"
        systems = bridge / "systems"
        standalone = bridge / "standalone"
        systems.mkdir(parents=True, exist_ok=True)
        standalone.mkdir(parents=True, exist_ok=True)
        for name in ("engine.py", "store.py", "a4_weights.pkl"):
            shutil.copy2(VIDEO / name, systems / name)
        for name in ("main.py", "a4_core.py", "yolov8n-pose.onnx", "pose_best.pt", "a4_weights.pkl"):
            target = standalone / name
            source = VIDEO / "standalone" / name
            if not target.exists() or target.stat().st_mtime_ns != source.stat().st_mtime_ns:
                shutil.copy2(source, target)
        self.module = load_file("homeease_original_video_engine", systems / "engine.py")
        import cv2
        cv2.setNumThreads(max(1, int(os.environ.get("HOMEEASE_VIDEO_CV_THREADS", "4"))))
        original_source = self.module.FrameSource

        class SharedSource(original_source):
            def __init__(self, spec):
                self.reader = None
                if spec.get("type") == "camera" and spec.get("shared_camera"):
                    from care.camera import FrameReader
                    self.reader = FrameReader(spec["shared_camera"])
                    self.realtime = True
                    self.label = f"摄像头 {spec.get('index', 0)}"
                else:
                    super().__init__(spec)

            def read(self):
                if self.reader:
                    frame, _, _ = self.reader.read()
                    return True, frame
                return super().read()

            def release(self):
                if self.reader:
                    self.reader.close()
                else:
                    super().release()

        self.module.FrameSource = SharedSource
        self.engine = self.module.DetectionEngine()
        if not self.engine.models_ready:
            raise RuntimeError(self.engine._load_error)
        self.detector_backend = "opencv"
        self.acceleration_error = None
        if os.environ.get("HOMEEASE_VIDEO_DNN", "auto") != "opencv":
            try:
                from care.adapters.onnx import OnnxNetwork
                self.engine._det.net = OnnxNetwork(VIDEO / "standalone" / "yolov8n-pose.onnx", device="cuda")
                self.detector_backend = "onnxruntime-cuda-fp32"
            except Exception as exc:
                self.acceleration_error = str(exc)
                print(f"Video CUDA acceleration unavailable; original OpenCV backend: {exc}", file=sys.stderr)
        # All native libraries are loaded now. Reuse the library inventory while
        # retaining the original per-prediction limits and restoration semantics.
        from threadpoolctl import ThreadpoolController
        self.threadpools = ThreadpoolController()
        self.module._threadpool_limits = self.threadpools.limit
        import numpy as np
        warm_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.engine._det.detect_frame(warm_frame)
        self.engine._pose.extract(warm_frame, [0, 0, 640, 480])
        with self.module.a4_fast_context():
            self.engine._alg.predict(np.zeros((30, 17, 3), dtype=np.float32))

    def initialize(self):
        import cv2
        return {"model": "A4", "pipeline": "YOLOv8n + P1 + A4", "config": self.engine.config(),
                "opencv_threads": cv2.getNumThreads(), "detector_backend": self.detector_backend,
                "acceleration_error": self.acceleration_error}

    def start(self, source):
        if self.engine.is_running():
            raise ValueError("检测正在运行，请先停止")
        return self.engine.start(source)

    def stop(self):
        result = self.engine.stop(timeout=8)
        return result

    def status(self):
        thread = self.engine._thread
        alive = bool(thread and thread.is_alive())
        return {**self.engine.status(), "records": self.engine.records()[:200],
                "person_visible": self.engine._last_det_n == 1, "alarm_at": self.engine._last_alarm,
                "processing": alive}

    def frame(self):
        return base64.b64encode(self.engine.latest_jpeg()).decode("ascii")

    def close(self):
        self.engine.stop(timeout=8)


class SignalWorker:
    def __init__(self):
        sys.path.insert(0, str(SIGNAL))
        from core.model import BestFallModel
        from core import detector, preprocess
        self.model = BestFallModel.load(str(SIGNAL / "model" / "best_model.pkl"))
        self.detector, self.preprocess = detector, preprocess
        self.stream = None

    def initialize(self):
        from core import config
        return {"model": self.model.name, "threshold": config.ALARM_THRESHOLD, "meta": self.model.meta}

    def offline(self, path):
        t, a = self.preprocess.read_acc_csv(path)
        if len(t) < 12:
            raise ValueError("CSV 没有足够的有效数据。原文件读取器要求 5 列：毫秒、合加速度、ax、ay、az")
        events, info = self.detector.detect_offline(t, a, self.model)
        step = max(1, len(t) // 2000)
        return {"events": events, "info": info, "waveform": {"t": t[::step].tolist(), "ax": a[::step, 0].tolist(),
                "ay": a[::step, 1].tolist(), "az": a[::step, 2].tolist()}}

    def stream_start(self):
        if self.stream is not None:
            raise ValueError("传感器流正在运行")
        self.stream = self.detector.StreamingDetector(self.model)
        return {"state": "listening"}

    def samples(self, samples):
        if self.stream is None:
            raise ValueError("请先启动传感器流")
        events = []
        for sample in samples:
            events.extend(self.stream.push(*sample))
        alarms = self.stream.pending_alarms[:]
        self.stream.pending_alarms.clear()
        return {"events": events, "alarms": alarms, "prob": self.stream.last_prob, "windows": self.stream.n_eval,
                "recent": self.stream.recent[-120:]}

    def flush(self):
        if self.stream is None:
            return {"events": []}
        events = self.stream.flush()
        self.stream = None
        return {"events": events}

    def close(self):
        self.stream = None


class ExerciseWorker:
    def __init__(self):
        import torch
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(EXERCISE))
        from homeease.api import create_app
        from homeease.engine import Predictor
        requested = os.environ.get("HOMEEASE_EXERCISE_DEVICE", "auto")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
        predictor = Predictor(device=self.device)
        import numpy as np
        predictor.stream().update(np.zeros((12, 3), dtype=np.float32), 0)
        self.client = TestClient(create_app(predictor=predictor))
        self.client.__enter__()
        self.pose = None
        self.camera_readers = {}

    def initialize(self):
        return {**self._request("get", "/health"), "device": self.device}

    def _request(self, method, path, body=None):
        response = self.client.request(method, path, json=body)
        if response.status_code >= 400:
            from fastapi import HTTPException
            raise HTTPException(response.status_code, response.json())
        return response.json() if response.content else {}

    def create(self, exercise, target_reps):
        return self._request("post", "/v1/sessions", {"exercise": exercise, "target_reps": target_reps})

    def frames(self, session_id, body):
        return self._request("post", f"/v1/sessions/{session_id}/frames", body)

    def snapshot(self, session_id):
        return self._request("get", f"/v1/sessions/{session_id}")

    def control(self, session_id, action):
        return self._request("post", f"/v1/sessions/{session_id}/control", {"action": action})

    def release(self, session_id):
        self.camera_detach(session_id)
        return self._request("delete", f"/v1/sessions/{session_id}")

    def camera_detach(self, session_id):
        reader = self.camera_readers.pop(session_id, None)
        if reader:
            reader.close()
        return {"detached": True}

    def pose_initialize(self):
        if self.pose is None:
            import numpy as np
            from homeease.pose import PoseEstimator
            self.pose = PoseEstimator(device=0 if self.device.startswith("cuda") else self.device)
            # Ultralytics/CUDA warm-up must finish before a live session's clock.
            self.pose.estimate(np.zeros((480, 640, 3), dtype=np.uint8))
        return {"model": "yolo11n-pose.pt", "device": self.device}

    def image(self, session_id, image, timestamp_ms, request_id):
        import cv2
        import numpy as np
        self.pose_initialize()
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法读取图像")
        return self._pose_frame(session_id, frame, timestamp_ms, request_id)

    def camera_frame(self, session_id, descriptor, base_stamp, last_sequence):
        from care.camera import FrameReader
        reader = self.camera_readers.get(session_id)
        if reader is None:
            reader = self.camera_readers[session_id] = FrameReader(descriptor)
        frame, sequence, stamp = reader.read()
        if sequence == last_sequence:
            return {"duplicate": True}
        self.pose_initialize()
        result = self._pose_frame(session_id, frame, max(0, (stamp - base_stamp) * 1000), f"camera-{sequence}")
        return {**result, "capture_sequence": sequence, "frame_shape": list(frame.shape)}

    def _pose_frame(self, session_id, frame, timestamp_ms, request_id):
        started = time.perf_counter()
        points, person_id = self.pose.estimate(frame)
        pose_ms = (time.perf_counter() - started) * 1000
        result = self.frames(session_id, {"request_id": request_id, "frames": [{"timestamp_ms": timestamp_ms,
                              "person_id": person_id, "keypoints": points.tolist()}]})
        required = {5: "left_shoulder", 6: "right_shoulder", 11: "left_hip", 12: "right_hip",
                    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}
        return {**result, "keypoints": points.tolist(), "person_id": person_id,
                "capture_diagnostics": {"timestamp_ms": timestamp_ms, "pose_ms": round(pose_ms, 2),
                    "processing_ms": round((time.perf_counter() - started) * 1000, 2), "device": self.device,
                    "missing_joints": [name for i, name in required.items() if points[i, 2] < .3]}}

    def close(self):
        for reader in self.camera_readers.values():
            reader.close()
        self.camera_readers.clear()
        self.client.__exit__(None, None, None)


class AudioWorker:
    def __init__(self):
        import os
        os.environ["HOMEEASE_ROOT"] = str(AUDIO)
        sys.path.insert(0, str(AUDIO / "src"))
        from algorithm3 import Algorithm3
        # Strict Silero availability: never silently substitute another VAD.
        from homeease.ml.audio.vad import create
        self.algorithm = Algorithm3(vad="silero")
        self.algorithm._vad = create("silero")
        self.algorithm.vad_name = "silero"
        self.pipe = None
        self.listener = None
        self.inquiry_since = None

    def initialize(self):
        result = self.algorithm.warm_up()
        if result.get("sound_model_error") or result.get("asr_error"):
            raise RuntimeError(result.get("sound_model_error") or result.get("asr_error"))
        return result

    def analyze(self, data, task, text=None):
        if task == "reply_text":
            return self.algorithm.classify_reply(text=text)
        import numpy as np
        x = self.algorithm.decode(base64.b64decode(data))
        try:
            if len(x) > 16000 * 120:
                raise ValueError("音频最长 120 秒")
            if task == "monitor":
                sounds = self.algorithm.label_sounds(x)
                help_result = self.algorithm.detect_spoken_help(x)
                return {"sounds": sounds, "help_result": help_result, "duration_s": len(x) / 16000,
                        "raw_audio_stored": False}
            if task == "sounds":
                return self.algorithm.label_sounds(x)
            if task == "help":
                return self.algorithm.detect_spoken_help(x)
            if task == "reply":
                return self.algorithm.classify_reply(audio=x)
            raise ValueError("未知音频任务")
        finally:
            x.fill(0)

    def outcome(self, record, level="PENDING", resolved=False, button_ok=False):
        return self.algorithm.inquiry_outcome(record, decision_level=level, resolved=resolved, button_ok=button_ok)

    def tag(self, markers):
        return self.algorithm.alarm_tag(markers)

    def live_start(self):
        from collections import deque
        from homeease.core.clock import SystemClock
        from homeease.ml.audio.monitor import SoundMonitor
        from homeease.ml.audio.pipeline import Function3Pipeline
        from homeease.ml.audio.help_listener import HelpListener
        if self.pipe is not None:
            raise ValueError("麦克风监听正在运行")
        self.clock = SystemClock()
        self.calls = deque(maxlen=32)
        self.inquiry_since = None
        algorithm = self.algorithm

        class LockedASR:
            def transcribe(self, *args, **kwargs):
                with algorithm._lock:
                    return algorithm._get_asr().transcribe(*args, **kwargs)

        self.listener = HelpListener(algorithm.cfg, self.clock, self.calls.append, consent=lambda: self.pipe is not None,
                                     asr=LockedASR(), inquiry_since=lambda: self.inquiry_since)
        self.monitor = SoundMonitor(algorithm.cfg.audio.sound_monitor, algorithm._get_classifier(), self.clock, store=None)
        self.pipe = Function3Pipeline(algorithm.cfg.audio, self.clock, self.monitor, voice_sink=self.listener.submit, vad=algorithm._get_vad())
        self.listener.start()
        return {"state": "listening", "sample_rate": 16000}

    def live_chunk(self, pcm, timestamp_ms, inquiry_active=False):
        import numpy as np
        from dataclasses import asdict
        if self.pipe is None:
            raise ValueError("监听未启动")
        if inquiry_active and self.inquiry_since is None:
            self.inquiry_since = self.clock.mono_ms()
        elif not inquiry_active:
            self.inquiry_since = None
        x = np.frombuffer(base64.b64decode(pcm), dtype="<f4").copy()
        try:
            if not 1 <= len(x) <= 32000 or not np.isfinite(x).all():
                raise ValueError("无效的音频数据块")
            if not hasattr(self, "stream_base") or self.stream_base is None:
                self.stream_base = self.clock.mono_ms() - float(timestamp_ms) - len(x) / 16
            self.pipe.process_chunk(x, self.stream_base + float(timestamp_ms))
            calls = list(self.calls)
            self.calls.clear()
            return {"events": [asdict(e) for e in self.monitor.recent[-30:]], "calls": calls,
                    "pipeline": self.pipe.status(), "listener": self.listener.status(), "raw_audio_stored": False}
        finally:
            x.fill(0)

    def live_stop(self):
        from dataclasses import asdict
        pipe, self.pipe = self.pipe, None
        if pipe:
            pipe.close(timeout=5)
        if self.listener:
            self.listener.stop()
            self.listener = None
        self.stream_base = None
        self.inquiry_since = None
        calls = list(getattr(self, "calls", []))
        if hasattr(self, "calls"):
            self.calls.clear()
        return {"state": "stopped", "calls": calls,
                "events": [asdict(e) for e in self.monitor.recent[-30:]] if pipe else []}

    def close(self):
        self.live_stop()


def protocol_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Unsupported worker result type: {type(value).__name__}")


def main():
    prepare()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    instance = None
    classes = {"video": VideoWorker, "signal": SignalWorker, "exercise": ExerciseWorker, "audio": AudioWorker}
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if instance is None:
                    instance = classes[sys.argv[1]]()
                method = request["method"]
                if method.startswith("_") or method == "close":
                    raise ValueError("未知调用")
                result = getattr(instance, method)(**request.get("params", {}))
                response = {"ok": True, "result": result, "worker_pid": os.getpid()}
                encoded = json.dumps(response, ensure_ascii=True, allow_nan=False, default=protocol_default)
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "fatal": instance is None, "error": str(getattr(exc, "detail", exc)),
                            "status": getattr(exc, "status_code", 422), "worker_pid": os.getpid()}
                encoded = json.dumps(response, ensure_ascii=True, allow_nan=False)
            protocol.write(encoded + "\n")
            protocol.flush()
    finally:
        if instance:
            instance.close()


if __name__ == "__main__":
    main()
