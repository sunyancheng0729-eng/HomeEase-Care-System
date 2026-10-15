"""One physical capture per device; lossless shared frames for isolated workers."""
import logging
import os
import struct
import threading
import time
from multiprocessing import shared_memory

import numpy as np

log = logging.getLogger(__name__)
HEADER = 16


class FrameReader:
    def __init__(self, descriptor):
        self.memory = shared_memory.SharedMemory(name=descriptor["name"])
        self.shape = tuple(descriptor["shape"])
        self.size = int(np.prod(self.shape))

    def read(self):
        # A sequence check prevents reading pixels from two different captures.
        for _ in range(100):
            before, stamp = struct.unpack_from("<Qd", self.memory.buf)
            if before and not before % 2:
                pixels = bytes(self.memory.buf[HEADER:HEADER + self.size])
                after = struct.unpack_from("<Q", self.memory.buf)[0]
                if before == after:
                    if time.perf_counter() - stamp > 5:
                        raise RuntimeError("摄像头画面中断，请检查连接后重新启动")
                    return np.frombuffer(pixels, np.uint8).reshape(self.shape).copy(), before // 2, stamp
            time.sleep(.001)
        raise RuntimeError("摄像头画面暂不可用")

    def close(self):
        self.memory.close()


class CameraCapture:
    def __init__(self, index, factory=None):
        import cv2
        self.index = index
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.jpeg = None
        self.sequence = 0
        self.frames = 0
        self.started = time.perf_counter()
        self.users = set()
        self.memory = None
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = (factory or cv2.VideoCapture)(index, backend)
        try:
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = (factory or cv2.VideoCapture)(index)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开摄像头 {index}。请关闭相机、会议软件，并检查 Windows 相机隐私权限")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise ValueError(f"摄像头 {index} 已连接但未返回画面，请检查设备占用或连接")
            self.shape = frame.shape
            self.memory = shared_memory.SharedMemory(create=True, size=HEADER + frame.nbytes)
            self.memory.buf[:HEADER] = bytes(HEADER)
            self._publish(frame)
        except Exception:
            self.cap.release()
            if self.memory:
                self.memory.close()
                self.memory.unlink()
            raise
        self.thread = threading.Thread(target=self._run, name=f"camera-{index}", daemon=True)
        self.thread.start()
        log.info("Camera %s opened, shape=%s", index, self.shape)

    def _publish(self, frame):
        import cv2
        if frame.shape != self.shape:
            raise RuntimeError("摄像头分辨率在采集中发生变化，请重新启动")
        self.sequence += 2
        struct.pack_into("<Q", self.memory.buf, 0, self.sequence - 1)
        self.memory.buf[HEADER:] = frame.tobytes()
        struct.pack_into("<d", self.memory.buf, 8, time.perf_counter())
        struct.pack_into("<Q", self.memory.buf, 0, self.sequence)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            if ok:
                self.jpeg = jpeg.tobytes()
            self.frames += 1

    def _run(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("摄像头连接中断")
                self._publish(frame)
        except Exception as exc:
            self.error = str(exc)
            struct.pack_into("<d", self.memory.buf, 8, 0)
            log.exception("Camera %s capture failed", self.index)
        finally:
            self.cap.release()

    def descriptor(self):
        return {"name": self.memory.name, "shape": list(self.shape), "index": self.index}

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.cap.release()
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("摄像头采集线程尚未退出，暂不能重新打开设备")
        self.memory.close()
        self.memory.unlink()
        log.info("Camera %s released", self.index)


class CameraHub:
    def __init__(self, factory=None):
        self.lock = threading.RLock()
        self.devices = {}
        self.factory = factory

    def acquire(self, index, owner):
        with self.lock:
            capture = self.devices.get(index)
            if capture and capture.error:
                raise ValueError(capture.error + "，请停止使用该摄像头的任务后重试")
            if capture is None:
                capture = CameraCapture(index, self.factory)
                self.devices[index] = capture
            capture.users.add(owner)
            log.info("Camera %s attached to %s", index, owner)
            return capture.descriptor()

    def release(self, owner):
        with self.lock:
            for index, capture in list(self.devices.items()):
                capture.users.discard(owner)
                if not capture.users:
                    capture.close()
                    del self.devices[index]

    def preview(self, owner):
        with self.lock:
            capture = next((c for c in self.devices.values() if owner in c.users), None)
            if capture is None or capture.error:
                return None
            with capture.lock:
                return capture.jpeg

    def status(self):
        with self.lock:
            return [{"index": c.index, "users": sorted(c.users), "error": c.error,
                     "capture_fps": round(c.frames / max(.1, time.perf_counter() - c.started), 1)}
                    for c in self.devices.values()]

    def close(self):
        with self.lock:
            for index, capture in list(self.devices.items()):
                capture.close()
                del self.devices[index]
