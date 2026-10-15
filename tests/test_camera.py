import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from care.camera import CameraHub, FrameReader


class Capture:
    def __init__(self, fail=False):
        self.closed = False
        self.fail = fail
        self.tick = 0

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def read(self):
        time.sleep(.005)
        self.tick += 1
        if self.fail and self.tick > 2:
            return False, None
        return True, np.full((48, 64, 3), self.tick % 255, np.uint8)

    def release(self):
        self.closed = True


def test_one_capture_shared_losslessly_and_released_only_after_last_user():
    captures = []
    def factory(*args):
        captures.append(Capture())
        return captures[-1]
    hub = CameraHub(factory)
    first = hub.acquire(0, "video")
    second = hub.acquire(0, "exercise")
    assert first == second and len(captures) == 1
    reader = FrameReader(first)
    try:
        for _ in range(30):
            frame, sequence, stamp = reader.read()
            assert np.all(frame == frame[0, 0, 0])
            assert frame[0, 0, 0] == sequence % 255
            assert stamp <= time.perf_counter()
        hub.release("video")
        assert not captures[0].closed
        assert hub.preview("exercise").startswith(b"\xff\xd8")
        assert hub.status()[0]["users"] == ["exercise"]
    finally:
        reader.close()
        hub.release("exercise")
        hub.close()
    assert captures[0].closed and hub.status() == []
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=first["name"])


def test_concurrent_acquire_opens_once():
    calls = []
    hub = CameraHub(lambda *args: calls.append(args) or Capture())
    try:
        threads = [threading.Thread(target=hub.acquire, args=(0, f"task-{i}")) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(calls) == 1
        assert len(hub.status()[0]["users"]) == 8
    finally:
        hub.close()


def test_capture_failure_is_visible_and_can_restart_after_release():
    hub = CameraHub(lambda *args: Capture(fail=True))
    try:
        hub.acquire(0, "video")
        deadline = time.monotonic() + 2
        while not hub.status()[0]["error"] and time.monotonic() < deadline:
            time.sleep(.01)
        assert hub.preview("video") is None
        with pytest.raises(ValueError, match="连接中断"):
            hub.acquire(0, "exercise")
        hub.release("video")
        assert hub.acquire(0, "exercise")["shape"] == [48, 64, 3]
    finally:
        hub.close()
