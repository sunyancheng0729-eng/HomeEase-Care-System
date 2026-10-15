import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time

import psutil

from care.config import ROOT, LOGS

log = logging.getLogger(__name__)


class ModuleError(RuntimeError):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


class AlgorithmProcess:
    def __init__(self, name):
        self.name = name
        self.process = None
        self.lock = threading.RLock()
        self.responses = queue.Queue()
        self.state = "not_initialized"
        self.error = None
        self.details = {}
        self.last_call = None
        self._logfile = None
        self.worker_process = None

    def _start(self):
        self.responses = queue.Queue()
        responses = self.responses
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("JOBLIB_MULTIPROCESSING", "0")
        self._logfile = (LOGS / f"{self.name}.log").open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "care.workers", self.name], cwd=ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._logfile,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        process = self.process

        def read():
            try:
                for line in process.stdout:
                    try:
                        responses.put(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("%s emitted non-protocol output", self.name)
            finally:
                responses.put({"ok": False, "fatal": True, "error": "模块进程已退出"})

        threading.Thread(target=read, daemon=True, name=f"{self.name}-reader").start()
        self.state = "loading"

    def call(self, method, timeout=90, **params):
        with self.lock:
            started = time.perf_counter()
            if self.process is None or self.process.poll() is not None:
                self.close()
                self._start()
            try:
                self.process.stdin.write(json.dumps({"method": method, "params": params}, allow_nan=False) + "\n")
                self.process.stdin.flush()
                response = self.responses.get(timeout=timeout)
            except (OSError, queue.Empty) as exc:
                self.close()
                self.state, self.error = "unavailable", "模块超时或连接中断，已释放进程，可重新初始化"
                log.exception("%s.%s failed", self.name, method)
                raise ModuleError(self.error) from exc
            self.last_call = time.time()
            if response.get("worker_pid"):
                try:
                    self.worker_process = psutil.Process(response["worker_pid"])
                except psutil.NoSuchProcess:
                    pass
            if not response.get("ok"):
                self.error = response.get("error", "模块不可用")
                if response.get("fatal"):
                    self.state = "unavailable"
                log.warning("%s.%s: %s", self.name, method, self.error)
                raise ModuleError(self.error, response.get("status", 422))
            self.state, self.error = "ready", None
            result = response.get("result")
            if method == "initialize":
                self.details = result
            if method not in {"status", "frame", "snapshot", "camera_frame"}:
                log.info("%s.%s completed in %.3fs", self.name, method, time.perf_counter() - started)
            return result

    def health(self):
        if self.process and self.process.poll() is not None and self.state == "ready":
            self.state, self.error = "unavailable", "模块进程已退出，可重新初始化"
        return {"id": self.name, "state": self.state, "error": self.error,
                "details": self.details, "last_call": self.last_call,
                "pid": self.process.pid if self.process and self.process.poll() is None else None}

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process:
                # Windows venv redirectors own a child interpreter. Track that
                # interpreter even if the redirector was killed independently.
                owned = set()
                for root in (self.worker_process,):
                    if root:
                        try:
                            owned.update([root, *root.children(recursive=True)])
                        except psutil.NoSuchProcess:
                            pass
                try:
                    owned.update(psutil.Process(process.pid).children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                try:
                    process.stdin.close()
                except (OSError, ValueError):
                    pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                _, alive = psutil.wait_procs(list(owned), timeout=1)
                for child in alive:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(alive, timeout=3)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=3)
                for stream in (process.stdout, process.stdin):
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
            self.worker_process = None
            if self._logfile:
                self._logfile.close()
                self._logfile = None
            self.state = "not_initialized"
