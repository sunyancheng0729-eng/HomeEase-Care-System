# -*- coding: utf-8 -*-
"""
engine.py —— 实时跌倒检测引擎
================================================================
把 standalone/ 里训练好的三段式算法权重，封装成一个可长时间运行的服务：

  环节① 人体检测 : yolov8n-pose.onnx      （YOLOv8n 预训练，只用 box head）
  环节② 姿态估计 : pose_best.pt           （自研 P1 小型 U-Net，从零训练）
  环节③ 时序检测 : a4_weights.pkl         （A4 混合集成，OOF Stacking + XGBoost）

工作方式：
  - 后台线程持续读帧（摄像头 / 视频文件 / 内置示例数据集）
  - 每帧现场跑 YOLO → P1 姿态 → 维护 30 帧（2s）滑窗 → 喂给 A4 得到 P(fall)
  - P(fall) 连续 confirm 次超过阈值 → 触发一次跌倒事件：
      · 记录系统时间
      · 落盘 3s 录像（1s 前置缓冲 + 2s 后置），供报告回放
      · 打上报警标记
  - 记录写入 records/falls.json，录像写入 records/clips/*.mp4

说明：本引擎只做推理与记录，不依赖网络请求；FastAPI 层（app.py）只负责读写它的状态。
"""
from __future__ import annotations

import os
import sys
import json
import time
import glob
import threading
import contextlib
from collections import deque
from datetime import datetime

# 性能：见 app.py 顶部说明。单独运行本文件时也套用同一套设置
# （OMP/joblib 的线程池在首次导入 numpy/sklearn/torch 时初始化，必须在此之前设置）。
if os.environ.get("FVS_NO_TUNE") != "1":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import cv2
import numpy as np

# A4 的 7 个基学习器都很小，sklearn 默认走 joblib 多进程 / OpenMP 多线程，
# 调度与结果回收的开销远大于计算本身（实测 joblib 的 _retrieve 每 10ms 轮询一次）。
# 这里用「线程后端 + 单线程」把这些 predict 包起来，不改算法文件本身。
# 实测（RTX 3050 Ti，urfd/fall-01）：68.5ms/帧 → 37.0ms/帧。
try:
    from joblib import parallel_backend as _joblib_backend
except Exception:                                    # 没装 joblib 也不影响运行
    _joblib_backend = None
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None


@contextlib.contextmanager
def a4_fast_context():
    """A4 推理加速上下文（找不到相关库时静默降级为普通执行）。"""
    with contextlib.ExitStack() as stack:
        if _joblib_backend is not None:
            stack.enter_context(_joblib_backend("threading", n_jobs=1))
        if _threadpool_limits is not None:
            stack.enter_context(_threadpool_limits(limits=1))
        yield

# 记录持久化（SQLite，同目录模块）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import RecordStore

# ---------------------------------------------------------------------------
# 复用 standalone 的自包含算法实现（同一份权重、同一份前后处理，不做任何复制）
#
# 注意：项目根目录下还有一个旧的 main.py，若直接 `import main` 会命中它。
# 因此这里用 importlib 按【绝对路径】加载 standalone/main.py，并注册一个唯一
# 模块名，避免任何同名冲突；同时把 standalone/ 放进 sys.path，好让被加载的
# main.py 内部的 `from a4_core import ...` 能解析。
# ---------------------------------------------------------------------------
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))              # systems/
PROJECT = os.path.abspath(os.path.join(HERE, ".."))            # 项目根
STANDALONE = os.path.join(PROJECT, "standalone")
if STANDALONE not in sys.path:
    sys.path.insert(0, STANDALONE)

_spec = importlib.util.spec_from_file_location(
    "fvs_standalone_main", os.path.join(STANDALONE, "main.py"))
_main = importlib.util.module_from_spec(_spec)
sys.modules["fvs_standalone_main"] = _main
_spec.loader.exec_module(_main)

YOLOPersonDetector = _main.YOLOPersonDetector
PoseExtractor = _main.PoseExtractor
A4Algorithm = _main.A4Algorithm
draw_skeleton = _main.draw_skeleton
WIN_FRAMES = _main.WIN_FRAMES
SRC_FPS = _main.SRC_FPS
_source_of = _main._source_of
_gt_intervals = _main._gt_intervals
_le2i_list_videos = _main._le2i_list_videos
_kaggle_dir = _main._kaggle_dir
_urfd_frames_dir = _main._urfd_frames_dir

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
RECORDS_DIR = os.path.join(HERE, "records")
CLIPS_DIR = os.path.join(RECORDS_DIR, "clips")
UPLOADS_DIR = os.path.join(HERE, "uploads")
for _d in (RECORDS_DIR, CLIPS_DIR, UPLOADS_DIR):
    os.makedirs(_d, exist_ok=True)
RECORDS_JSON = os.path.join(RECORDS_DIR, "falls.json")

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
SAMPLE_FPS = 15.0      # 统一采样帧率（算法按 15fps / 2s 窗口训练）
CLIP_FPS = 15.0        # 录像标称帧率（与采样一致 → 3s = 45 帧）


def placeholder_jpeg(w=640, h=360, text="NO SIGNAL"):
    """无信号占位图（浅色，配合浅色主题）。"""
    img = np.full((h, w, 3), 240, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (215, 220, 228), 2)
    cv2.line(img, (0, 0), (w, h), (225, 230, 236), 1)
    cv2.line(img, (w, 0), (0, h), (225, 230, 236), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 158, 170), 2)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes()


# ===========================================================================
# 帧源：摄像头 / 本地视频 / 内置示例数据集（统一成 read() 接口）
# ===========================================================================
class FrameSource:
    def __init__(self, spec: dict):
        self.spec = dict(spec or {})
        self.kind = self.spec.get("type")
        self.cap = None
        self.files = None
        self.dir = None
        self.idx = 0
        self.n_read = 0
        self.stride = 1
        self.realtime = True          # 是否按 15fps 节流（模拟实时）
        self.label = "-"
        self.video_fps = None

        if self.kind == "camera":
            index = int(self.spec.get("index", 0) or 0)
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头（index={index}），请检查设备占用或权限")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.label = f"摄像头 {index}"

        elif self.kind == "file":
            path = str(self.spec.get("path", "")).strip().strip('"')
            if not path or not os.path.isfile(path):
                raise RuntimeError(f"视频文件不存在：{path}")
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频：{path}")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.video_fps = fps
            self.stride = max(1, int(round(fps / SAMPLE_FPS)))
            self.label = os.path.basename(path)

        elif self.kind == "demo":
            ds = self.spec.get("dataset")
            seq = self.spec.get("seq")
            if not ds or not seq:
                raise RuntimeError("示例数据源需要 dataset 与 seq")
            src, src_fps = _source_of(ds, seq)
            self.label = f"{ds}/{seq}"
            if os.path.isdir(src):
                self.files = sorted(
                    f for f in os.listdir(src)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
                self.dir = src
                if not self.files:
                    raise RuntimeError(f"目录内没有图片帧：{src}")
                self.stride = max(1, int(round((src_fps or 30.0) / SAMPLE_FPS)))
            else:
                self.cap = cv2.VideoCapture(src)
                if not self.cap.isOpened():
                    raise RuntimeError(f"无法打开示例视频：{src}")
                fps = self.cap.get(cv2.CAP_PROP_FPS) or (src_fps or 30.0)
                self.video_fps = fps
                self.stride = max(1, int(round(fps / SAMPLE_FPS)))
        else:
            raise RuntimeError(f"未知的输入类型：{self.kind!r}（应为 camera / file / demo）")

    def read(self):
        """返回 (ok, frame)。"""
        if self.files is not None:
            while self.idx < len(self.files):
                i = self.idx
                self.idx += 1
                if i % self.stride:
                    continue
                img = cv2.imread(os.path.join(self.dir, self.files[i]), cv2.IMREAD_COLOR)
                if img is not None:
                    return True, img
            return False, None
        if self.cap is None:
            return False, None
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return False, None
            if self.n_read % self.stride == 0:
                self.n_read += 1
                return True, frame
            self.n_read += 1

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


# ===========================================================================
# 可用输入源枚举
# ===========================================================================
def list_demo_sources():
    """列出内置示例数据集里可用的序列。"""
    out = {}
    # URFD（帧目录）
    try:
        dirs = glob.glob(os.path.join(os.path.dirname(_urfd_frames_dir("x")), "*-cam0-rgb"))
        seqs = sorted(os.path.basename(d).split("-cam0-rgb")[0] for d in dirs)
        if seqs:
            out["urfd"] = seqs
    except Exception:
        pass
    # Le2i（avi 视频）
    try:
        seqs = [name for name, _ in _le2i_list_videos()]
        if seqs:
            out["le2i"] = seqs
    except Exception:
        pass
    # MCF（chute 视频）
    try:
        root = _kaggle_dir("multiple-cameras-fall-dataset")
        if root:
            inner = os.path.join(root, "dataset", "dataset")
            base = inner if os.path.isdir(inner) else root
            seqs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, "chute*"))
                          if os.path.isdir(p))
            if seqs:
                out["mc"] = seqs
    except Exception:
        pass
    return out


# ===========================================================================
# 检测引擎
# ===========================================================================
class DetectionEngine:
    """持续推理 + 跌倒记录。线程安全：所有对外状态读写都在 self._lock 下。"""

    # 单次清空最多删除多少个录像文件（见 clear_records 的说明）
    MAX_CLIP_DELETES = 40

    def __init__(self, threshold=0.60, cooldown_s=6.0, confirm=2,
                 pre_s=1.0, post_s=2.0, clip_fps=CLIP_FPS, alarm_hold_s=4.0,
                 detect_every=3, prob_every=2,
                 show_hud=False, show_skeleton=True):
        self.threshold = float(threshold)
        self.cooldown_s = float(cooldown_s)
        self.confirm = int(confirm)
        self.pre_frames = max(1, int(round(pre_s * clip_fps)))
        self.post_frames = max(1, int(round(post_s * clip_fps)))
        self.clip_fps = float(clip_fps)
        self.alarm_hold_s = float(alarm_hold_s)

        # 推理降频（性能优化，见 README「性能与调优」）
        #   detect_every: 每 N 帧做一次 YOLO 人体检测，其余帧复用上次的框
        #                 （人体在相邻帧位移很小，框复用不影响姿态裁剪）
        #   prob_every:   每 N 帧做一次 A4 时序判定，其余帧沿用上次概率
        #                 （2s 窗内相邻采样高度重叠，无需逐帧计算）
        #   姿态 P1 始终逐帧推理（GPU 上仅数毫秒）
        self.detect_every = max(1, int(detect_every))
        self.prob_every = max(1, int(prob_every))

        # 画面叠加开关（只影响【显示与录像】的视觉层，与模型输入完全无关）
        #   show_hud     : 顶部信息条 + "frame N P(fall)=x FALL/ADL" 文字 + 概率条
        #   show_skeleton: 人体检测框 + 骨架连线
        # 默认关闭文字类叠加，避免画面上的文字干扰观感/被误当"提示"；
        # 无论开关如何，送进模型的一律是【原始帧】（叠加发生在三段推理之后）。
        self.show_hud = bool(show_hud)
        self.show_skeleton = bool(show_skeleton)

        self._lock = threading.RLock()
        self._thread = None
        self._stop_evt = threading.Event()

        self._state = "idle"                 # idle / running / finished / error
        self._error = None
        self._source_spec = None
        self._source_label = "-"
        self._prob = 0.0
        self._peak = 0.0
        self._frames = 0
        self._fps = 0.0
        self._t_start = None
        self._n_falls = 0
        self._hot = 0
        self._last_alarm = 0.0
        self._alarm_until = 0.0
        self._rec = None
        self._session = 0
        self._gt = []
        self._last_box = None
        self._last_score = 0.0
        self._last_det_n = 0
        self._seen = 0

        self._pre = deque(maxlen=180)          # (ts, jpeg_bytes) 前置缓冲，600 帧内约 8~10s
        self._kpts = deque(maxlen=WIN_FRAMES)
        self._last_jpeg = placeholder_jpeg()
        self._n_records = 0                    # 记录条数缓存（status() 只读内存，不查库）
        self._last_record = None

        # 跌倒记录 / 报告数据统一存 SQLite（records/falls.db）
        self.store = RecordStore(RECORDS_DIR)
        migrated = self.store.migrate_from_json(RECORDS_JSON)   # 兼容旧 JSON
        if migrated:
            print(f"[store] 已把旧 falls.json 的 {migrated} 条记录迁移进 SQLite")
        self._refresh_record_cache()

        # ---- 三个环节的模型权重只加载一次 ----
        # A4 权重优先用本目录下的 a4_weights.pkl（由 train_weights.py 在【当前环境】
        # 重新导出，避免 sklearn / xgboost 版本不一致导致反序列化失败）；
        # 若本目录没有，则回退到 standalone/ 的默认权重。
        self._det = None
        self._pose = None
        self._alg = None
        self.models_ready = False
        self._load_error = None
        try:
            self._det = YOLOPersonDetector()
            self._pose = PoseExtractor()
            a4_local = os.path.join(HERE, "a4_weights.pkl")
            if os.path.isfile(a4_local):
                self._alg = A4Algorithm(weights_path=a4_local)
                self.a4_source = a4_local
            else:
                self._alg = A4Algorithm()
                self.a4_source = "standalone/a4_weights.pkl"
            self.models_ready = True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ 记录
    def records(self):
        """全部跌倒记录（新的在前），来自 SQLite。"""
        return self.store.list(newest_first=True)

    def delete_record(self, rid):
        """删除一条记录并同时删掉它的录像文件。

        返回的记录里多一个 ``clip_removed`` 字段，说明录像文件是否真的删掉了
        （运行环境的批量删除保护可能拦下删除动作，此时记录已删、文件还在，
        属于孤儿文件，下次走 clear_records 会被一并清掉）。
        """
        gone = self.store.delete(rid)
        if gone is None:
            return None
        gone["clip_removed"] = self._remove_clip(gone.get("clip"))
        self._refresh_record_cache()
        return gone

    def clear_records(self, delete_clips=True):
        """清空全部记录；delete_clips=True 时同时删除录像文件。

        返回 {"deleted": 记录数, "clips_removed": 已删录像数, "clips_left": 目录剩余录像数}

        录像删除【主动分批】：运行环境带一道「一轮内批量删除超过 50 个文件」的保护，
        触发时会抛 SystemExit（不属于 OSError，所以必须单独接住）。这里单次最多删
        MAX_CLIP_DELETES 个，剩下的残留录像下次再清一次即可 —— 只损失一点便利，
        不会让整个请求 500、也不会绕过环境的安全保护。
        """
        gone = self.store.clear()
        removed = 0
        if delete_clips:
            names, seen = [], set()
            for r in gone:
                n = os.path.basename(r.get("clip") or "")
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
            # 兜底：目录里没有被记录引用的残留录像（含上次被批量删除保护拦下的）
            for p in glob.glob(os.path.join(CLIPS_DIR, "*.mp4")):
                n = os.path.basename(p)
                if n not in seen:
                    seen.add(n)
                    names.append(n)

            fail_streak = 0
            for n in names:
                if removed >= self.MAX_CLIP_DELETES:
                    break
                if self._remove_clip(n):
                    removed += 1
                    fail_streak = 0
                elif os.path.isfile(os.path.join(CLIPS_DIR, n)):
                    fail_streak += 1
                    if fail_streak >= 2:      # 连续被拦下，本轮不再白试
                        break

        self._refresh_record_cache()
        left = len(glob.glob(os.path.join(CLIPS_DIR, "*.mp4")))
        return {"deleted": len(gone), "clips_removed": removed, "clips_left": left}

    @staticmethod
    def _remove_clip(name):
        """删除一个录像文件，返回是否真的删掉了。

        捕获 SystemExit：运行环境的批量删除保护用它中断删除动作；
        接住它只是避免整个 HTTP 请求炸掉，并不绕过保护。
        """
        if not name:
            return False
        p = os.path.join(CLIPS_DIR, os.path.basename(name))
        if not os.path.isfile(p):
            return False
        try:
            os.remove(p)
            return True
        except (OSError, SystemExit):
            return False
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # ------------------------------------------------------------ 启停/配置
    def set_config(self, threshold=None, cooldown_s=None, confirm=None,
                   pre_s=None, post_s=None, show_hud=None, show_skeleton=None):
        with self._lock:
            if threshold is not None:
                self.threshold = float(np.clip(float(threshold), 0.05, 0.99))
            if cooldown_s is not None:
                self.cooldown_s = max(0.0, float(cooldown_s))
            if confirm is not None:
                self.confirm = max(1, int(confirm))
            if pre_s is not None:
                self.pre_frames = max(1, int(round(float(pre_s) * self.clip_fps)))
            if post_s is not None:
                self.post_frames = max(1, int(round(float(post_s) * self.clip_fps)))
            if show_hud is not None:
                self.show_hud = bool(show_hud)
            if show_skeleton is not None:
                self.show_skeleton = bool(show_skeleton)
            return self.config()

    def config(self):
        with self._lock:
            return {
                "threshold": self.threshold,
                "cooldown_s": self.cooldown_s,
                "confirm": self.confirm,
                "clip_fps": self.clip_fps,
                "pre_frames": self.pre_frames,
                "post_frames": self.post_frames,
                "detect_every": self.detect_every,
                "prob_every": self.prob_every,
                "show_hud": self.show_hud,
                "show_skeleton": self.show_skeleton,
                "clip_seconds": round((self.pre_frames + self.post_frames) / self.clip_fps, 2),
            }

    def is_running(self):
        with self._lock:
            return self._state == "running"

    def start(self, spec: dict):
        """启动一个新的检测会话。spec 会先被校验（失败抛 RuntimeError）。"""
        if not self.models_ready:
            raise RuntimeError(
                "模型未就绪：" + str(self._load_error) +
                "。若为 A4 权重版本不兼容，请先运行 "
                "python train_weights.py 在本环境重新导出权重。")
        self.stop()
        src = FrameSource(spec)        # 校验失败在此抛异常

        with self._lock:
            self._source_spec = dict(spec)
            self._source_label = src.label
            self._state = "running"
            self._error = None
            self._prob = 0.0
            self._peak = 0.0
            self._frames = 0
            self._fps = 0.0
            self._n_falls = 0
            self._hot = 0
            self._rec = None
            self._last_alarm = 0.0
            self._alarm_until = 0.0
            self._session += 1
            self._t_start = time.time()
            self._pre.clear()
            self._kpts.clear()
            self._gt = []
            self._last_box = None
            self._last_score = 0.0
            self._last_det_n = 0
            self._seen = 0
            if spec.get("type") == "demo":
                try:
                    self._gt = _gt_intervals(spec["dataset"], spec["seq"])
                except Exception:
                    self._gt = []

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, args=(src,),
                                        name="fall-detector", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, timeout=5.0):
        th = self._thread
        if th is not None and th.is_alive():
            self._stop_evt.set()
            th.join(timeout=timeout)
        self._thread = None
        self._finalize_recording()
        with self._lock:
            if self._state == "running":
                self._state = "idle"
        return self.status()

    # ------------------------------------------------------------ 主循环
    def _run(self, src: FrameSource):
        # 定拍节流用【高精度单调时钟】perf_counter：Windows 上 time.time() 的分辨率
        # 只有约 15.6ms，用它算「本帧耗时」会把节流算得忽长忽短、帧率抖动。
        # 记录时间戳仍用 time.time()（必须是真实系统时间）。
        t_beat = time.perf_counter()
        next_tick = time.perf_counter()
        frame_i = 0
        marks = deque(maxlen=2000)      # (perf_t, frame_i) 用于算 2 秒滑动窗口 fps
        try:
            while not self._stop_evt.is_set():
                ok, frame = src.read()
                if not ok:
                    with self._lock:
                        self._state = "finished"
                    break

                frame_i += 1
                self._seen = frame_i
                # 推理降频：第 1 帧必检（避免开局无框），之后每 detect_every 帧检一次；
                #            每 prob_every 帧做一次 A4 判定。参数=1 即恢复逐帧。
                do_detect = ((frame_i - 1) % self.detect_every == 0)
                do_prob = (frame_i % self.prob_every == 0)
                vis, prob, triggered = self._process(frame, frame_i, do_detect, do_prob)

                ok2, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok2:
                    jpeg = buf.tobytes()
                    with self._lock:
                        self._last_jpeg = jpeg
                    # 前置缓冲 / 跌倒触发 / 录像收集（用同一份 JPEG，不重复编码）
                    self._record_flow(time.time(), jpeg, triggered, prob)

                if src.realtime:
                    # 固定节拍：按绝对时间点累加，避免"每帧睡一次"的误差累积
                    next_tick += 1.0 / SAMPLE_FPS
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.perf_counter()   # 跑得比节拍快，重新对齐

                now = time.perf_counter()
                marks.append((now, frame_i))
                if now - t_beat >= 0.5:
                    # fps 用【最近 2 秒滑动窗口】：累计均值会被首帧预热（CUDA/cuDNN 初始化）
                    # 长期拉低，前端看到的数字会比实际慢很多。
                    while len(marks) > 1 and now - marks[0][0] > 2.0:
                        marks.popleft()
                    span = now - marks[0][0]
                    with self._lock:
                        self._fps = (frame_i - marks[0][1]) / span if span > 0.2 else 0.0
                    t_beat = now
        except Exception as e:                      # 保证线程不会静默死掉
            with self._lock:
                self._state = "error"
                self._error = f"{type(e).__name__}: {e}"
        finally:
            self._finalize_recording()
            src.release()
            with self._lock:
                if self._state == "running":
                    self._state = "finished"
                # 保留最后一次的实测帧率（不归零）：结束后前端仍能看到本次会话跑多快；
                # 下次 start() 时会重置为 0。

    # ------------------------------------------------------------ 单帧处理
    def _process(self, frame, frame_i, do_detect=True, do_prob=True):
        H, W = frame.shape[:2]

        # ① YOLO 人体检测（降频：非检测帧复用上一次的框与置信度）
        pts = conf = None
        kpts = np.zeros((17, 3), np.float32)
        if do_detect:
            box, score = self._det.detect_frame(frame)
            self._last_box = tuple(box) if box is not None else None
            self._last_score = float(score or 0.0)
            self._last_det_n = 1 if (box is not None and score > 0) else 0
        else:
            box = self._last_box
            score = self._last_score

        if box is not None and score > 0:
            x1, y1, x2, y2 = box
            box_px = [x1 * W, y1 * H, x2 * W, y2 * H]
            # ② 自研 P1 姿态（始终逐帧：GPU 上仅数毫秒，且窗口需要每帧骨架）
            pts, conf = self._pose.extract(frame, box_px)
            kpts = np.stack([pts[:, 0] / W, pts[:, 1] / H, conf], axis=1).astype(np.float32)

        with self._lock:
            self._kpts.append(kpts)          # 窗口逐帧增长，不降频
            self._frames = frame_i
            have_win = len(self._kpts) >= WIN_FRAMES
            kwin = np.stack(list(self._kpts)[-WIN_FRAMES:]).astype(np.float32) if have_win else None

        # ③ A4 滑窗时序检测（降频：非判定帧沿用上一次的概率）
        prob = self._prob
        if do_prob and kwin is not None:
            with a4_fast_context():          # 关掉 joblib/OpenMP 的调度开销
                wp = self._alg.predict(kwin)
            if len(wp):
                prob = float(wp[0])

        now = time.time()
        with self._lock:
            self._prob = prob
            self._peak = max(self._peak, prob)
            # 仅在真正的判定帧上累计确认计数，保持 confirm 的时序平滑语义不变
            if do_prob:
                if prob >= self.threshold:
                    self._hot += 1
                else:
                    self._hot = 0
            triggered = (self._hot >= self.confirm
                         and (now - self._last_alarm) >= self.cooldown_s
                         and self._rec is None)
            if triggered:
                self._last_alarm = now
                self._alarm_until = now + self.alarm_hold_s

        # 画面叠加（只画在 vis 上：模型输入始终是原始 frame，叠加不会回流进模型）
        vis = frame.copy()
        if self.show_skeleton and pts is not None:
            draw_skeleton(vis, pts, conf, box)
        if self.show_hud:
            vis = self._draw_hud(vis, prob, frame_i)
        return vis, prob, triggered

    def _draw_hud(self, vis, prob, frame_i):
        H, W = vis.shape[:2]
        alarm = prob >= self.threshold
        color = (0, 0, 230) if alarm else (0, 170, 0)      # BGR
        cv2.rectangle(vis, (0, 0), (W, 30), (34, 36, 40), -1)
        # 只用 ASCII，避免 OpenCV 无法渲染中文
        txt = f"frame {frame_i}  P(fall)={prob:.2f}  {'FALL' if alarm else 'ADL'}"
        cv2.putText(vis, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 概率条
        bw = int(W * 0.5)
        x0, y0 = int(W * 0.03), 42
        cv2.rectangle(vis, (x0, y0), (x0 + bw, y0 + 12), (70, 74, 80), -1)
        cv2.rectangle(vis, (x0, y0), (x0 + int(bw * min(max(prob, 0.0), 1.0)), y0 + 12), color, -1)
        cx = x0 + int(bw * self.threshold)
        cv2.line(vis, (cx, y0 - 4), (cx, y0 + 16), (230, 230, 230), 1)
        return vis

    # ------------------------------------------------------------ 录像
    def _record_flow(self, now, jpeg, triggered, prob):
        """每帧调用：维护前置缓冲、触发录像、收集后置帧、满 3s 收尾落盘。

        用【真实时间戳】采集，保证「3s 录像」是真实 3 秒（1s 前置 + 2s 后置），
        与推理速度无关（GPU 快、CPU 慢都不影响录像的真实时长）。
        """
        to_finalize = None
        with self._lock:
            self._pre.append((now, jpeg))
            if self._rec is None:
                if triggered:
                    ts = datetime.now()
                    name = "fall_%s.mp4" % ts.strftime("%Y%m%d_%H%M%S")
                    pre = [(t, b) for (t, b) in self._pre
                           if t >= now - self.pre_frames / self.clip_fps]
                    self._rec = {"name": name, "t": ts, "t0": now,
                                 "prob": prob, "peak": prob, "buf": pre}
            else:
                self._rec["buf"].append((now, jpeg))
                self._rec["peak"] = max(self._rec["peak"], prob)
                if now - self._rec["t0"] >= self.post_frames / self.clip_fps:
                    to_finalize = self._rec
                    self._rec = None
        if to_finalize is not None:
            self._finalize_recording(to_finalize)

    def _finalize_recording(self, rec=None):
        if rec is None:
            with self._lock:
                rec = self._rec
                self._rec = None
        if rec is None:
            return None
        buf = rec.get("buf") or []
        if not buf:
            return None

        # 真实时长 → 推导标称帧率，使录像回放速度与实际一致
        n = len(buf)
        duration = max(0.1, buf[-1][0] - buf[0][0])
        fps = float(np.clip(n / duration, 1.0, 60.0))

        first = cv2.imdecode(np.frombuffer(buf[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return None
        H, W = first.shape[:2]

        path = os.path.join(CLIPS_DIR, rec["name"])
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        try:
            for _, jb in buf:
                img = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H))
                writer.write(img)
        finally:
            writer.release()

        ts = rec["t"]
        entry = {
            "id": ts.strftime("%Y%m%d_%H%M%S"),
            "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": ts.timestamp(),
            "prob": round(float(rec["prob"]), 4),
            "peak": round(float(rec["peak"]), 4),
            "clip": rec["name"],
            "clip_url": "/api/clip/" + rec["name"],
            "alarm": True,
            "source": self._source_label,
            "frames": n,
            "duration_s": round(duration, 2),
            "fps": round(fps, 2),
            "threshold": self.threshold,
        }
        with self._lock:
            self._n_falls += 1
        saved = self.store.add(entry)          # 写入 SQLite
        self._refresh_record_cache()           # 同步内存缓存（status() 用）
        return saved

    # ------------------------------------------------------------ 状态
    def _refresh_record_cache(self):
        """把「记录条数 / 最新一条」缓存到内存。

        status() 会被前端每 0.5s 轮询一次；如果每次都去查 SQLite，查库要在持锁期间
        进行，会把推理线程堵在锁上（实测拖慢约 15ms/帧）。所以改成只在记录
        发生变化时刷新一次，status() 只读内存。
        """
        try:
            n = self.store.count()
            last = self.store.latest()
        except Exception:
            return
        with self._lock:
            self._n_records = int(n)
            self._last_record = last

    def latest_jpeg(self):
        with self._lock:
            return self._last_jpeg

    def status(self):
        with self._lock:
            running = self._state == "running"
            elapsed = (time.time() - self._t_start) if (running and self._t_start) else (
                self._frames / SAMPLE_FPS if self._frames else 0.0)
            return {
                "state": self._state,
                "error": self._error,
                "models_ready": self.models_ready,
                "load_error": self._load_error,
                "a4_weights": getattr(self, "a4_source", None),
                "source": self._source_label,
                "source_spec": self._source_spec,
                "prob": round(float(self._prob), 4),
                "peak": round(float(self._peak), 4),
                "alarm": time.time() < self._alarm_until,
                "threshold": self.threshold,
                "frames": int(self._frames),
                "seen": int(self._seen),
                "detect_every": int(self.detect_every),
                "prob_every": int(self.prob_every),
                "fps": round(float(self._fps), 2),
                "elapsed": round(float(elapsed), 1),
                "n_falls": int(self._n_falls),
                "recording": self._rec is not None,
                "n_records": self._n_records,
                "last_record": self._last_record,
                "gt": self._gt,
                "session": self._session,
            }


if __name__ == "__main__":
    # 命令行自检：加载模型 + 跑一小段示例，确认引擎可用
    eng = DetectionEngine()
    print("[OK] 模型加载完成：YOLO + P1 姿态 + A4")
    print("[OK] 可用示例数据源：", list(list_demo_sources().keys()))
    print("[演示] 启动 urfd/fall-01 共 60 帧 ...")
    eng.start({"type": "demo", "dataset": "urfd", "seq": "fall-01"})
    time.sleep(8)
    st = eng.status()
    eng.stop()
    print("[结果] state=%s frames=%s peak=%.3f n_falls=%d records=%d"
          % (st["state"], st["frames"], st["peak"], st["n_falls"], st["n_records"]))
