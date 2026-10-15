# -*- coding: utf-8 -*-
"""跌倒检测器：把「最优组合」封装成两种可直接上线的调用形态。

  ① detect_offline(t, a, model)  —— 整段离线检测（模拟数据输入 / 历史回放）
     严格复现 `experiments/exp03_alg.py` 的评测口径：整段重采样+滤波 → 全窗滑窗
     → 逐窗概率 → 阈值命中 → 事件聚合。

  ② StreamingDetector          —— 持续输入的传感器实时流
     维护 STREAM_BUFFER_S 秒滑动原始缓冲；每当时间轴前进一个 step（0.25 s）
     就评估一个「结束于当前时刻」的 2 s 窗口；命中则进入事件状态机，
     命中间断超过合并间隔且缓冲已覆盖报警后 1 s 时，事件闭合 → 落库。

**实时性说明**：最优组合的滤波是零相位（非因果）filtfilt，不能无延迟地用于
实时流。本实现对每个窗口额外取 FILTER_PAD_S=0.25 s 的前后上下文、并把评估
推迟 0.25 s，使流式结果与离线整段滤波几乎一致，代价是 0.25 s 固定检测延迟。

两级输出统一为 FallEvent 字典，含报告所需的 3 s 加速度切片。
"""
import time
import numpy as np

from . import config as C
from . import preprocess as P
from . import windowize as W

PAD = C.FILTER_PAD_S


# ---------------------------------------------------------------------------
# 报告切片
# ---------------------------------------------------------------------------
def snippet_from_signal(t_src, ax, ay, az, t_alarm, fs=C.RESAMPLE_FS,
                        pre=C.REPORT_PRE_S, post=C.REPORT_POST_S):
    """取报警时刻前后 (pre+post) 秒的**已滤波**加速度曲线，可直接画图。

    实现：先在 [t_alarm-(pre+pad), t_alarm+(post+pad)] 上重采样并做同款 butter5
    零相位滤波（与检测口径一致），再取中间 [t_alarm-pre, t_alarm+post] 段输出。
    t 为相对报警时刻的秒；mag = sqrt(ax²+ay²+az²)。
    """
    t_src = np.asarray(t_src, np.float64)
    if len(t_src) < 12:
        return {"t": [], "ax": [], "ay": [], "az": [], "mag": []}
    lo = max(t_alarm - pre - PAD, float(t_src[0]))
    hi = min(t_alarm + post + PAD, float(t_src[-1]))
    seg = np.arange(lo, hi + 1e-9, 1.0 / fs)
    if len(seg) < 12:
        seg = np.linspace(lo, hi, 12)
    A = np.stack([np.interp(seg, t_src, np.asarray(v, np.float64))
                  for v in (ax, ay, az)], axis=1)
    A = P.apply_butter5(A, fs)
    keep = (seg >= t_alarm - pre - 1e-9) & (seg <= t_alarm + post + 1e-9)
    rel = seg[keep] - t_alarm
    X, Y, Z = A[keep, 0], A[keep, 1], A[keep, 2]
    Mag = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    r = lambda v: [round(float(x), 5) for x in v]
    return {"t": [round(float(x), 4) for x in rel],
            "ax": r(X), "ay": r(Y), "az": r(Z), "mag": r(Mag)}


# ---------------------------------------------------------------------------
# 离线整段检测
# ---------------------------------------------------------------------------
def detect_offline(t_raw, a_raw, model, threshold=C.ALARM_THRESHOLD, fs=C.RESAMPLE_FS):
    """整段检测。返回 (events, info)。

    events: list[dict] —— 每个元素即一条可入库的跌倒记录（含 snippet）。
    info  : {'n_windows','probs'(list),'t_end'(list),'duration_s','n_hits'}
    """
    if len(a_raw) < 10:
        return [], {"n_windows": 0, "probs": [], "t_end": [], "duration_s": 0.0, "n_hits": 0}

    norm = P.normalize(t_raw, a_raw, fs=fs)
    sig = np.stack([norm["ax"], norm["ay"], norm["az"], norm["mag"]], axis=1)
    wins, t0_idx = W.make_windows(sig, fs)
    if len(wins) == 0:
        return [], {"n_windows": 0, "probs": [], "t_end": [],
                    "duration_s": float(norm["t"][-1] - norm["t"][0]), "n_hits": 0}

    probs = model.predict_proba(wins)
    t_end = np.asarray([norm["t"][i + wins.shape[1] - 1] for i in t0_idx], np.float64)
    hit_mask = probs >= threshold
    hit_t, hit_p = t_end[hit_mask], probs[hit_mask]

    events = []
    for ev in W.merge_hits(hit_t):
        m = (hit_t >= ev["t_win_start"] - 1e-9) & (hit_t <= ev["t_win_end"] + 1e-9)
        sc = hit_p[m]
        events.append(_make_event(
            t_alarm=ev["t_alarm"], t_win_start=ev["t_win_start"], t_win_end=ev["t_win_end"],
            n_windows=ev["n_windows"], scores=sc, threshold=threshold,
            snippet=snippet_from_signal(norm["t"], norm["ax"], norm["ay"], norm["az"],
                                        ev["t_alarm"], fs=fs),
            source="sim",
        ))
    info = {"n_windows": int(len(wins)), "probs": [round(float(x), 5) for x in probs],
            "t_end": [round(float(x), 4) for x in t_end],
            "duration_s": float(norm["t"][-1] - norm["t"][0]), "n_hits": int(hit_mask.sum())}
    return events, info


def _make_event(t_alarm, t_win_start, t_win_end, n_windows, scores, threshold, snippet, source):
    sc = np.asarray(scores, np.float64)
    return {
        "source": source,
        "t_alarm": float(t_alarm),
        "t_win_start": float(t_win_start),
        "t_win_end": float(t_win_end),
        "n_windows": int(n_windows),
        "score": float(sc[0]) if len(sc) else 0.0,        # 最早命窗概率（报警依据）
        "score_max": float(sc.max()) if len(sc) else 0.0,
        "score_mean": float(sc.mean()) if len(sc) else 0.0,
        "threshold": float(threshold),
        "snippet": snippet,
    }


# ---------------------------------------------------------------------------
# 流式检测器
# ---------------------------------------------------------------------------
class StreamingDetector:
    """持续输入的加速度流检测器（传感器模式）。

    用法：
        det = StreamingDetector(model)
        for t, ax, ay, az in sensor_stream:      # 逐样本喂入
            for ev in det.push(t, ax, ay, az):
                save(ev)                          # ev 为闭合的跌倒事件
        for ev in det.flush():                    # 流结束时收尾
            save(ev)
    """

    def __init__(self, model, threshold=C.ALARM_THRESHOLD, fs=C.RESAMPLE_FS):
        self.model = model
        self.threshold = float(threshold)
        self.fs = float(fs)
        self.win_n = W.n_samples(fs, C.WIN_S)
        self.win_span = (self.win_n - 1) / self.fs   # 窗内首末样本的时间跨度
        self.step_s = C.STEP_S
        self.merge_gap = C.MERGE_GAP_WIN * C.STEP_S
        self.pad_s = PAD
        # 事件闭合延迟：保证报警后 post 秒的数据已到达（供报告切片滤波）
        self.close_delay = max(self.merge_gap, C.REPORT_POST_S)
        # 原始滑动缓冲
        self._t = []
        self._a = []
        self.t0 = None
        self.t_now = None
        self.next_eval = None                  # 下一个待评估窗口的结束时刻
        self.cur = None                        # 当前未闭合事件
        self.pending_alarms = []               # 命中即刻产生的报警（供前端实时提示）
        self.recent = []                       # 最近若干 (t_end, prob) 供前端显示
        self.log = []                          # 全部窗口评估日志（供前端画完整概率曲线）
        self.log_base = 0                      # log 已丢弃的前缀长度（绝对下标偏移）
        self.last_prob = None
        self.n_eval = 0

    # ------------------------------------------------------------------ 入
    def push(self, t, ax, ay, az):
        """喂入一个样本（t 为秒，单调递增）。返回本次**新闭合**的事件列表。"""
        t = float(t)
        if self.t0 is None:
            self.t0 = t
            self.next_eval = t + self.win_span   # 与离线「按样本索引」的窗口端点严格对齐
        if self.t_now is not None and t < self.t_now:
            t = self.t_now                       # 防回退
        self.t_now = t
        self._t.append(t)
        self._a.append((float(ax), float(ay), float(az)))

        out = []
        guard = 0
        while (self.next_eval is not None and
               self.next_eval + self.pad_s <= t + 1e-9 and guard < 1000):
            guard += 1
            self._eval_window(self.next_eval)
            self.next_eval += self.step_s
        ev = self._maybe_close()
        if ev is not None:
            out.append(ev)
        self._trim()
        return out

    # -------------------------------------------------------------- 内部
    def _eval_window(self, t_eval):
        """评估结束于 t_eval 的 2 s 窗口。

        正常流式路径下，`push` 只在 `t_eval + pad_s` 已到达时才调用本函数，因此
        后向上下文恒为完整的 pad_s；`flush` 补评尾部窗口时后向上下文少于 pad_s，
        这里统一按 `min(t_eval + pad_s, t_now)` 取（够多少用多少），避免流最后一
        个窗口被丢弃而漏报。
        """
        if not self._t:
            return
        if self._t[0] > t_eval - self.win_span + 1e-9:
            return                               # 2 s 窗口所需的样本尚未到齐
        hi_use = min(t_eval + self.pad_s, self.t_now)
        if hi_use < t_eval - 1e-9:
            return
        # 前向上下文能拿多少拿多少（流刚开始时会略短于 pad_s）
        ctx_lo = max(t_eval - self.win_span - self.pad_s, self._t[0])
        seg_t, seg_a = self._slice(ctx_lo, hi_use)
        if len(seg_t) < 12:
            return
        n_before = max(0, int(round((t_eval - self.win_span - ctx_lo) * self.fs)))
        n_after = int(round((hi_use - t_eval) * self.fs))
        grid = t_eval + np.arange(-(self.win_n - 1 + n_before), n_after + 1) / self.fs
        A = np.stack([np.interp(grid, seg_t, seg_a[:, c]) for c in range(3)], axis=1)
        A = P.apply_butter5(A, self.fs)
        w3 = A[n_before:n_before + self.win_n]
        win = np.concatenate([w3, np.linalg.norm(w3, axis=1)[:, None]], axis=1)
        win = win[None, :, :].astype(np.float32)
        p = float(self.model.predict_proba(win)[0])
        self.last_prob = p
        self.n_eval += 1
        rec = {"t": round(float(t_eval - self.t0), 3), "p": round(p, 4)}
        self.recent.append(rec)
        if len(self.recent) > 400:
            self.recent = self.recent[-400:]
        self.log.append(rec)
        if len(self.log) > 20000:                # 长时运行保护：只保留最近 1 万个窗口
            self.log = self.log[10000:]
            self.log_base += 10000

        if p >= self.threshold:
            if self.cur is None:
                self.cur = {"t_alarm": t_eval, "last_hit": t_eval,
                            "t_win_start": t_eval - self.win_span, "n": 0,
                            "scores": [], "wall_alarm": time.time()}
                # 命中即刻报警（实时输出），记录待事件闭合后再落库
                self.pending_alarms.append({
                    "t_alarm": float(t_eval), "wall_alarm": self.cur["wall_alarm"],
                    "score": p, "threshold": self.threshold,
                    "t_win_start": t_eval - self.win_span,
                })
            self.cur["last_hit"] = t_eval
            self.cur["n"] += 1
            self.cur["scores"].append(p)
            self.cur["t_win_end"] = t_eval

    def _maybe_close(self):
        """命中中断超过闭合延迟 → 闭合事件。"""
        c = self.cur
        if c is None:
            return None
        if self.t_now >= c["last_hit"] + self.close_delay - 1e-9:
            ev = self._finalize(c, truncated=False)
            self.cur = None
            return ev
        return None

    def _finalize(self, c, truncated):
        t_alarm = c["t_alarm"]
        seg_t, seg_a = self._slice(t_alarm - (C.REPORT_PRE_S + self.pad_s),
                                  t_alarm + C.REPORT_POST_S + self.pad_s)
        if len(seg_t) >= 12:
            snip = snippet_from_signal(seg_t, seg_a[:, 0], seg_a[:, 1], seg_a[:, 2],
                                       t_alarm, fs=self.fs)
        else:
            snip = {"t": [], "ax": [], "ay": [], "az": [], "mag": []}
        ev = _make_event(
            t_alarm=t_alarm, t_win_start=c["t_win_start"],
            t_win_end=c.get("t_win_end", c["last_hit"]), n_windows=c["n"],
            scores=c["scores"], threshold=self.threshold, snippet=snip, source="sensor",
        )
        ev["wall_alarm"] = c.get("wall_alarm")
        ev["truncated"] = bool(truncated)
        ev["stream_alarm"] = round(float(t_alarm - self.t0), 3)
        return ev

    def flush(self, complete_tail=True):
        """流结束：补评尾部窗口 + 把未闭合事件收尾输出。

        `complete_tail=True` 会把「结束时刻 ≤ 当前时刻」但尚未评估的尾部窗口补齐
        （正常路径因等待 pad_s 后向上下文而剩下最后 0.25 s 的窗口），使一次完整
        回放的结果与离线整段检测一致 —— 否则恰好在录制末尾发生的跌倒会漏报。
        """
        out = []
        if complete_tail and self.next_eval is not None:
            while self.next_eval <= self.t_now + 1e-9:
                self._eval_window(self.next_eval)
                self.next_eval += self.step_s
        if self.cur is not None:
            out.append(self._finalize(self.cur, truncated=True))
            self.cur = None
        return out

    def pop_alarms(self):
        """取出并清空「命中即刻」报警队列（实时报警用）。"""
        out = self.pending_alarms
        self.pending_alarms = []
        return out

    # ------------------------------------------------------------- 缓冲
    def _slice(self, lo, hi):
        t = np.asarray(self._t, np.float64)
        a = np.asarray(self._a, np.float64)
        m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        return t[m], a[m]

    def _trim(self):
        if self.t_now is None:
            return
        lo = self.t_now - C.STREAM_BUFFER_S
        i = 0
        for i in range(len(self._t)):
            if self._t[i] >= lo:
                break
        if i > 0:
            self._t = self._t[i:]
            self._a = self._a[i:]

    # ------------------------------------------------------------- 状态
    def status(self):
        return {
            "t_stream": round(float(self.t_now - self.t0), 3) if self.t_now is not None else 0.0,
            "last_prob": round(float(self.last_prob), 4) if self.last_prob is not None else None,
            "alarming": self.cur is not None,
            "n_eval": self.n_eval,
            "n_samples": len(self._t),
        }
