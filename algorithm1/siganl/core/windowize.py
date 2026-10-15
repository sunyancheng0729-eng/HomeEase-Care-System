# -*- coding: utf-8 -*-
"""滑窗化与事件聚合（最优组合：窗长 2.0 s、步长 0.25 s、合并间隔 2 窗）。

步长 0.25 s 的物理依据：跌倒冲击持续 0.3~0.6 s，0.25 s 步长可保证任意
0.5 s 长的事件至少被 2 个窗口完整覆盖，不会漏掉冲击峰。
"""
import numpy as np

from . import config as C


def n_samples(fs, sec):
    return max(1, int(round(fs * sec)))


def make_windows(sig, fs, win_s=C.WIN_S, step_s=C.STEP_S):
    """[T,C] 或 [T] → windows [N,win_n,C], t0_idx [N]（窗口起始**样本下标**）。"""
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    win_n, step_n = n_samples(fs, win_s), n_samples(fs, step_s)
    T = len(sig)
    if T < win_n:
        return np.zeros((0, win_n, sig.shape[1]), np.float32), np.array([], np.int64)
    t0s = list(range(0, T - win_n + 1, step_n))
    wins = np.stack([sig[i:i + win_n] for i in t0s]).astype(np.float32)
    return wins, np.asarray(t0s, np.int64)


def windows_from_grid(t, sig, fs, win_s=C.WIN_S, step_s=C.STEP_S):
    """给定绝对时间网格 t 与信号 sig，按绝对时间切窗。

    返回 (wins [N,win_n,C], t_start[N], t_end[N])，时间均为绝对秒。
    用于流式检测：窗口端点严格落在 step_s 的整数倍上。
    """
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    win_n = n_samples(fs, win_s)
    step_n = n_samples(fs, step_s)
    if len(sig) < win_n:
        return np.zeros((0, win_n, sig.shape[1]), np.float32), np.array([]), np.array([])
    starts = list(range(0, len(sig) - win_n + 1, step_n))
    wins = np.stack([sig[i:i + win_n] for i in starts]).astype(np.float32)
    t_start = np.asarray([t[i] for i in starts], np.float64)
    return wins, t_start, t_start + win_s


def merge_hits(hit_t_end, win_s=C.WIN_S, gap_win=C.MERGE_GAP_WIN):
    """把命中窗口（按结束时刻排序）聚成事件。

    规则（与 00_common/windowize.py 的 MERGE_GAP_WIN 语义一致）：
      相邻命中窗口的**序号间隔** ≤ gap_win 视为同一事件；
      事件区间 = [首个命中窗起始, 末个命中窗结束]；
      报警时刻 = **最早命中窗口的结束时刻**（= 最早可报警时刻）。
    返回 list[dict]，字段：t_win_start / t_win_end / t_alarm / n_windows。
    """
    hit = np.sort(np.asarray(hit_t_end, np.float64))
    if len(hit) == 0:
        return []
    tol = 1e-6
    events, cur = [], [hit[0]]
    for x in hit[1:]:
        if (x - cur[-1]) <= gap_win * C.STEP_S + tol:
            cur.append(x)
        else:
            events.append(cur)
            cur = [x]
    events.append(cur)
    out = []
    for ev in events:
        t_first_end = float(ev[0])
        t_last_end = float(ev[-1])
        out.append({
            "t_win_start": t_first_end - win_s,
            "t_win_end": t_last_end,
            "t_alarm": t_first_end,          # 报警时刻 = 最早命窗结束
            "n_windows": int(len(ev)),
        })
    return out
