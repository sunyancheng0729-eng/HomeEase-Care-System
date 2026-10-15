# -*- coding: utf-8 -*-
"""最优组合专用预处理：重采样 60 Hz + Butterworth 4 阶低通 5 Hz（零相位、保重力）。

只保留**最优路径**，不带实验期的 6 种候选滤波器——系统部署不需要消融开关。
实现与 `00_common/preprocess.py` 中 `resample=60.0, filt="butter5"` 的分支逐行一致，
以保证「实验结论」与「上线的系统」是同一套数值行为。
"""
import numpy as np
from scipy import signal as sps

from . import config as C


# ---------------------------------------------------------------------------
# 滤波核
# ---------------------------------------------------------------------------
def butter5_sos(fs):
    """Butterworth 4 阶低通 5 Hz 的 SOS 系数（截止频率按 fs 上限保护）。"""
    fc = min(C.BUTTER_CUTOFF, 0.45 * fs)
    return sps.butter(C.BUTTER_ORDER, fc, btype="low", fs=fs, output="sos")


def _sos_filtfilt(sos, x):
    """零相位滤波（filtfilt），短信号自动收紧 padlen。

    注意：filtfilt 为非因果实现，离线/整段分析无相位滞后；
    实时流式推理时在**有限的滑动缓冲**上做同样运算，属工程近似，
    会引入约半个窗的"回看"依赖（系统内以缓冲长度封顶）。
    """
    n = len(x)
    if n < 12:
        return x.copy()
    padlen = min(3 * (2 * sos.shape[0] + 1), n - 1)
    return sps.sosfiltfilt(sos, x, padlen=padlen)


def apply_butter5(sig, fs):
    """对 [N] 或 [N,C] 信号做 butter5 零相位低通。"""
    x2 = sig if sig.ndim == 2 else sig[:, None]
    sos = butter5_sos(fs)
    out = np.empty_like(x2, dtype=np.float64)
    for c in range(x2.shape[1]):
        out[:, c] = _sos_filtfilt(sos, x2[:, c])
    return out if sig.ndim == 2 else out[:, 0]


# ---------------------------------------------------------------------------
# 重采样
# ---------------------------------------------------------------------------
def uniform_grid(t_s, fs, anchor=None):
    """定频时间网格。

    anchor=None 时以 t_s[0] 为起点（离线整段用）；
    anchor 给定时网格点对齐到绝对时刻 k/fs（流式用，保证多次评估共用同一网格）。
    """
    if anchor is None:
        return np.arange(t_s[0], t_s[-1] + 1e-12, 1.0 / fs)
    k0 = int(np.floor(anchor * fs))
    return np.arange(k0 / fs, t_s[-1] + 1e-9, 1.0 / fs)


def resample_signal(t_s, sig, fs, anchor=None):
    """按时间戳线性插值到定频 fs。sig: [N] 或 [N,C]。"""
    grid = uniform_grid(t_s, fs, anchor)
    if len(grid) == 0:
        grid = np.array([t_s[-1]], dtype=np.float64)
    if sig.ndim == 1:
        return grid, np.interp(grid, t_s, sig)
    cols = [np.interp(grid, t_s, sig[:, c]) for c in range(sig.shape[1])]
    return grid, np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# 规范化入口
# ---------------------------------------------------------------------------
def normalize(t_raw, a_raw, fs=C.RESAMPLE_FS, anchor=None):
    """原始 (t, [N,3]) → 规范信号。

    返回 dict：
      t [M] / ax,ay,az [M] / mag [M] / fs
    """
    t_raw = np.asarray(t_raw, np.float64)
    a_raw = np.asarray(a_raw, np.float64)
    if a_raw.ndim == 1:
        a_raw = a_raw[:, None]
    order = np.argsort(t_raw)
    t_raw, a_raw = t_raw[order], a_raw[order]

    t, sig = resample_signal(t_raw, a_raw, fs, anchor=anchor)
    sig = apply_butter5(sig, fs)
    ax, ay, az = sig[:, 0], sig[:, 1], sig[:, 2]
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    return {"t": t, "ax": ax, "ay": ay, "az": az, "mag": mag, "fs": float(fs)}


def read_acc_csv(path, has_header=False):
    """读取三轴加速度 CSV。

    支持两种列序（自动判别）：
      URFD 原生：t_ms, |a|, ax, ay, az   —— 5 列，第 1 列毫秒
      通用上传：t, ax, ay, az 或 t_ms, ax, ay, az —— 4 列，第 1 列时间
    返回 (t_s, a_raw[N,3])，时间统一换算为秒并从 0 起算。
    """
    import csv
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if not r:
                continue
            try:
                vals = [float(x) for x in r[:6] if str(x).strip() != ""]
            except ValueError:
                continue  # 跳过表头 / 脏行
            rows.append(vals)
    if not rows:
        raise ValueError("CSV 为空或无法解析")
    arr = np.asarray([r[:5] + [np.nan] * (5 - len(r)) for r in rows], np.float64)
    if arr.shape[1] >= 5:                       # URFD: t_ms, |a|, ax, ay, az
        t_ms, ax, ay, az = arr[:, 0], arr[:, 2], arr[:, 3], arr[:, 4]
        t_s = (t_ms - t_ms[0]) / 1000.0
    elif arr.shape[1] == 4:                     # t, ax, ay, az
        t0 = arr[:, 0]
        t_s = t0 - t0[0]
        if np.nanmedian(np.diff(t_s)) > 1.0:    # 看起来是毫秒
            t_s = t_s / 1000.0
        ax, ay, az = arr[:, 1], arr[:, 2], arr[:, 3]
    else:
        raise ValueError(f"无法识别的列数：{arr.shape[1]}")
    a = np.stack([ax, ay, az], axis=1)
    m = np.isfinite(a).all(axis=1) & np.isfinite(t_s)
    return t_s[m], a[m]
