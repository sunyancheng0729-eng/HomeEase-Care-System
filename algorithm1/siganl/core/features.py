# -*- coding: utf-8 -*-
"""39 维统计特征（5 族）—— 与 `00_common/features.py` 数值逐一等价的精简版。

输入：单窗口 [win_n, 4] = (ax, ay, az, mag)，等间隔、已滤波。
族划分与维度：
  motion 13：模长统计(6) + SMA(1) + 三轴标准差(3) + 轴间相关(3)
  impact  8：峰值偏离 / 最小模长 / 失重占比 / 加加速度 / 冲击上升斜率 + 三轴绝对峰(3)
  orient  8：前/后半窗重力方向单位向量(6) + 半窗方向余弦 + 翻转角(度)
  freq    7：主频 / 谱质心 / 谱熵 / 谱带宽 + 三段能量占比
  calm    3：后半窗/全窗 的 标准差比、能量比、SMA 比（后冲击静止）
"""
import numpy as np

from . import config as C

G = C.G
FREEFALL_TH = C.FREEFALL_TH

GROUPS = C.FEATURE_GROUPS
GROUP_DESC = {
    "motion": "运动强度族（模长统计/三轴活动度/轴间相关）",
    "impact": "冲击·失重族（峰值偏离/自由落体/加加速度/上升斜率）",
    "orient": "姿态方向族（重力方向/窗内方向翻转角）",
    "freq": "频域族（主频/谱质心/谱熵/分频段能量）",
    "calm": "后冲击静止族（后半窗活动度占比）",
}


def _motion(w, fs):
    ax, ay, az, mag = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
    f = [
        float(np.mean(mag)), float(np.std(mag)), float(np.max(mag)),
        float(np.min(mag)), float(np.ptp(mag)), float(np.sqrt(np.mean(mag ** 2))),
        float(np.mean(np.abs([ax, ay, az]))),                 # SMA
        float(np.std(ax)), float(np.std(ay)), float(np.std(az)),
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(np.stack([ax, ay, az]))
    c = np.nan_to_num(c, nan=0.0)
    f += [float(c[0, 1]), float(c[0, 2]), float(c[1, 2])]
    return f


def _impact(w, fs):
    mag = w[:, 3]
    dt = 1.0 / max(fs, 1e-6)
    d = np.diff(mag) / dt
    peak_dev = float(np.max(np.abs(mag - G)))
    min_mag = float(np.min(mag))
    free_ratio = float(np.mean(mag < FREEFALL_TH))
    jerk_max = float(np.max(np.abs(d))) if len(d) else 0.0
    i_pk = int(np.argmax(np.abs(mag - G)))
    rise = float((mag[i_pk] - mag[0]) / max(i_pk * dt, dt))    # 冲击上升斜率 (g/s)
    return [peak_dev, min_mag, free_ratio, jerk_max, rise,
            float(np.max(np.abs(w[:, 0]))), float(np.max(np.abs(w[:, 1]))),
            float(np.max(np.abs(w[:, 2])))]


def _orient(w, fs):
    h = len(w) // 2
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros_like(v)
    m1 = unit(np.median(w[:h, :3], axis=0))
    m2 = unit(np.median(w[h:, :3], axis=0))
    cos = float(np.clip(np.dot(m1, m2), -1.0, 1.0))
    return [float(m1[0]), float(m1[1]), float(m1[2]),
            float(m2[0]), float(m2[1]), float(m2[2]),
            cos, float(np.degrees(np.arccos(cos)))]


def _freq(w, fs):
    mag = w[:, 3] - np.mean(w[:, 3])
    n = len(mag)
    if n < 8:
        return [0.0] * 7
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft(mag * win))
    fr = np.fft.rfftfreq(n, d=1.0 / fs)
    p = sp ** 2
    tot = p.sum() + 1e-12
    pn = p / tot
    peak_f = float(fr[int(np.argmax(sp[1:]) + 1)]) if len(sp) > 1 else 0.0
    centroid = float((fr * pn).sum())
    entropy = float(-(pn[pn > 0] * np.log(pn[pn > 0])).sum() / np.log(len(pn) + 1e-12))
    bw = float(np.sqrt(((fr - centroid) ** 2 * pn).sum()))
    def band(a, b):
        m = (fr >= a) & (fr < b)
        return float(pn[m].sum())
    return [peak_f, centroid, entropy, bw, band(0, 3), band(3, 10), band(10, 25)]


def _calm(w, fs):
    h = len(w) // 2
    mag = w[:, 3]
    s_all = float(np.std(mag)) + 1e-9
    s_tail = float(np.std(mag[h:]))
    e = mag ** 2
    e_all = float(e.mean()) + 1e-9
    e_tail = float(e[h:].mean())
    sma = np.abs(w[:, :3]).sum(axis=1)
    sma_all = float(sma.mean()) + 1e-9
    sma_tail = float(sma[h:].mean())
    return [s_tail / s_all, e_tail / e_all, sma_tail / sma_all]


_FUNCS = {"motion": _motion, "impact": _impact, "orient": _orient,
          "freq": _freq, "calm": _calm}

_NAMES = {
    "motion": ["mag_mean", "mag_std", "mag_max", "mag_min", "mag_ptp", "mag_rms",
               "sma", "std_ax", "std_ay", "std_az", "corr_xy", "corr_xz", "corr_yz"],
    "impact": ["peak_dev", "min_mag", "freefall_ratio", "jerk_max", "rise_slope",
               "max_abs_ax", "max_abs_ay", "max_abs_az"],
    "orient": ["dir1_x", "dir1_y", "dir1_z", "dir2_x", "dir2_y", "dir2_z",
               "half_cos", "half_angle_deg"],
    "freq": ["peak_freq", "spec_centroid", "spec_entropy", "spec_bw",
             "e_0_3Hz", "e_3_10Hz", "e_10_25Hz"],
    "calm": ["tail_std_ratio", "tail_energy_ratio", "tail_sma_ratio"],
}


def feature_names(groups=GROUPS):
    names = []
    for g in groups:
        names += _NAMES[g]
    return names


def window_feature(w, fs, groups=GROUPS):
    """单窗口 [win_n,4] → 特征向量 [D]。"""
    out = []
    for g in groups:
        out += _FUNCS[g](w, fs)
    return np.asarray(out, np.float64)


def extract(windows, fs, groups=GROUPS):
    """[N,win_n,4] → X [N,D]（nan/inf 置 0）。"""
    if len(windows) == 0:
        return np.zeros((0, len(feature_names(groups))), np.float64)
    X = np.stack([window_feature(w, fs, groups) for w in windows])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
