# -*- coding: utf-8 -*-
"""
训练并保存 A4 混合集成权重（一次性脚本）
==========================================

在完整统一池（URFD + MCF + Le2i = 223 序列）上训练 A4，
把训练好的基学习器 + 元学习器 pickle 保存到 standalone/a4_weights.pkl。
之后 main.py 直接加载 a4_weights.pkl，无需每次重新训练。

用法:
  E:/anaconda3/python.exe train_a4.py
"""
import os
import sys
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for _p in (os.path.join(ROOT, "00_common"),
           os.path.join(ROOT, "02_pose_cnn"),
           os.path.join(ROOT, "01_detect_yolo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# 关键：import 同目录 a4_core，让 pickle 存的类路径是 a4_core.*，
# 与 main.py 加载时（也 import a4_core）完全一致。
from a4_core import HybridAlgorithm, make_windows

from windowize import make_windows as _mw  # noqa: F401  (仅确保窗口常量一致，实际用 a4_core 的)
from splits import unified_pool
from pose_extract import extract_skeleton_sequence

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}


def _gt_intervals(dataset, seq):
    import urfd_loader, mc_loader, le2i_loader
    if dataset == "urfd":
        return urfd_loader.fall_intervals_s(seq)
    if dataset == "mc":
        return mc_loader.fall_intervals_s(seq)
    return le2i_loader.fall_intervals_s(seq)


def build_train_items():
    items = []
    for d, s, *_ in unified_pool():
        dd = extract_skeleton_sequence(d, s)   # 读骨架缓存（YOLO+P1 真实产出）
        kpts = dd["kpts"]
        fidx = dd["frame_idx"]
        fps = SRC_FPS[d]
        wins, t0s, t1s = make_windows(kpts)
        if len(t0s) == 0:
            continue
        t0_s = fidx[t0s].astype(np.float64) / fps
        iv = _gt_intervals(d, s)
        items.append((kpts, t0_s, iv))
    return items


def main():
    print("[1/2] 构建训练数据（读取骨架缓存）...", flush=True)
    items = build_train_items()
    print(f"      共 {len(items)} 条序列", flush=True)

    print("[2/2] 训练 A4 混合集成 ...", flush=True)
    alg = HybridAlgorithm()
    alg.fit(items)

    out = os.path.join(HERE, "a4_weights.pkl")
    with open(out, "wb") as f:
        pickle.dump({
            "bases": alg.bases,
            "meta": alg.meta,
        }, f)
    print(f"[完成] A4 权重已保存 → {out}", flush=True)


if __name__ == "__main__":
    main()
