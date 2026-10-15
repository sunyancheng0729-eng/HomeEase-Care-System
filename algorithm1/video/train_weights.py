# -*- coding: utf-8 -*-
"""
train_weights.py —— 在【当前 Python 环境】重新导出 A4 权重
================================================================
为什么需要它？
  A4 的基学习器里有 sklearn / xgboost 模型，pickle 反序列化对这两个库的
  版本很敏感。standalone/a4_weights.pkl 是在另一个环境（sklearn 1.5.1 /
  xgboost 2.1.1）里导出的，换到本环境（如 animedev：sklearn 1.9 /
  xgboost 3.x）可能加载失败。

  所以这里用【当前环境】重新训练一次 A4，导出到 systems/a4_weights.pkl。
  engine.py 会优先加载这个文件（找不到才回退 standalone 的）。

训练数据用的是之前已经用「YOLO 检测 + 自研 P1 姿态」推理好的骨架缓存，
不需要重新跑检测（节省大量时间），模型结构与超参与论文中的 A4 完全一致。

用法：
  E:/anaconda3/envs/animedev/python.exe train_weights.py
"""
import os
import sys
import time
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))               # systems/
ROOT = os.path.abspath(os.path.join(HERE, ".."))                 # 项目根
STANDALONE = os.path.join(ROOT, "standalone")
for _p in (STANDALONE,
           os.path.join(ROOT, "00_common"),
           os.path.join(ROOT, "02_pose_cnn"),
           os.path.join(ROOT, "01_detect_yolo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# 与 main.py / engine.py 用同一份类定义（pickle 类路径统一为 a4_core.*）
from a4_core import HybridAlgorithm, make_windows
from splits import unified_pool
from pose_extract import extract_skeleton_sequence

SRC_FPS = {"urfd": 30.0, "mc": 30.0, "le2i": 25.0}
OUT_PATH = os.path.join(HERE, "a4_weights.pkl")


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
        dd = extract_skeleton_sequence(d, s)      # 读骨架缓存
        kpts = dd["kpts"]
        fidx = dd["frame_idx"]
        fps = SRC_FPS[d]
        wins, t0s, t1s = make_windows(kpts)
        if len(t0s) == 0:
            continue
        t0_s = fidx[t0s].astype(np.float64) / fps
        items.append((kpts, t0_s, _gt_intervals(d, s)))
    return items


def main():
    t0 = time.time()
    print("[1/2] 构建训练数据（读取骨架缓存 YOLO+P1 的真实产出）...", flush=True)
    items = build_train_items()
    print(f"      共 {len(items)} 条序列，用时 {time.time()-t0:.1f}s", flush=True)

    print("[2/2] 训练 A4 混合集成（7 基学习器 OOF Stacking + XGBoost 元学习器）...",
          flush=True)
    alg = HybridAlgorithm()
    alg.fit(items)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({"bases": alg.bases, "meta": alg.meta}, f)

    # 立刻回读验证，确保本环境能加载
    with open(OUT_PATH, "rb") as f:
        back = pickle.load(f)
    print(f"[完成] 权重已保存 → {OUT_PATH}")
    print(f"       基学习器 {len(back['bases'])} 个，元学习器 "
          f"{'已训练' if back['meta'] is not None else '无'}，"
          f"总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
