# -*- coding: utf-8 -*-
"""最优跌倒检测算法组合（A2）核心包。

对外只暴露三条能力：
  · BestFallModel        —— RF(400)+XGB(400) 概率平均分类器（可存/取）
  · detect_offline       —— 整段离线检测（模拟数据 / 历史回放）
  · StreamingDetector    —— 持续输入的实时传感器流检测

超参见 `config.py`；本包不含任何实验期候选方法，只保留实验判定的最优路径。
"""
from . import config          # noqa: F401
from .model import BestFallModel, feature_importance          # noqa: F401
from .detector import StreamingDetector, detect_offline       # noqa: F401

__all__ = ["config", "BestFallModel", "feature_importance",
           "StreamingDetector", "detect_offline"]
