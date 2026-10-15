# -*- coding: utf-8 -*-
"""最优组合分类器：RF(400) + XGB(400) 概率平均。

与 `04_alg_ml/src/ml_algorithm.py`（A2）参数完全一致：
  RF  : n_estimators=400, max_depth=10, min_samples_leaf=2, class_weight='balanced'
  XGB : n_estimators=400, max_depth=5, lr=0.06, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=(负/正)
两者取 predict_proba 的**算术平均**作为窗口跌倒概率。

模型产物（RF + XGB 两个已 fit 的估计器）用 joblib 序列化到 model/best_model.pkl。
"""
import os
import json
import numpy as np

from . import config as C
from . import features as F


class BestFallModel:
    """A2 最优组合：RF + XGB 概率平均。"""

    name = "A2_RF+XGB_avg"

    def __init__(self, fs=C.RESAMPLE_FS, seed=C.TRAIN_SEED):
        self.fs = float(fs)
        self.seed = int(seed)
        self.models = {}
        self.meta = {}

    # ---------------------------------------------------------------- 训练
    def _build(self, spw):
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        m = {}
        rf = dict(C.RF_PARAMS); rf.update(random_state=self.seed, n_jobs=-1)
        m["rf"] = RandomForestClassifier(**rf)
        xg = dict(C.XGB_PARAMS); xg.update(scale_pos_weight=spw, random_state=self.seed,
                                           n_jobs=-1, eval_metric="logloss", verbosity=0)
        m["xgb"] = XGBClassifier(**xg)
        return m

    def set_inference_jobs(self, n_jobs=C.INFER_JOBS):
        """固定推理期线程数。

        流式检测每次只评估**一个**窗口，`n_jobs=-1` 只会为单样本创建/同步线程池
        （实测单窗 ≈100 ms，绝大部分是调度开销；RF 400 树 22 ms + XGB 0.4 ms 才是
        真实计算量）。固定为单线程后单窗 ≈23 ms，而预测值是确定性的，**结果逐位不变**。
        """
        for m in self.models.values():
            try:
                if hasattr(m, "n_jobs"):
                    m.n_jobs = n_jobs
                if hasattr(m, "nthread"):
                    m.nthread = n_jobs
                booster = getattr(m, "get_booster", None)
                if callable(booster):
                    booster().set_param({"nthread": int(n_jobs)})
            except Exception:                                  # pragma: no cover
                pass
        self.infer_n_jobs = int(n_jobs)
        return self

    def fit(self, wins, labels, fs=None):
        """wins [N,win_n,4]、labels [N] → self。"""
        if fs is not None:
            self.fs = float(fs)
        y = np.asarray(labels, np.int64)
        X = F.extract(wins, self.fs)
        spw = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        self.models = self._build(spw)
        for m in self.models.values():
            m.fit(X, y)
        self.set_inference_jobs()          # 训练用满核，推理收敛到单线程
        self.meta = {
            "algorithm": self.name,
            "fs": self.fs,
            "seed": self.seed,
            "n_features": int(X.shape[1]),
            "feature_groups": list(C.FEATURE_GROUPS),
            "feature_names": F.feature_names(),
            "n_train_windows": int(len(y)),
            "pos_ratio": float((y == 1).mean()) if len(y) else 0.0,
            "scale_pos_weight": float(spw),
            "config": {
                "resample_fs": C.RESAMPLE_FS, "filter": C.FILTER_METHOD,
                "butter_order": C.BUTTER_ORDER, "butter_cutoff": C.BUTTER_CUTOFF,
                "win_s": C.WIN_S, "step_s": C.STEP_S, "merge_gap_win": C.MERGE_GAP_WIN,
                "rf": C.RF_PARAMS, "xgb": C.XGB_PARAMS,
            },
        }
        return self

    # ---------------------------------------------------------------- 推理
    def predict_proba(self, wins):
        """wins [N,win_n,4] → 平均跌倒概率 [N]。空输入返回空数组。"""
        if len(wins) == 0 or not self.models:
            return np.zeros(len(wins), np.float64)
        X = F.extract(wins, self.fs)
        probs = [m.predict_proba(X)[:, 1] for m in self.models.values()]
        return np.mean(probs, axis=0)

    # ------------------------------------------------------------ 持久化
    def save(self, path):
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"models": self.models, "meta": self.meta}, path)

    @classmethod
    def load(cls, path):
        import joblib
        blob = joblib.load(path)
        obj = cls(fs=blob.get("meta", {}).get("fs", C.RESAMPLE_FS))
        obj.models = blob["models"]
        obj.meta = blob.get("meta", {})
        obj.set_inference_jobs()           # 载入即固定推理线程
        return obj


def feature_importance(model):
    """RF 与 XGB 的归一化重要性平均，返回 [(name, value)] 降序。"""
    names = F.feature_names()
    imp = []
    if "rf" in model.models:
        imp.append(np.asarray(model.models["rf"].feature_importances_, np.float64))
    if "xgb" in model.models:
        imp.append(np.asarray(model.models["xgb"].feature_importances_, np.float64))
    if not imp:
        return []
    v = np.mean(imp, axis=0)
    v = v / (v.sum() + 1e-12)
    return sorted(zip(names, v.tolist()), key=lambda kv: -kv[1])
