# -*- coding: utf-8 -*-
"""
a4_core.py —— A4 混合集成核心（与 main.py 同目录，供其 import）
================================================================

包含：滑窗、运动学特征提取、时序网络、A4 集成（fit + predict）。
独立成文件是为了让 train_a4.py 和 main.py 共享同一套类定义，
从而 pickle 序列化的类路径（a4_core.KinematicBase 等）能被两端一致解析。

仅依赖第三方包：numpy, torch, sklearn, xgboost。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN_FRAMES = 30
STEP_FRAMES = 8
IN_DIM = 17 * 3
NOSE = 0
SH_L, SH_R = 5, 6
HIP_L, HIP_R = 11, 12


def make_windows(kpts, win=WIN_FRAMES, step=STEP_FRAMES):
    T = len(kpts)
    if T < win:
        return np.zeros((0, win, 17, 3), np.float32), [], []
    t0s = list(range(0, T - win + 1, step))
    wins = np.stack([kpts[i:i + win] for i in t0s]).astype(np.float32)
    t1s = [i + win for i in t0s]
    return wins, t0s, t1s


def _window_labels(t0_s, iv, n_windows):
    labels = np.zeros(n_windows, np.float32)
    win_len_s = WIN_FRAMES / 15.0
    for j, w_start in enumerate(t0_s):
        w_end = w_start + win_len_s
        for a, b in iv:
            if w_start <= b and w_end >= a:
                labels[j] = 1.0
                break
    return labels


def _mid(kpts, i, j):
    a, b = kpts[:, i], kpts[:, j]
    c = np.minimum(a[:, 2], b[:, 2])
    return (a[:, :2] + b[:, :2]) / 2.0, c


def extract_window_features(win):
    k = win
    sh, _ = _mid(k, SH_L, SH_R)
    hp, _ = _mid(k, HIP_L, HIP_R)
    nose = k[:, NOSE]
    v = sh - hp
    norm = np.linalg.norm(v, axis=1) + 1e-6
    cos_ang = np.clip(-v[:, 1] / norm, -1, 1)
    sin_ang = np.sin(np.arccos(cos_ang))
    hip_y = hp[:, 1]
    nose_y = nose[:, 1]
    sh_y = sh[:, 1]

    def grad(y):
        return np.diff(y, prepend=y[:1])

    hip_vy = grad(hip_y)
    hip_acc = grad(hip_vy)
    f = [np.max(sin_ang), np.mean(sin_ang), np.std(sin_ang),
         np.min(norm), np.mean(norm),
         np.min(hip_y), np.max(hip_y), np.mean(hip_y), hip_y[-1] - hip_y[0],
         np.max(np.abs(hip_vy)), np.max(np.abs(hip_acc)),
         np.min(nose_y), np.max(np.abs(grad(nose_y))),
         np.min(sh_y), np.max(np.abs(grad(sh_y)))]
    xs = k[:, :, 0][k[:, :, 2] > 0.3]
    ys = k[:, :, 1][k[:, :, 2] > 0.3]
    if len(xs) > 2:
        hh = np.percentile(ys, 95) - np.percentile(ys, 5)
        ww = np.percentile(xs, 95) - np.percentile(xs, 5)
        f += [hh / (ww + 1e-6), ww, hh]
    else:
        f += [1.0, 0.0, 0.0]
    return np.asarray(f, np.float32)


class _LSTM(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


class _GRU(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(-1)


class _BiLSTM(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


class _TCN(nn.Module):
    def __init__(self, d_in=IN_DIM, hidden=64):
        super().__init__()
        self.c1 = nn.Conv1d(d_in, hidden, 5, padding=2)
        self.c2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.c1(x.transpose(1, 2)))
        h = F.relu(self.c2(h))
        h = h.mean(-1)
        return self.head(h).squeeze(-1)


MODELS = {"lstm": _LSTM, "gru": _GRU, "bilstm": _BiLSTM, "tcn": _TCN}


class KinematicBase:
    def __init__(self, clf):
        self.clf = clf

    def _feats(self, kpts):
        wins, _, _ = make_windows(kpts)
        if len(wins) == 0:
            return None
        return np.stack([extract_window_features(w) for w in wins]).astype(np.float32)

    def fit(self, items):
        X, y = _build_kinematic_xy(items)
        self.clf.fit(X, y)
        return self

    def predict(self, kpts):
        Fm = self._feats(kpts)
        if Fm is None:
            return np.zeros(0, np.float32)
        return self.clf.predict_proba(Fm)[:, 1].astype(np.float32)


class TemporalBase:
    def __init__(self, net_name, epochs=25, lr=1e-3, batch=128, device=None):
        self.net_name = net_name
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MODELS[net_name]().to(self.device)

    def fit(self, items):
        self.net.train()
        X, y = _build_temporal_xy(items, self.device)
        if len(X) == 0 or y.sum() == 0 or (y == 1).all():
            self._const = float(y.mean()) if len(y) else 0.0
            return self
        self._const = None
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        n = len(X)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                logits = self.net(X[idx])
                loss = F.binary_cross_entropy_with_logits(logits, y[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, kpts):
        wins, _, _ = make_windows(kpts)
        if len(wins) == 0:
            return np.zeros(0, np.float32)
        if getattr(self, "_const", None) is not None:
            return np.full(len(wins), self._const, np.float32)
        X = torch.from_numpy(wins.reshape(len(wins), WIN_FRAMES, IN_DIM)).float().to(self.device)
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(X)).cpu().numpy()
        return p.astype(np.float32)


def _build_kinematic_xy(items):
    X, y = [], []
    for kpts, t0_s, iv in items:
        wins, _, _ = make_windows(kpts)
        if len(wins) == 0:
            continue
        X.append(np.stack([extract_window_features(w) for w in wins]))
        y.append(_window_labels(t0_s, iv, len(wins)))
    if not X:
        return np.zeros((0, 18), np.float32), np.zeros(0, np.float32)
    return np.concatenate(X, 0).astype(np.float32), np.concatenate(y, 0).astype(np.float32)


def _build_temporal_xy(items, device):
    X, y = [], []
    for kpts, t0_s, iv in items:
        wins, _, _ = make_windows(kpts)
        if len(wins) == 0:
            continue
        X.append(wins.reshape(len(wins), WIN_FRAMES, IN_DIM))
        y.append(_window_labels(t0_s, iv, len(wins)))
    if not X:
        return torch.zeros((0, WIN_FRAMES, IN_DIM)), torch.zeros(0)
    X = torch.from_numpy(np.concatenate(X, 0)).float().to(device)
    y = torch.from_numpy(np.concatenate(y, 0)).float().to(device)
    return X, y


class HybridAlgorithm:
    """A4：多样化基学习器 + OOF Stacking + XGBoost 元学习器。"""
    def __init__(self, n_folds=3, temporal_epochs=25):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        self.n_folds = n_folds
        self.bases = [
            KinematicBase(RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=4,
                n_jobs=-1, random_state=2026)),
            KinematicBase(xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                eval_metric="logloss", random_state=2026, n_jobs=-1)),
            KinematicBase(LogisticRegression(C=1.0, max_iter=1000)),
            TemporalBase("lstm", epochs=temporal_epochs),
            TemporalBase("gru", epochs=temporal_epochs),
            TemporalBase("bilstm", epochs=temporal_epochs),
            TemporalBase("tcn", epochs=temporal_epochs),
        ]
        self.meta = None

    def fit(self, train_items):
        n = len(train_items)
        if n < 8:
            for b in self.bases:
                b.fit(train_items)
            self.meta = None
            return self

        n_meta = len(self.bases)
        oof_preds = [[] for _ in range(n)]
        labels = [[] for _ in range(n)]
        for bi, base in enumerate(self.bases):
            oof = self._oof_base(base, train_items)
            for i in range(n):
                if bi == 0:
                    labels[i] = _window_labels(train_items[i][1], train_items[i][2], len(oof[i]))
                oof_preds[i].append(oof[i])

        Xm = np.concatenate(
            [np.stack([oof_preds[i][j] for j in range(n_meta)], axis=1) for i in range(n)],
            axis=0).astype(np.float32)
        ym = np.concatenate(labels, axis=0).astype(np.float32)

        import xgboost as xgb
        if ym.sum() > 0 and (ym == 1).sum() < len(ym):
            self.meta = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                reg_lambda=1.0, eval_metric="logloss", random_state=2026)
            self.meta.fit(Xm, ym)
        else:
            self.meta = None

        for b in self.bases:
            b.fit(train_items)
        return self

    def _oof_base(self, base, items):
        n = len(items)
        oof = [None] * n
        for f in range(self.n_folds):
            test_idx = [i for i in range(n) if i % self.n_folds == f]
            train_idx = [i for i in range(n) if i % self.n_folds != f]
            base.fit([items[i] for i in train_idx])
            for i in test_idx:
                oof[i] = base.predict(items[i][0])
        return oof

    def predict(self, kpts):
        preds = [b.predict(kpts) for b in self.bases]
        if len(preds[0]) == 0:
            return preds[0]
        X = np.stack(preds, axis=1).astype(np.float32)
        if self.meta is None:
            return np.mean(X, axis=1).astype(np.float32)
        return self.meta.predict_proba(X)[:, 1].astype(np.float32)
