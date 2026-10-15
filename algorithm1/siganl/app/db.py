# -*- coding: utf-8 -*-
"""SQLite 持久化层。

两张表：
  fall_events  —— 每条跌倒记录一行（时间、来源、报警依据、阈值、模型版本…）
  fall_samples —— 该次跌倒的 3 s 加速度切片（每采样点一行，供报告画图）

数据库文件默认 `runtime/fall.db`，可用环境变量 `FALL_SYSTEM_DB` 覆盖。
"""
import os
import json
import time
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DB_PATH = os.environ.get("FALL_SYSTEM_DB") or os.path.join(ROOT, "runtime", "fall.db")

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS fall_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alarm_time  TEXT    NOT NULL,   -- 报警的系统时间（本地，毫秒精度）
    alarm_ts    REAL    NOT NULL,   -- 同上，epoch 秒
    source      TEXT    NOT NULL,   -- sensor（实时流）| sim（模拟数据）
    seq         TEXT,               -- 数据来源序列 / 上传文件名
    stream_alarm REAL,              -- 报警在数据流内的相对时刻（秒）
    score       REAL,               -- 报警窗口概率（最早命窗）
    score_max   REAL,
    score_mean  REAL,
    threshold   REAL,
    n_windows   INTEGER,            -- 命中窗口数
    t_win_start REAL,
    t_win_end   REAL,
    duration_s  REAL,               -- 该次数据总时长
    model       TEXT,               -- 模型版本标识
    note        TEXT,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS fall_samples (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fall_id INTEGER NOT NULL,
    idx     INTEGER NOT NULL,
    t_rel   REAL,                   -- 相对报警时刻（秒），-2.0 .. +1.0
    ax      REAL, ay REAL, az REAL, mag REAL,
    FOREIGN KEY (fall_id) REFERENCES fall_events(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_samples_fall ON fall_samples(fall_id, idx);
CREATE INDEX IF NOT EXISTS idx_events_time ON fall_events(alarm_ts DESC);
"""


def _ensure_dir():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)


@contextmanager
def conn():
    """每次调用开一个连接（SQLite 轻量，避免多线程共享句柄）。"""
    _ensure_dir()
    with _LOCK:
        c = sqlite3.connect(DB_PATH, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        try:
            yield c
            c.commit()
        finally:
            c.close()


def init_db():
    with conn() as c:
        c.executescript(SCHEMA)
    return DB_PATH


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------
def _now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], now.timestamp()


def _fmt_ts(ts):
    dt = datetime.fromtimestamp(float(ts))
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], dt.timestamp()


def insert_fall(event, model_name="", seq=None, duration_s=None, note=None):
    """写入一条跌倒记录 + 其 3 s 切片。返回新记录 id。

    **报警系统时间的口径**：优先用事件自带的 `wall_alarm`（命中窗口被评估、即系统
    最早可确认跌倒的那个时刻，含 0.25 s 零相位滤波延迟）。传感器流的事件要等
    「最后一次命中 + 1 s」才闭合落库，若用落库时刻当报警时间会整体偏晚 1 s 以上；
    模拟数据路径没有 wall_alarm，退化为写入时刻。
    """
    wa = event.get("wall_alarm")
    alarm_time, alarm_ts = (_fmt_ts(wa) if wa else _now())
    created_at = _now()[0]                    # 落库时刻（与报警时刻分开记，便于审计延迟）
    snip = event.get("snippet") or {}
    with conn() as c:
        cur = c.execute(
            """INSERT INTO fall_events
               (alarm_time, alarm_ts, source, seq, stream_alarm, score, score_max,
                score_mean, threshold, n_windows, t_win_start, t_win_end,
                duration_s, model, note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (alarm_time, alarm_ts, event.get("source", "unknown"), seq,
             event.get("stream_alarm", event.get("t_alarm")),
             event.get("score"), event.get("score_max"), event.get("score_mean"),
             event.get("threshold"), event.get("n_windows"),
             event.get("t_win_start"), event.get("t_win_end"),
             duration_s, model_name, note, created_at))
        fid = cur.lastrowid
        rows = []
        for i, (t, x, y, z, m) in enumerate(zip(snip.get("t", []), snip.get("ax", []),
                                                snip.get("ay", []), snip.get("az", []),
                                                snip.get("mag", []))):
            rows.append((fid, i, t, x, y, z, m))
        if rows:
            c.executemany(
                """INSERT INTO fall_samples (fall_id, idx, t_rel, ax, ay, az, mag)
                   VALUES (?,?,?,?,?,?,?)""", rows)
    return fid


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------
def list_falls(limit=200, source=None):
    q = "SELECT * FROM fall_events"
    args = []
    if source:
        q += " WHERE source = ?"
        args.append(source)
    q += " ORDER BY alarm_ts DESC LIMIT ?"
    args.append(int(limit))
    with conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def get_fall(fid):
    with conn() as c:
        row = c.execute("SELECT * FROM fall_events WHERE id = ?", (fid,)).fetchone()
        if row is None:
            return None
        ev = dict(row)
        srow = c.execute(
            """SELECT t_rel, ax, ay, az, mag FROM fall_samples
               WHERE fall_id = ? ORDER BY idx""", (fid,)).fetchall()
    ev["snippet"] = {
        "t": [r["t_rel"] for r in srow], "ax": [r["ax"] for r in srow],
        "ay": [r["ay"] for r in srow], "az": [r["az"] for r in srow],
        "mag": [r["mag"] for r in srow],
    }
    return ev


def delete_fall(fid):
    with conn() as c:
        c.execute("DELETE FROM fall_samples WHERE fall_id = ?", (fid,))
        cur = c.execute("DELETE FROM fall_events WHERE id = ?", (fid,))
        return cur.rowcount > 0


def clear_falls():
    with conn() as c:
        c.execute("DELETE FROM fall_samples")
        cur = c.execute("DELETE FROM fall_events")
        return cur.rowcount


def stats():
    with conn() as c:
        n = c.execute("SELECT COUNT(*) FROM fall_events").fetchone()[0]
        n_sensor = c.execute("SELECT COUNT(*) FROM fall_events WHERE source='sensor'").fetchone()[0]
        n_sim = c.execute("SELECT COUNT(*) FROM fall_events WHERE source='sim'").fetchone()[0]
        last = c.execute("SELECT alarm_time FROM fall_events ORDER BY alarm_ts DESC LIMIT 1").fetchone()
    return {"total": n, "sensor": n_sensor, "sim": n_sim,
            "last_alarm_time": last[0] if last else None, "db_path": DB_PATH}
