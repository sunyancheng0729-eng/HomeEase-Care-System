# -*- coding: utf-8 -*-
"""
store.py —— 跌倒记录的持久化存储（SQLite）
================================================================
记录与报告数据统一存在 SQLite 里（systems/records/falls.db），
用 Python 标准库 sqlite3，无需额外依赖。

表结构 falls：
  id          INTEGER  主键（自增）
  fall_time   TEXT     跌倒时间（系统本地时间，YYYY-MM-DD HH:MM:SS）
  epoch       REAL     同一时间的时间戳（便于排序/筛选）
  prob        REAL     触发时的概率
  peak        REAL     该次事件峰值概率
  clip        TEXT     录像文件名（3 秒，位于 records/clips/）
  duration_s  REAL     录像时长（秒）
  frames      INTEGER  录像帧数
  fps         REAL     录像标称帧率
  threshold   REAL     触发时的判定阈值
  source      TEXT     输入源（摄像头 0 / urfd/fall-01 ...）
  alarm       INTEGER  报警标记（1 = 报警）
  created_at  TEXT     入库时间

线程安全：每次操作单独开连接 + 进程内 RLock，SQLite 自身也做文件级加锁。
"""
from __future__ import annotations

import os
import json
import sqlite3
import threading
from datetime import datetime

DB_NAME = "falls.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS falls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fall_time   TEXT    NOT NULL,
    epoch       REAL,
    prob        REAL,
    peak        REAL,
    clip        TEXT,
    duration_s  REAL,
    frames      INTEGER,
    fps         REAL,
    threshold   REAL,
    source      TEXT,
    alarm       INTEGER DEFAULT 1,
    created_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_falls_epoch ON falls(epoch DESC);
"""

_COLUMNS = ("fall_time", "epoch", "prob", "peak", "clip", "duration_s",
            "frames", "fps", "threshold", "source", "alarm", "created_at")


class RecordStore:
    def __init__(self, records_dir: str):
        os.makedirs(records_dir, exist_ok=True)
        self.db_path = os.path.join(records_dir, DB_NAME)
        self._lock = threading.RLock()
        self._init_db()

    # ------------------------------------------------------------ 基础
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, timeout=15)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": int(row["id"]),
            "time": row["fall_time"],
            "epoch": float(row["epoch"] or 0.0),
            "prob": float(row["prob"] or 0.0),
            "peak": float(row["peak"] or 0.0),
            "clip": row["clip"] or "",
            "clip_url": "/api/clip/" + (row["clip"] or ""),
            "duration_s": float(row["duration_s"] or 0.0),
            "frames": int(row["frames"] or 0),
            "fps": float(row["fps"] or 0.0),
            "threshold": float(row["threshold"] or 0.0),
            "source": row["source"] or "",
            "alarm": bool(row["alarm"]),
            "created_at": row["created_at"] or "",
        }

    # ------------------------------------------------------------ 增查删
    def add(self, entry: dict) -> dict:
        """写入一条跌倒记录，返回带数据库 id 的完整记录。"""
        with self._lock, self._conn() as c:
            cur = c.execute(
                f"INSERT INTO falls ({','.join(_COLUMNS)}) "
                f"VALUES ({','.join('?' * len(_COLUMNS))})",
                (entry.get("time"), entry.get("epoch"), entry.get("prob"),
                 entry.get("peak"), entry.get("clip"), entry.get("duration_s"),
                 entry.get("frames"), entry.get("fps"), entry.get("threshold"),
                 entry.get("source"), 1 if entry.get("alarm", True) else 0,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            rid = cur.lastrowid
            row = c.execute("SELECT * FROM falls WHERE id = ?", (rid,)).fetchone()
        return self._row_to_dict(row)

    def list(self, limit: int | None = None, newest_first: bool = True) -> list[dict]:
        order = "DESC" if newest_first else "ASC"
        sql = f"SELECT * FROM falls ORDER BY epoch {order}, id {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock, self._conn() as c:
            rows = c.execute(sql).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def latest(self) -> dict | None:
        rows = self.list(limit=1, newest_first=True)
        return rows[0] if rows else None

    def count(self) -> int:
        with self._lock, self._conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM falls").fetchone()[0])

    def delete(self, rid) -> dict | None:
        """删除一条，返回被删记录（用于同时删录像文件）；不存在返回 None。"""
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM falls WHERE id = ?", (rid,)).fetchone()
            if row is None:
                return None
            c.execute("DELETE FROM falls WHERE id = ?", (rid,))
        return self._row_to_dict(row)

    def clear(self) -> list[dict]:
        """清空全部，返回被删记录列表（用于同时删录像文件）。"""
        with self._lock:
            old = self.list(newest_first=False)
            with self._conn() as c:
                c.execute("DELETE FROM falls")
                try:
                    c.execute("DELETE FROM sqlite_sequence WHERE name='falls'")
                except sqlite3.Error:
                    pass
        return old

    def stats(self) -> dict:
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) n, MAX(peak) peak, MIN(epoch) first_epoch, "
                "MAX(epoch) last_epoch FROM falls").fetchone()
        return {"count": int(row["n"] or 0),
                "max_peak": float(row["peak"]) if row["peak"] is not None else None,
                "first_epoch": row["first_epoch"],
                "last_epoch": row["last_epoch"]}

    # ------------------------------------------------------------ 迁移
    def migrate_from_json(self, json_path: str) -> int:
        """把旧的 falls.json 一次性导入 SQLite（导入后重命名为 .bak）。"""
        if not os.path.isfile(json_path):
            return 0
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return 0
        if not isinstance(data, list) or not data:
            return 0
        n = 0
        for r in data:
            try:
                self.add({
                    "time": r.get("time"),
                    "epoch": r.get("epoch"),
                    "prob": r.get("prob"),
                    "peak": r.get("peak"),
                    "clip": r.get("clip"),
                    "duration_s": r.get("duration_s"),
                    "frames": r.get("frames"),
                    "fps": r.get("fps"),
                    "threshold": r.get("threshold"),
                    "source": r.get("source"),
                    "alarm": r.get("alarm", True),
                })
                n += 1
            except Exception:
                continue
        try:
            os.replace(json_path, json_path + ".bak")
        except OSError:
            pass
        return n
