import contextlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from care.config import DB_PATH


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.lock = threading.RLock()
        with self.connection() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL,
                    level TEXT NOT NULL, state TEXT NOT NULL, channel TEXT NOT NULL,
                    dedup_key TEXT UNIQUE, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_time ON events(created_at);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sounds (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, mode TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY, event_id TEXT, created_at TEXT NOT NULL, status TEXT NOT NULL,
                    kind TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_at REAL NOT NULL DEFAULT 0,
                    payload TEXT NOT NULL, FOREIGN KEY(event_id) REFERENCES events(id)
                );
                CREATE TABLE IF NOT EXISTS coverage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, channel TEXT NOT NULL,
                    started_at TEXT NOT NULL, ended_at TEXT, state TEXT NOT NULL, reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, module TEXT NOT NULL,
                    action TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runtime_state (id INTEGER PRIMARY KEY, last_heartbeat TEXT NOT NULL);
            """)
            defaults = {
                "resident": {"name": "", "room": "客厅", "notes": ""},
                "contacts": [], "audio_consent": False, "exercise_target": 10,
                "escalation_seconds": 60, "webhook_url": "", "notification_test_at": None,
            }
            conn.execute("INSERT OR IGNORE INTO settings VALUES (1,?)", (encode(defaults),))

    @contextlib.contextmanager
    def connection(self):
        with self.lock:
            conn = sqlite3.connect(self.path, timeout=15)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def settings(self):
        with self.connection() as conn:
            return json.loads(conn.execute("SELECT value FROM settings WHERE id=1").fetchone()[0])

    def save_settings(self, value):
        with self.connection() as conn:
            conn.execute("UPDATE settings SET value=? WHERE id=1", (encode(value),))
            self._audit(conn, "settings", "updated", {"audio_consent": value["audio_consent"]})
        return value

    @staticmethod
    def _audit(conn, module, action, payload):
        conn.execute("INSERT INTO audit(at,module,action,payload) VALUES (?,?,?,?)", (now(), module, action, encode(payload)))

    def audit(self, module, action, payload):
        with self.connection() as conn:
            self._audit(conn, module, action, payload)

    def add_event(self, event, dedup_key=None):
        with self.connection() as conn:
            if dedup_key:
                row = conn.execute("SELECT payload FROM events WHERE dedup_key=?", (dedup_key,)).fetchone()
                if row:
                    return json.loads(row[0]), False
            event = {"id": uuid.uuid4().hex[:12], "created_at": now(), "revision": 1, **event}
            conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", (
                event["id"], event["created_at"], event["mode"], event["level"], event["state"],
                event["channel"], dedup_key, encode(event)))
            self._audit(conn, "events", "created", {"id": event["id"], "channel": event["channel"], "mode": event["mode"]})
        return event, True

    def event(self, event_id):
        with self.connection() as conn:
            row = conn.execute("SELECT payload FROM events WHERE id=?", (event_id,)).fetchone()
        if not row:
            raise KeyError(event_id)
        return json.loads(row[0])

    def update_event(self, event_id, mutate, revision=None):
        with self.connection() as conn:
            row = conn.execute("SELECT payload FROM events WHERE id=?", (event_id,)).fetchone()
            if not row:
                raise KeyError(event_id)
            event = json.loads(row[0])
            if revision is not None and event["revision"] != revision:
                raise ValueError("记录已更新，请刷新后重试")
            mutate(event)
            event["revision"] += 1
            conn.execute("UPDATE events SET level=?,state=?,payload=? WHERE id=?", (
                event["level"], event["state"], encode(event), event_id))
            self._audit(conn, "events", "updated", {"id": event_id, "revision": event["revision"]})
        return event

    def events(self, mode=None, state=None, limit=500, since=None):
        where, args = ["1=1"], []
        for column, value in (("mode", mode), ("state", state)):
            if value:
                where.append(f"{column}=?")
                args.append(value)
        if since:
            where.append("created_at>=?")
            args.append(since)
        args.append(limit)
        with self.connection() as conn:
            rows = conn.execute("SELECT payload FROM events WHERE " + " AND ".join(where) + " ORDER BY created_at DESC LIMIT ?", args).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_session(self, session):
        with self.connection() as conn:
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,payload=excluded.payload", (
                session["session_id"], session["created_at"], session["mode"], session["status"], encode(session)))
        return session

    def sessions(self, mode=None, since=None, limit=1000):
        with self.connection() as conn:
            rows = conn.execute("SELECT payload FROM sessions WHERE (? IS NULL OR mode=?) AND (? IS NULL OR created_at>=?) ORDER BY created_at DESC LIMIT ?", (
                mode, mode, since, since, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_sound(self, result, mode):
        result = {"id": uuid.uuid4().hex[:12], "created_at": now(), "mode": mode, **result}
        with self.connection() as conn:
            conn.execute("INSERT INTO sounds VALUES (?,?,?,?)", (result["id"], result["created_at"], mode, encode(result)))
        return result

    def sounds(self, limit=30):
        with self.connection() as conn:
            return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM sounds ORDER BY created_at DESC LIMIT ?", (limit,))]

    def notify(self, event_id, payload, kind="inapp"):
        notice = {"id": uuid.uuid4().hex[:12], "event_id": event_id, "created_at": now(), **payload}
        with self.connection() as conn:
            conn.execute("INSERT INTO notifications(id,event_id,created_at,status,kind,payload) VALUES (?,?,?,?,?,?)", (
                notice["id"], event_id, notice["created_at"], "queued", kind, encode(notice)))
        return notice

    def notifications(self, deliver=False, event_id=None, since=None, limit=100):
        with self.connection() as conn:
            if deliver:
                for row in conn.execute("SELECT id,payload FROM notifications WHERE kind='inapp' AND status='queued'").fetchall():
                    payload = {**json.loads(row["payload"]), "delivered_at": now()}
                    conn.execute("UPDATE notifications SET status='delivered',payload=? WHERE id=?", (encode(payload), row["id"]))
            rows = conn.execute("SELECT * FROM notifications WHERE (? IS NULL OR event_id=?) AND (? IS NULL OR created_at>=?) ORDER BY created_at DESC LIMIT ?", (event_id, event_id, since, since, limit)).fetchall()
        return [{**json.loads(r["payload"]), "status": r["status"], "kind": r["kind"], "attempts": r["attempts"]} for r in rows]

    def acknowledge_notification(self, notice_id):
        with self.connection() as conn:
            row = conn.execute("SELECT payload FROM notifications WHERE id=? AND kind='inapp'", (notice_id,)).fetchone()
            if row:
                payload = {**json.loads(row[0]), "seen_at": now()}
                conn.execute("UPDATE notifications SET status='seen',payload=? WHERE id=?", (encode(payload), notice_id))

    def recover_coverage(self):
        with self.connection() as conn:
            row = conn.execute("SELECT last_heartbeat FROM runtime_state WHERE id=1").fetchone()
            if row:
                stamp = row[0]
                conn.execute("UPDATE coverage SET ended_at=MAX(started_at,?) WHERE ended_at IS NULL", (stamp,))
                for channel in ("vision", "accel", "audio"):
                    conn.execute("INSERT INTO coverage(channel,started_at,ended_at,state,reason) VALUES (?,?,?,'unavailable','服务离线区间')", (channel, stamp, now()))

    def heartbeat(self):
        with self.connection() as conn:
            conn.execute("INSERT INTO runtime_state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET last_heartbeat=excluded.last_heartbeat", (now(),))

    def set_coverage(self, channel, state, reason):
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM coverage WHERE channel=? AND ended_at IS NULL", (channel,)).fetchone()
            if row and row["state"] == state and row["reason"] == reason:
                return
            timestamp = now()
            conn.execute("UPDATE coverage SET ended_at=? WHERE channel=? AND ended_at IS NULL", (timestamp, channel))
            conn.execute("INSERT INTO coverage(channel,started_at,state,reason) VALUES (?,?,?,?)", (channel, timestamp, state, reason))

    def coverage(self, since=None):
        with self.connection() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM coverage WHERE (? IS NULL OR ended_at IS NULL OR ended_at>=?) ORDER BY started_at DESC LIMIT 5000", (since, since))]

    def diagnostics(self):
        with self.connection() as conn:
            return [{**dict(r), "payload": json.loads(r["payload"])} for r in conn.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 100")]
