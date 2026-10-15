"""Clear fall/training history while the server is stopped; keep resident settings."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from care.config import RUNTIME


def clear_history(root):
    root = Path(root).resolve()
    address = root / "server-address.json"
    if address.exists():
        import psutil
        pid = json.loads(address.read_text(encoding="utf-8")).get("pid")
        if pid and psutil.pid_exists(pid):
            raise RuntimeError("Stop the server before clearing history")
    counts = {}
    with sqlite3.connect(root / "homeease.sqlite3") as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        counts["notifications"] = conn.execute(
            "DELETE FROM notifications WHERE event_id IN (SELECT id FROM events WHERE channel IN ('vision','accel'))"
        ).rowcount
        counts["fall_events"] = conn.execute("DELETE FROM events WHERE channel IN ('vision','accel')").rowcount
        counts["sessions"] = conn.execute("DELETE FROM sessions").rowcount
    records = root / "video_bridge" / "systems" / "records"
    database = records / "falls.db"
    if database.exists():
        with sqlite3.connect(database) as conn:
            counts["engine_falls"] = conn.execute("DELETE FROM falls").rowcount
    counts["evidence_files"] = 0
    for directory in (records / "clips", root / "playback"):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
                path.unlink()
                counts["evidence_files"] += 1
    return counts


if __name__ == "__main__":
    print(json.dumps(clear_history(RUNTIME), indent=2))
