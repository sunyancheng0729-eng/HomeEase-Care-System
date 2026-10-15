import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("HOMEEASE_SYSTEM_DATA", str(ROOT / "runtime"))).resolve()
DB_PATH = RUNTIME / "homeease.sqlite3"
UPLOADS = RUNTIME / "uploads"
LOGS = RUNTIME / "logs"
VIDEO = ROOT / "algorithm1" / "video"
SIGNAL = ROOT / "algorithm1" / "siganl"
EXERCISE = ROOT / "algorithm2"
AUDIO = ROOT / "algorithm3" / "algorithm3_handover"
EXERCISES = {
    "chair_sit_to_stand": "坐站训练",
    "squat": "深蹲训练",
    "standing_leg_raise": "站立抬腿",
    "hurdle_step": "跨步训练",
}


def prepare():
    for directory in (RUNTIME, UPLOADS, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
