"""Local video/webcam exercise demo and JSON report, with no remote video upload."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.engine import ExerciseSession, Predictor
from homeease.pose import PoseEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--exercise", choices=["chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"], default="chair_sit_to_stand")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output", default="reports/video_session.json")
    args = parser.parse_args()
    webcam = args.source.isdecimal()
    capture = cv2.VideoCapture(int(args.source) if webcam else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    pose = PoseEstimator(0 if args.device.startswith("cuda") else "cpu")
    session = ExerciseSession(Predictor(device=args.device), target_reps=args.target, exercise=args.exercise)
    started, index = time.monotonic(), 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            points, identity = pose.estimate(frame)
            timestamp = (time.monotonic() - started) * 1000 if webcam else index / fps * 1000
            index += 1
            if session.status == "running":
                state = session.update(timestamp, points, identity)
            else:
                state = session.snapshot()
            if not args.no_display:
                cv2.putText(frame, f"{state['count']}/{args.target}  {state['phase']}  {state['quality']}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 200, 80), 2)
                cv2.imshow("HomeEase exercise", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    session.control("pause" if session.status == "running" else "resume")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    result = session.control("end")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
