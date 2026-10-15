"""Export four real held-out skeleton replays and verify SDK/HTTP integration."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.engine import ExerciseSession, Predictor
from homeease.exercises import EXERCISES
from scripts.prepare_data import project
from scripts.train_compare import replay


def coco_points(pose):
    result = np.zeros((17, 3), dtype=np.float32)
    result[5:] = pose
    result[:, :2] = result[:, :2] * 3 + np.array([960, 720])
    return result.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Optional HTTP server URL, e.g. http://127.0.0.1:8002")
    args = parser.parse_args()
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    predictor = Predictor()
    reports = []
    example_dir = ROOT / "examples"
    example_dir.mkdir(exist_ok=True)
    for exercise, config in EXERCISES.items():
        source = replay(skeletons, records, "test", config["movement"])[0]
        poses = project(source["skel"])
        frames = [{"timestamp_ms": round(i / 30 * 1000, 6), "person_id": "primary", "keypoints": coco_points(p)} for i, p in enumerate(poses)]
        stream = ExerciseSession(predictor, exercise=exercise, target_reps=source["expected_count"])
        for frame in frames:
            stream.update(frame["timestamp_ms"], frame["keypoints"], frame["person_id"])
        result = stream.control("end")
        result["reference_count"] = source["expected_count"]
        result["source"] = source["id"]
        if args.url:
            import requests
            with requests.Session() as client:
                response = client.post(f"{args.url}/v1/sessions", json={"exercise": exercise, "target_reps": source["expected_count"]})
                response.raise_for_status()
                sid = response.json()["session_id"]
                for start in range(0, len(frames), 100):
                    response = client.post(f"{args.url}/v1/sessions/{sid}/frames", json={"request_id": f"{exercise}-{start}", "frames": frames[start:start + 100]})
                    response.raise_for_status()
                final = client.post(f"{args.url}/v1/sessions/{sid}/control", json={"action": "end"})
                final.raise_for_status()
                assert final.json()["count"] == result["count"], "HTTP and SDK count differ"
                result["http_count"] = final.json()["count"]
                client.delete(f"{args.url}/v1/sessions/{sid}").raise_for_status()
        (example_dir / f"{exercise}_frames.json").write_text(json.dumps({"request_id": "example-001", "frames": frames[:100]}, indent=2))
        (example_dir / f"{exercise}_replay.json").write_text(json.dumps({"exercise": exercise, "reference_count": source["expected_count"], "frames": frames}))
        reports.append(result)
        print(f"{exercise}: expected={source['expected_count']} predicted={result['count']}", flush=True)
    (ROOT / "reports/replay_sessions.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
