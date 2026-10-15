"""Measure original inference stages; writes only to an isolated data directory."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HOMEEASE_SYSTEM_DATA", str(ROOT / "runtime" / "performance-audit"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
sys.path.insert(0, str(ROOT))


def summary(values):
    return {"n": len(values), "median_ms": round(statistics.median(values) * 1000, 2),
            "p95_ms": round(sorted(values)[int((len(values) - 1) * .95)] * 1000, 2)}


def video(seconds):
    from care.config import prepare
    from care.workers import VideoWorker
    prepare()
    worker = VideoWorker()
    timings = {}
    for owner, method, label in ((worker.engine._det, "detect_frame", "detector"),
                                 (worker.engine._pose, "extract", "pose"),
                                 (worker.engine._alg, "predict", "a4"),
                                 (worker.engine, "_process", "total")):
        original = getattr(owner, method)
        timings[label] = []
        def timed(*args, _original=original, _label=label, **kwargs):
            started = time.perf_counter()
            result = _original(*args, **kwargs)
            timings[_label].append(time.perf_counter() - started)
            return result
        setattr(owner, method, timed)
    try:
        worker.start({"type": "file", "path": str(ROOT / "samples" / "fall-01-cam0.mp4")})
        deadline = time.monotonic() + seconds
        while worker.engine.is_running() and time.monotonic() < deadline:
            time.sleep(.2)
        return {"stages": {k: summary(v[3:] or v) for k, v in timings.items() if v},
                "status": worker.engine.status(), "runtime": worker.initialize()}
    finally:
        worker.close()


def exercise(seconds):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "algorithm2"))
    from homeease.engine import Predictor, ExerciseSession
    from homeease.pose import PoseEstimator
    cap = cv2.VideoCapture(str(ROOT / "samples" / "fall-01-cam0.mp4"))
    try:
        images = []
        for _ in range(30):
            ok, frame = cap.read()
            if ok:
                images.append(frame)
    finally:
        cap.release()
    result = {}
    outputs = {}
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        predictor = Predictor(device=device)
        estimator = PoseEstimator(0 if device == "cuda" else "cpu")
        session = ExerciseSession(predictor, exercise="squat")
        pose_times, update_times, points_all = [], [], []
        for i, image in enumerate(images):
            start = time.perf_counter()
            points, person = estimator.estimate(image)
            pose_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            session.update(i * 1000 / 30, points, person)
            update_times.append(time.perf_counter() - start)
            points_all.append(points)
        outputs[device] = np.asarray(points_all)
        result[device] = {"pose": summary(pose_times[3:]), "update": summary(update_times[3:])}
    if "cuda" in outputs:
        result["cpu_cuda_max_keypoint_delta"] = float(np.max(np.abs(outputs["cpu"] - outputs["cuda"])))
    return result


def detector(seconds):
    import cv2
    import numpy as np
    from care.config import prepare
    from care.workers import VideoWorker
    from care.adapters.onnx import OnnxNetwork
    prepare()
    worker = VideoWorker()
    original = cv2.dnn.readNetFromONNX(str(ROOT / "algorithm1" / "video" / "standalone" / "yolov8n-pose.onnx"))
    candidate = OnnxNetwork(ROOT / "algorithm1" / "video" / "standalone" / "yolov8n-pose.onnx", device="cuda")
    cap = cv2.VideoCapture(str(ROOT / "samples" / "fall-01-cam0.mp4"))
    timings = {"opencv": [], "onnxruntime": []}
    errors, score_errors = [], []
    try:
        for index in range(80):
            ok, frame = cap.read()
            if not ok:
                break
            outputs = []
            for label, net in (("opencv", original), ("onnxruntime", candidate)):
                worker.engine._det.net = net
                start = time.perf_counter()
                outputs.append(worker.engine._det.detect_frame(frame))
                timings[label].append(time.perf_counter() - start)
            (box, score), (box2, score2) = outputs
            assert (box is None) == (box2 is None)
            if box is not None:
                errors.append(float(np.max(np.abs(box - box2))))
                score_errors.append(abs(score - score2))
        return {"timings": {k: summary(v[3:]) for k, v in timings.items()},
                "frames": len(timings["opencv"]), "max_box_error": max(errors), "max_score_error": max(score_errors)}
    finally:
        cap.release()
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", choices=["video", "exercise", "detector"])
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    result = globals()[args.module](args.seconds)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    (ROOT / "artifacts" / f"profile-{args.module}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
