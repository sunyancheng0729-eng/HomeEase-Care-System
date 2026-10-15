"""Numerical comparisons against original SDKs, in isolated import namespaces."""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
sys.path.insert(0, str(ROOT))
from care.config import prepare, VIDEO, SIGNAL, EXERCISE, AUDIO
from care.adapters.process import AlgorithmProcess


def signal():
    import numpy as np
    sys.path.insert(0, str(SIGNAL))
    from core.detector import detect_offline, StreamingDetector
    from core.model import BestFallModel
    from core.preprocess import read_acc_csv
    source = ROOT / "samples" / "fall-01-acc.csv"
    t, a = read_acc_csv(source)
    expected, info = detect_offline(t, a, BestFallModel.load(SIGNAL / "model" / "best_model.pkl"))
    worker = AlgorithmProcess("signal")
    try:
        actual = worker.call("offline", path=str(source))
        assert expected == actual["events"]
        assert info == actual["info"]
        original = StreamingDetector(BestFallModel.load(SIGNAL / "model" / "best_model.pkl"))
        worker.call("stream_start")
        events_before, events_after = [], []
        alarms_before, alarms_after = [], []
        samples = np.column_stack([t, a]).tolist()
        for index in range(0, len(samples), 25):
            batch = samples[index:index + 25]
            for sample in batch:
                events_before.extend(original.push(*sample))
            alarms_before.extend(original.pop_alarms())
            observed = worker.call("samples", samples=batch)
            events_after.extend(observed["events"])
            alarms_after.extend(observed["alarms"])
            assert observed["windows"] == original.n_eval
            assert observed["prob"] == original.last_prob
        events_before.extend(original.flush())
        events_after.extend(worker.call("flush")["events"])
        strip_wall = lambda rows: [{k: v for k, v in row.items() if k != "wall_alarm"} for row in rows]
        assert strip_wall(events_before) == strip_wall(events_after)
        assert strip_wall(alarms_before) == strip_wall(alarms_after)
        return {"events": len(expected), "windows": info["n_windows"], "max_probability_error": 0.0,
                "stream_events_equal": True, "stream_alarms_equal": True, "stream_windows": original.n_eval}
    finally:
        worker.close()


def exercise():
    import numpy as np
    sys.path.insert(0, str(EXERCISE))
    from homeease.engine import Predictor, ExerciseSession
    worker = AlgorithmProcess("exercise")
    device = worker.call("initialize")["device"]
    predictor = Predictor(device=device)
    results = {}
    try:
        for key in ("chair_sit_to_stand", "squat", "standing_leg_raise", "hurdle_step"):
            replay = json.loads((EXERCISE / "examples" / f"{key}_replay.json").read_text())
            original = ExerciseSession(predictor, exercise=key, target_reps=10)
            created = worker.call("create", exercise=key, target_reps=10)
            sid = created["session_id"]
            for index in range(0, len(replay["frames"]), 100):
                frames = replay["frames"][index:index+100]
                for f in frames:
                    original.update(f["timestamp_ms"], f["keypoints"], f["person_id"])
                result = worker.call("frames", session_id=sid, body={"request_id": f"batch-{index}", "frames": frames})
            expected = original.control("end")
            actual = worker.call("control", session_id=sid, action="end")
            for name in ("count", "cycles", "interruptions", "phase", "phase_probabilities", "active_duration_s", "valid_duration_s", "elapsed_s", "model_id"):
                assert actual[name] == expected[name], (key, name)
            results[key] = {"count": actual["count"], "frames": len(replay["frames"]), "cycles_equal": True,
                            "phase_probabilities_equal": True}
            worker.call("release", session_id=sid)
        import cv2
        import struct
        from multiprocessing import shared_memory
        from homeease.pose import PoseEstimator
        cap = cv2.VideoCapture(str(ROOT / "samples" / "fall-01-cam0.mp4"))
        try:
            ok, image = cap.read()
            assert ok
        finally:
            cap.release()
        memory = shared_memory.SharedMemory(create=True, size=16 + image.nbytes)
        sid = worker.call("create", exercise="squat", target_reps=10)["session_id"]
        try:
            expected_points, expected_person = PoseEstimator(device=0 if device.startswith("cuda") else device).estimate(image)
            worker.call("pose_initialize")
            stamp = time.perf_counter()
            struct.pack_into("<Qd", memory.buf, 0, 2, stamp)
            memory.buf[16:] = image.tobytes()
            observed = worker.call("camera_frame", session_id=sid,
                descriptor={"name": memory.name, "shape": list(image.shape)}, base_stamp=stamp, last_sequence=0)
            assert observed["person_id"] == expected_person
            assert np.allclose(observed["keypoints"], expected_points, atol=1e-6)
            results["shared_camera_pose"] = {"person_equal": True,
                "max_keypoint_error": float(np.max(np.abs(np.asarray(observed["keypoints"]) - expected_points)))}
        finally:
            worker.call("release", session_id=sid)
            memory.close()
            memory.unlink()
        return results
    finally:
        worker.close()


def video():
    import cv2
    import numpy as np
    from care.workers import VideoWorker, load_file
    worker = VideoWorker()
    optimized_threads = worker.initialize()["opencv_threads"]
    sys.path.insert(0, str(VIDEO / "standalone"))
    original = load_file("homeease_video_reference", VIDEO / "standalone" / "main.py")
    detector = original.YOLOPersonDetector()
    pose = original.PoseExtractor()
    model = original.A4Algorithm(weights_path=str(VIDEO / "a4_weights.pkl"))
    cap = cv2.VideoCapture(str(ROOT / "samples" / "fall-01-cam0.mp4"))
    skeletons, errors, frame_count = [], [], 0
    try:
        for index in range(70):
            ok, frame = cap.read()
            if not ok:
                break
            if index % 2:
                continue
            frame_count += 1
            h, w = frame.shape[:2]
            cv2.setNumThreads(20)
            box, score = detector.detect_frame(frame)
            cv2.setNumThreads(optimized_threads)
            adapter_box, adapter_score = worker.engine._det.detect_frame(frame)
            assert np.allclose(box, adapter_box, atol=1e-6, rtol=0) if box is not None else adapter_box is None
            assert abs(score - adapter_score) <= 1e-6
            points = np.zeros((17,3), np.float32)
            if box is not None and score > 0:
                bbox = np.array(box)*[w,h,w,h]
                pts, conf = pose.extract(frame,bbox)
                pts2, conf2 = worker.engine._pose.extract(frame,bbox)
                assert np.allclose(pts,pts2,atol=1e-5)
                assert np.allclose(conf,conf2,atol=1e-6)
                points = np.stack([pts[:,0]/w,pts[:,1]/h,conf],axis=1).astype(np.float32)
            skeletons.append(points)
        assert len(skeletons) >= 30
        window=np.stack(skeletons[:30])
        with worker.module.a4_fast_context():
            reference=model.predict(window)
            actual=worker.engine._alg.predict(window)
        assert np.allclose(reference,actual,atol=1e-6)
        import hashlib
        copied=Path(worker.module.__file__).read_bytes()
        assert copied==(VIDEO/"engine.py").read_bytes()
        from types import SimpleNamespace
        from unittest.mock import patch
        reference_engine = worker.module.DetectionEngine()
        cap.release()
        cap = cv2.VideoCapture(str(ROOT / "samples" / "fall-01-cam0.mp4"))
        errors, triggers = [], []
        clock = {"now": 1000.0}
        with patch.object(worker.module, "time", SimpleNamespace(time=lambda: clock["now"])):
            for index in range(160):
                ok, frame = cap.read()
                if not ok:
                    break
                if index % 2:
                    continue
                number = index // 2 + 1
                clock["now"] = 1000 + number / 15
                args = (frame, number, (number - 1) % 3 == 0, number % 2 == 0)
                _, p1, alarm1 = reference_engine._process(*args)
                _, p2, alarm2 = worker.engine._process(*args)
                errors.append(abs(p1 - p2))
                assert abs(p1 - p2) <= 1e-6, (number, p1, p2)
                assert alarm1 == alarm2, (number, alarm1, alarm2)
                if alarm2:
                    triggers.append(number)
        assert len(errors) == 80 and triggers
        return {"observed_frames_compared":frame_count,"max_probability_error":float(np.max(np.abs(reference-actual))),
                "detector_and_pose_equal":True,"engine_source_identical":True,"probability":actual.tolist(),
                "detector_backend": worker.detector_backend, "detector_tolerance": 1e-6,
                "full_pipeline_frames": len(errors), "full_pipeline_max_probability_error": max(errors),
                "full_pipeline_trigger_frames_equal": triggers,
                "clock": "same 15 Hz timeline on both original engines; no recording side effects"}
    finally:
        cap.release()
        worker.close()


def audio():
    import numpy as np
    import os
    os.environ["HOMEEASE_ROOT"]=str(AUDIO)
    # An existing original ASR option, applied identically to both sides only
    # for reproducibility. The running product retains the original default.
    os.environ["HOMEEASE_ASR_TEMPERATURE"]="0"
    sys.path.insert(0,str(AUDIO/"src"))
    from algorithm3 import Algorithm3
    original=Algorithm3(vad="silero")
    worker=AlgorithmProcess("audio")
    try:
        data=(AUDIO/"samples"/"scene_quiet_home.wav").read_bytes()
        x=original.decode(data)
        expected=original.label_sounds(x)
        actual=worker.call("analyze",task="sounds",data=base64.b64encode(data).decode())
        assert actual==expected
        data=(AUDIO/"samples"/"reply_help_en.wav").read_bytes()
        expected_help=original.detect_spoken_help(original.decode(data))
        actual_help=worker.call("analyze",task="help",data=base64.b64encode(data).decode())
        joint=worker.call("analyze",task="monitor",data=base64.b64encode(data).decode())
        assert joint["sounds"] == original.label_sounds(original.decode(data))
        assert joint["help_result"]["help"] == actual_help["help"]
        assert [s["verdict"] for s in joint["help_result"]["segments"]] == [s["verdict"] for s in actual_help["segments"]]
        expected_segments = [dict(s) for s in expected_help["segments"]]
        actual_segments = [dict(s) for s in actual_help["segments"]]
        logprob_error = 0.0
        assert len(expected_segments) == len(actual_segments)
        for before, after in zip(expected_segments, actual_segments):
            lp_before, lp_after = before.pop("asr_avg_logprob", None), after.pop("asr_avg_logprob", None)
            assert before == after
            if lp_before is not None:
                assert lp_after is not None
                delta = abs(lp_before - lp_after)
                assert delta <= .05, (lp_before, lp_after)
                logprob_error = max(logprob_error, delta)
        assert {k:v for k,v in actual_help.items() if k != "segments"} == {k:v for k,v in expected_help.items() if k != "segments"}
        text_results={}
        for text in ("我需要帮助", "I'm fine", "Don't worry, she is fine."):
            expected=original.classify_reply(text=text)
            actual=worker.call("analyze",task="reply_text",data="",text=text)
            assert actual==expected
            text_results[text]=actual["intent"]
        return {"sound_events_equal":True,"joint_monitor_equal":True,"help_verdicts_equal":True,"help":actual_help["help"],"reply_intents":text_results,
                "asr_logprob_max_error":logprob_error,"asr_logprob_tolerance":.05,
                "asr_temperature":"0 for parity test only; product preserves original fallback default"}
    finally:
        worker.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--module",choices=["video","signal","exercise","audio"])
    args=parser.parse_args()
    prepare()
    if args.module:
        result=globals()[args.module]()
        output = ROOT / "artifacts" / "integration_parity.json"
        results = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
        results[args.module] = result
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("RESULT:"+json.dumps(result,ensure_ascii=True))
    else:
        results={}
        for name in ("signal","exercise","audio","video"):
            command=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--module",name],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",timeout=300)
            if command.returncode:
                print(command.stdout)
                print(command.stderr)
                raise SystemExit(command.returncode)
            line=next(line for line in command.stdout.splitlines() if line.startswith("RESULT:"))
            results[name]=json.loads(line[7:])
            print(name,results[name],flush=True)
        (ROOT/"artifacts"/"integration_parity.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
