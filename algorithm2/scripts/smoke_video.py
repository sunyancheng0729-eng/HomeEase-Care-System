"""Exercise the real CNN/video path using a static COCO image, not a count test."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.pose import PoseEstimator


def main():
    estimator = PoseEstimator(0 if torch.cuda.is_available() else "cpu")
    manifest = json.loads((ROOT / "data/coco/manifest.json").read_text())
    required = [5, 6, 11, 12, 13, 14, 15, 16]
    for entry in manifest:
        name = entry["url"].rsplit("/", 1)[-1]
        image = cv2.imread(str(ROOT / "data/coco" / name))
        points, person = estimator.estimate(image)
        if person == "primary" and points[required, 2].min() >= .3:
            break
    else:
        raise RuntimeError("No usable single-person image in the saved sample")
    h, w = image.shape[:2]
    video = ROOT / "reports/smoke_static_person.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    for _ in range(60):
        writer.write(image)
    writer.release()
    subprocess.run([sys.executable, str(ROOT / "scripts/run_video.py"), "--source", str(video), "--no-display",
                    "--output", str(ROOT / "reports/smoke_video_session.json")], check=True, cwd=ROOT)
    result = json.loads((ROOT / "reports/smoke_video_session.json").read_text())
    result["test_scope"] = "real_pose_cnn_on_repeated_static_COCO_image_not_exercise_accuracy"
    result["image_source"] = entry
    assert result["count"] == 0, "Static image produced a spurious complete cycle"
    (ROOT / "reports/smoke_video_session.json").write_text(json.dumps(result, indent=2))
    print("Video/CNN smoke passed; static count 0; not a live exercise validation")


if __name__ == "__main__":
    main()
