"""Reusable pose CNN adapter. Ambiguous multi-person observations abstain."""
import json
from pathlib import Path

import numpy as np

from .engine import ROOT


class PoseEstimator:
    def __init__(self, device="cpu", artifact_dir=None):
        from ultralytics import YOLO
        directory = Path(artifact_dir or ROOT / "artifacts")
        config = json.loads((directory / "selected_pose.json").read_text())
        self.model = YOLO(str(directory / config["model"]))
        self.size, self.device = config["imgsz"], device

    def estimate(self, bgr):
        result = self.model.predict(bgr, imgsz=self.size, device=self.device, verbose=False, conf=.35)[0]
        if len(result.boxes) != 1:
            return np.zeros((17, 3), dtype=np.float32), "no_person" if len(result.boxes) == 0 else "multiple_people"
        return result.keypoints.data[0].cpu().numpy(), "primary"
