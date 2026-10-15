"""Test fixed selected model on unseen views/noise; never recalibrate here."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.engine import ExerciseSession, Predictor
from homeease.exercises import EXERCISES
from homeease.features import extract_sequence
from scripts.prepare_data import project
from scripts.train_compare import replay, infer, count_probs


def main():
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    predictor = Predictor(device="cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(123)
    outcomes = []
    for angle, noise in [(15, 0), (45, 0), (75, 0), (45, 1)]:
        details = []
        for exercise, config in EXERCISES.items():
            for source in replay(skeletons, records, "test", config["movement"]):
                poses = project(source["skel"], angle)
                if noise:
                    poses[:, :, :2] += rng.normal(0, noise, poses[:, :, :2].shape)
                probabilities = np.mean([infer(m["model"], extract_sequence(poses, mode=m["mode"]),
                                               m["mean"], m["std"], predictor.device, context=m["context"])
                                         for m in predictor.members], axis=0)
                counter = count_probs(probabilities, predictor.selected["min_dwell_s"], exercise=exercise,
                                      strategy=predictor.selected["decoder_strategy"],
                                      threshold=predictor.selected["phase_evidence_threshold"])
                details.append({"session": source["id"], "exercise": exercise,
                                "reference": source["expected_count"], "count": counter.count})
        outcomes.append({"model": predictor.selected["model"], "angle_degrees": angle, "noise_source_units_std": noise,
                         "mae": float(np.mean([abs(d["count"] - d["reference"]) for d in details])),
                         "mape_pct": float(np.mean([abs(d["count"] - d["reference"]) / d["reference"] * 100 for d in details])),
                         "details": details})
        print(outcomes[-1], flush=True)
    (ROOT / "reports/robustness.json").write_text(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()
