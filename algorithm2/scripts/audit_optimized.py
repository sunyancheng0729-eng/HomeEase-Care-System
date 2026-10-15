"""Additional fixed-model checks; never tune a model from this output."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.engine import Predictor
from homeease.exercises import EXERCISE_NAMES
from homeease.features import extract_sequence
from homeease.models import build_model
from scripts.prepare_data import project
from scripts.train_compare import count_probs, infer, replay
from scripts.optimize_top import cpu_benchmark


def main():
    predictor = Predictor(device="cuda" if torch.cuda.is_available() else "cpu")
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    output = {"model": predictor.selected["model"], "negative_all_targets": [], "static_all_targets": [], "same_run_cpu_benchmarks": []}
    selected = predictor.selected
    def probabilities(pose):
        return np.mean([infer(m["model"], extract_sequence(pose, mode=m["mode"]), m["mean"], m["std"],
                              predictor.device, context=m["context"]) for m in predictor.members], axis=0)
    def count(probs, exercise):
        return count_probs(probs, selected["min_dwell_s"], exercise=exercise, strategy=selected["decoder_strategy"],
                           threshold=selected["phase_evidence_threshold"]).count
    for session in replay(skeletons, records, "test", "m07"):
        p = probabilities(project(session["skel"]))
        for exercise in EXERCISE_NAMES:
            output["negative_all_targets"].append({"source": session["id"], "target": exercise, "count": count(p, exercise), "expected": 0})
    pose = project(skeletons["m01_s09_e01_skel"])[0]
    p = probabilities(np.repeat(pose[None], 300, axis=0))
    for exercise in EXERCISE_NAMES:
        output["static_all_targets"].append({"target": exercise, "count": count(p, exercise), "expected": 0})
    # Rebenchmark the old and new default with identical warmup/thread settings.
    old = json.loads((ROOT / "reports/optimization/baseline/selected.json").read_text())
    for filename in old["members"] + selected["members"]:
        package = torch.load(ROOT / "artifacts" / filename, map_location="cpu", weights_only=True)
        model = build_model(package["config"], package["input_dim"])
        model.load_state_dict(package["state_dict"])
        output["same_run_cpu_benchmarks"].append({"model": filename.removesuffix(".pt"), "context_frames": package["context_frames"],
                                                   **cpu_benchmark(model, package["input_dim"], package["context_frames"])})
    (ROOT / "reports/optimization/deployment_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    assert all(r["count"] == 0 for r in output["negative_all_targets"] + output["static_all_targets"]), "False cycles found; report and investigate"


if __name__ == "__main__":
    main()
