"""Final consistent decoder evaluation and validation-only calibration."""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.exercises import EXERCISES, EXERCISE_NAMES, decode
from homeease.features import extract_sequence
from homeease.models import build_model
from scripts.prepare_data import project
from scripts.train_compare import CONFIGS, benchmark, count_metrics, infer, replay


def action_predictions(p):
    return np.column_stack([p[:, i * 4:i * 4 + 4].sum(1) for i in range(4)] + [p[:, 16]]).argmax(1)


def evaluate(probs, truth, session_probs, sessions, dwell, strategy="argmax", threshold=.35):
    per_exercise = {}
    for exercise in EXERCISE_NAMES:
        selected = [(p, s) for p, s in zip(session_probs, sessions) if s["exercise"] == exercise]
        per_exercise[exercise] = count_metrics([p for p, _ in selected], [s for _, s in selected], dwell, strategy=strategy, threshold=threshold)
    negative = [int(s["exercise"] == "other") for s in sessions]
    phase_names = [f"{e}/{phase}" for e, cfg in EXERCISES.items() for phase in cfg["phases"]] + ["other"]
    return {"phase_macro_f1": f1_score(truth, probs.argmax(1), average="macro", labels=range(17), zero_division=0),
            "action_macro_f1": f1_score(truth // 4, action_predictions(probs), average="macro"),
            "phase_report": classification_report(truth, probs.argmax(1), labels=range(17), target_names=phase_names, output_dict=True, zero_division=0),
            "action_confusion_matrix": confusion_matrix(truth // 4, action_predictions(probs), labels=range(5)).tolist(),
            "confusion_matrix": confusion_matrix(truth, probs.argmax(1), labels=range(17)).tolist(),
            **count_metrics(session_probs, sessions, dwell, strategy=strategy, threshold=threshold), "per_exercise": per_exercise}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    sessions = {s: sum([replay(skeletons, records, s, m) for m in ["m05", "m01", "m06", "m02", "m07"]], []) for s in ["validation", "test"]}
    history = {r["model"]: r for r in json.loads((ROOT / "reports/experiments.json").read_text())}
    results, predictions = [], {}
    for cfg in CONFIGS:
        path = ROOT / "artifacts" / f"{cfg['name']}.pt"
        if not path.exists():
            if args.preview:
                continue
            raise FileNotFoundError(path)
        package = torch.load(path, weights_only=True, map_location="cpu")
        if package.get("num_classes") != 17:
            if args.preview:
                continue
            raise ValueError(f"Incomplete 4-exercise checkpoint: {path}")
        model = build_model(package["config"], package["input_dim"]).to(device).eval()
        model.load_state_dict(package["state_dict"])
        mean, std = np.array(package["mean"], dtype=np.float32), np.array(package["std"], dtype=np.float32)
        mode, name = package["config"]["features"], cfg["name"]
        predictions[name] = {}
        for split in ["validation", "test"]:
            probs, truth = [], []
            for r in records:
                if r["split"] == split:
                    x = extract_sequence(project(skeletons[f"{r['id']}_skel"]), mode=mode)
                    probs.append(infer(model, x, mean, std, device))
                    truth.append(skeletons[f"{r['id']}_labels"])
            session_probs = [infer(model, extract_sequence(project(s["skel"]), mode=mode), mean, std, device) for s in sessions[split]]
            predictions[name][split] = {"phase": np.concatenate(probs), "truth": np.concatenate(truth), "sessions": session_probs}
        # Decoder dwell is calibrated once per model on validation recordings.
        options = []
        for strategy, threshold in [("argmax", .35)] + [(strategy, t) for strategy in ["ordered_posterior", "ordered_transition"] for t in [.25, .35, .5]]:
            for dwell in [0, .033, .067, .1]:
                metrics = count_metrics(predictions[name]["validation"]["sessions"], sessions["validation"], dwell, strategy=strategy, threshold=threshold)
                options.append({"strategy": strategy, "threshold": threshold, "dwell_s": dwell, "mae": metrics["count_mae"], "cycle_recall": metrics["cycle_recall"]})
        choice = min(options, key=lambda o: (o["mae"], -o["cycle_recall"], -o["dwell_s"], -o["threshold"]))
        row = {"model": name, "config": package["config"], "decoder_dwell_s": choice["dwell_s"], "decoder_strategy": choice["strategy"], "phase_evidence_threshold": choice["threshold"], "decoder_validation_grid": options,
               "training_s": history.get(name, {}).get("training_s"), "history": history.get(name, {}).get("history", []),
               **benchmark(model, package["input_dim"])}
        for split in ["validation", "test"]:
            p = predictions[name][split]
            row[split] = evaluate(p["phase"], p["truth"], p["sessions"], sessions[split], choice["dwell_s"], choice["strategy"], choice["threshold"])
        results.append(row)
        print(f"{name}: val MAE={row['validation']['count_mae']:.2f}, test MAE={row['test']['count_mae']:.2f}, action F1={row['test']['action_macro_f1']:.3f}, dwell={choice['dwell_s']}", flush=True)
    if args.preview:
        (ROOT / "reports/decoder_preview.json").write_text(json.dumps(results, indent=2))
        return
    ranking = sorted(results, key=lambda r: (r["validation"]["count_mae"], -r["validation"]["phase_macro_f1"]))
    members = [r["model"] for r in ranking[:2]]
    ensemble_predictions = {}
    for split in ["validation", "test"]:
        ensemble_predictions[split] = {"phase": np.mean([predictions[m][split]["phase"] for m in members], axis=0),
                                      "truth": predictions[members[0]][split]["truth"],
                                      "sessions": [np.mean(p, axis=0) for p in zip(*[predictions[m][split]["sessions"] for m in members])]}
    options = [{"dwell_s": d, "strategy": strategy, "threshold": threshold,
                **count_metrics(ensemble_predictions["validation"]["sessions"], sessions["validation"], d, strategy=strategy, threshold=threshold)}
               for strategy, threshold in [("argmax", .35)] + [(strategy, t) for strategy in ["ordered_posterior", "ordered_transition"] for t in [.25, .35, .5]] for d in [0, .033, .067, .1]]
    best = min(options, key=lambda o: (o["count_mae"], -o["cycle_recall"], -o["dwell_s"], -o["threshold"]))
    ensemble = {"model": "soft_vote_top2", "members": members, "decoder_dwell_s": best["dwell_s"], "decoder_strategy": best["strategy"], "phase_evidence_threshold": best["threshold"],
                "parameters": sum(r["parameters"] for r in ranking[:2]), "weight_kib": sum(r["weight_kib"] for r in ranking[:2]),
                "cpu_p50_ms": sum(r["cpu_p50_ms"] for r in ranking[:2]), "cpu_p95_ms": sum(r["cpu_p95_ms"] for r in ranking[:2]),
                "latency_note": "sum of independently benchmarked member CPU times; not a measured joint p95"}
    for split in ["validation", "test"]:
        p = ensemble_predictions[split]
        ensemble[split] = evaluate(p["phase"], p["truth"], p["sessions"], sessions[split], best["dwell_s"], best["strategy"], best["threshold"])
    results.append(ensemble)
    best_mae = min(r["validation"]["count_mae"] for r in results)
    eligible = [r for r in results if r["validation"]["count_mae"] <= best_mae + .25]
    best_f1 = max(r["validation"]["phase_macro_f1"] for r in eligible)
    eligible = [r for r in eligible if r["validation"]["phase_macro_f1"] >= best_f1 - .02]
    selected = min(eligible, key=lambda r: r["cpu_p50_ms"])
    (ROOT / "artifacts/selected.json").write_text(json.dumps({"api_version": "v1", "model": selected["model"],
        "members": [f"{m}.pt" for m in selected.get("members", [selected["model"]])], "pose_model": "yolo11n-pose.pt",
        "confidence_threshold": 0, "min_dwell_s": selected["decoder_dwell_s"], "decoder": "conditioned_target_action_support_015",
        "decoder_strategy": selected["decoder_strategy"], "phase_evidence_threshold": selected["phase_evidence_threshold"],
        "selection_rule": "validation MAE <= best+.25, joint phase F1 >= eligible best-.02, minimum CPU p50"}, indent=2))
    (ROOT / "reports/experiments.json").write_text(json.dumps(results, indent=2))
    fields = ["model", "parameters", "cpu_p50_ms", "cpu_p95_ms", "weight_kib", "val_phase_f1", "val_mae", "test_action_f1", "test_phase_f1", "test_mae", "test_mape_pct", "test_cycle_recall"]
    with (ROOT / "reports/comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k: r[k] for k in fields[:5]}, "val_phase_f1": r["validation"]["phase_macro_f1"], "val_mae": r["validation"]["count_mae"],
                             "test_action_f1": r["test"]["action_macro_f1"], "test_phase_f1": r["test"]["phase_macro_f1"],
                             "test_mae": r["test"]["count_mae"], "test_mape_pct": r["test"]["count_mape_pct"], "test_cycle_recall": r["test"]["cycle_recall"]})
    # Keep cached probability sequences for reproducible decoder audits.
    cached = {f"{r['model']}_{split}_{i}": p for r in results[:-1] for split in ["validation", "test"] for i, p in enumerate(predictions[r['model']][split]["sessions"])}
    np.savez_compressed(ROOT / "reports/session_probabilities.npz", **cached)
    print("SELECTED", selected["model"], "dwell", selected["decoder_dwell_s"], flush=True)


if __name__ == "__main__":
    main()
