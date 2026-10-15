"""Subject-disjoint training, ablations and validation-only model selection."""
import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.counter import CycleCounter
from homeease.features import extract_sequence, FEATURE_DIMS
from homeease.models import build_model
from homeease.exercises import EXERCISES, EXERCISE_NAMES, MOVEMENT_TO_INDEX, decode
from scripts.prepare_data import project

CONFIGS = [
    {"name": "tcn_b2_w32", "kind": "tcn", "width": 32, "blocks": 2},
    {"name": "tcn_b3_w32", "kind": "tcn", "width": 32, "blocks": 3},
    {"name": "tcn_b4_w32", "kind": "tcn", "width": 32, "blocks": 4},
    {"name": "tcn_b3_w64", "kind": "tcn", "width": 64, "blocks": 3},
    {"name": "ds_tcn_b3_w32", "kind": "tcn", "width": 32, "blocks": 3, "depthwise": True},
    {"name": "ds_tcn_b4_w64", "kind": "tcn", "width": 64, "blocks": 4, "depthwise": True},
    {"name": "gru_b2_w32", "kind": "gru", "width": 32, "blocks": 2},
    {"name": "gru_b2_w64", "kind": "gru", "width": 64, "blocks": 2},
    {"name": "transformer_b2_w32", "kind": "transformer", "width": 32, "blocks": 2},
    {"name": "tcn_coords", "kind": "tcn", "width": 32, "blocks": 3, "features": "coords"},
    {"name": "tcn_kinematics", "kind": "tcn", "width": 32, "blocks": 3, "features": "kinematics"},
]


def seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def windows(x, context=64):
    padded = np.pad(x, ((context - 1, 0), (0, 0)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, context, axis=0).transpose(0, 2, 1).copy()


def infer(model, x, mean, std, device, context=64):
    data = windows((x - mean) / std, context)
    output = []
    with torch.inference_mode():
        for start in range(0, len(data), 256):
            output.append(model(torch.tensor(data[start:start + 256], device=device))[:, -1].softmax(-1).cpu().numpy())
    return np.concatenate(output)


def replay(skeletons, records, split, movement="m05"):
    sessions = []
    for subject in sorted({r["subject"] for r in records if r["split"] == split}):
        clips = [r for r in records if r["subject"] == subject and r["movement"] == movement and not r["incorrect"]]
        merged, truth = [], []
        for r in sorted(clips, key=lambda r: r["episode"]):
            sk = skeletons[f"{r['id']}_skel"].copy()
            ankle = (sk[0, 16] + sk[0, 20]) / 2
            sk[:, :, 0] -= ankle[0]
            sk[:, :, 2] -= ankle[2]
            merged.append(sk)
            truth.append(skeletons[f"{r['id']}_labels"])
        if merged:
            exercise = next((k for k, v in EXERCISES.items() if v["movement"] == movement), "other")
            sessions.append({"id": f"s{subject:02}_{movement}_reconstructed", "skel": np.concatenate(merged),
                             "exercise": exercise, "labels": np.concatenate(truth),
                             "expected_count": len(clips) - 1 if movement == "m05" else len(clips) if movement in MOVEMENT_TO_INDEX else 0})
    return sessions


def count_probs(probabilities, dwell=.067, confidence=0, exercise="chair_sit_to_stand", strategy="argmax", threshold=.35):
    counter = CycleCounter(min_dwell_s=dwell)
    from homeease.counter import PHASES
    for i, p in enumerate(probabilities):
        if len(p) == 17:
            phase, canonical, _ = decode(p, exercise)
        else:
            phase, canonical = int(p.argmax()), p
        if strategy in ["ordered_posterior", "ordered_transition"]:
            counter.update_probabilities(canonical, i / 30, threshold, infer_return=strategy == "ordered_transition")
        else:
            counter.update(PHASES[phase] if p.max() >= confidence else "other", i / 30)
    return counter


def match_events(predicted, reference, tolerance=.5):
    unused = set(range(len(reference)))
    matched = 0
    for event in predicted:
        options = [i for i in unused if abs(reference[i]["end_s"] - event["end_s"]) <= tolerance]
        if options:
            best = min(options, key=lambda i: abs(reference[i]["end_s"] - event["end_s"]))
            unused.remove(best)
            matched += 1
    return matched, len(predicted) - matched, len(unused)


def count_metrics(session_probs, sessions, dwell=.067, confidence=0, strategy="argmax", threshold=.35):
    errors, relative, exact, details = [], [], [], []
    tp = fp = fn = 0
    for prob, session in zip(session_probs, sessions):
        exercise = session["exercise"] if session["exercise"] != "other" else "chair_sit_to_stand"
        if prob.shape[1] == 17 and confidence == 0:
            if session.get("_probability_source") is not prob:
                session["_probability_source"] = prob
                session["_canonical_probabilities"] = np.asarray([decode(p, exercise)[1] for p in prob])
            counting_probabilities = session["_canonical_probabilities"]
        else:
            counting_probabilities = prob
        counter = count_probs(counting_probabilities, dwell, confidence, exercise, strategy, threshold)
        expected = session["expected_count"]
        error = abs(counter.count - expected)
        errors.append(error)
        if expected:
            relative.append(error / expected)
        exact.append(error == 0)
        if "_reference_events" not in session:
            target = EXERCISE_NAMES.index(exercise)
            phase_labels = np.where(session["labels"] // 4 == target, session["labels"] % 4, 4)
            session["_reference_events"] = count_probs(np.eye(5)[phase_labels], .033, exercise=exercise).events
        reference = session["_reference_events"]
        a, b, c = match_events(counter.events, reference)
        tp, fp, fn = tp + a, fp + b, fn + c
        details.append({"session": session["id"], "exercise": session["exercise"], "expected": expected, "predicted": counter.count,
                        "events": counter.events})
    return {"count_mae": float(np.mean(errors)), "count_mape_pct": float(100 * np.mean(relative)),
            "count_accuracy_pct": float(100 * max(0, 1 - np.mean(relative))), "exact_session_pct": float(100 * np.mean(exact)),
            "cycle_recall": tp / max(tp + fn, 1), "cycle_precision": tp / max(tp + fp, 1),
            "matched_cycles": tp, "spurious_cycles": fp, "missed_cycles": fn, "sessions": details}


def benchmark(model, dimension):
    model = copy.deepcopy(model).cpu().eval()
    sample = torch.zeros(1, 64, dimension)
    with torch.inference_mode():
        for _ in range(30):
            model(sample)
        elapsed = []
        for _ in range(200):
            start = time.perf_counter()
            model(sample)
            elapsed.append((time.perf_counter() - start) * 1000)
    return {"cpu_p50_ms": float(np.median(elapsed)), "cpu_p95_ms": float(np.quantile(elapsed, .95)),
            "parameters": sum(p.numel() for p in model.parameters()), "weight_kib": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    sessions = {s: sum([replay(skeletons, records, s, movement) for movement in ["m05", "m01", "m06", "m02", "m07"]], [])
                for s in ["validation", "test"]}
    results = []
    probabilities = {}
    configs = [c for c in CONFIGS if not args.only or c["name"] in args.only]
    for config in configs:
        name = config["name"]
        config = {"features": "hybrid", "dropout": .1, **config, "seed": args.seed}
        mode = config["features"]
        seeds(args.seed)
        training = [(extract_sequence(project(skeletons[f"{r['id']}_skel"], angle), mode=mode), skeletons[f"{r['id']}_labels"])
                    for r in records if r["split"] == "train" for angle in [25, 45, 65]]
        # Statistics are fitted exclusively on training participants.
        values = np.concatenate([x for x, _ in training])
        mean, std = values.mean(0), np.maximum(values.std(0), .1)
        xx, yy = [], []
        for x, y in training:
            w = windows((x - mean) / std)
            yw = windows(y[:, None]).squeeze(-1)
            for i in sorted(set(range(0, len(x), 16)) | {len(x) - 1}):
                xx.append(w[i])
                labels = yw[i].copy()
                labels[:max(0, 63 - i)] = -100
                yy.append(labels)
        loader = DataLoader(TensorDataset(torch.tensor(np.asarray(xx)), torch.tensor(np.asarray(yy))),
                            batch_size=64, shuffle=True, num_workers=0)
        model = build_model(config, FEATURE_DIMS[mode]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        labels_all = np.concatenate([y for _, y in training])
        class_weights = torch.tensor((1 / np.maximum(np.bincount(labels_all, minlength=17), 1)) ** .5, dtype=torch.float32, device=device)
        class_weights /= class_weights.mean()
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        val_clips = [(extract_sequence(project(skeletons[f"{r['id']}_skel"]), mode=mode), skeletons[f"{r['id']}_labels"])
                     for r in records if r["split"] == "validation"]
        best_score, history, best_state = -1, [], None
        started = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # Small coordinate/feature noise only in training.
                prediction = model(x + torch.randn_like(x) * .015)
                loss = loss_fn(prediction.transpose(1, 2), y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                model.eval()
                val_probs = [infer(model, x, mean, std, device) for x, _ in val_clips]
                score = f1_score(np.concatenate([y for _, y in val_clips]), np.concatenate(val_probs).argmax(1), average="macro")
                history.append({"epoch": epoch + 1, "train_loss": total_loss / len(loader), "validation_macro_f1": score})
                if score > best_score:
                    best_score = score
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"{name} epoch {epoch+1:02}: loss={total_loss/len(loader):.4f}, val_f1={score:.4f}", flush=True)
        training_s = time.perf_counter() - started
        model.load_state_dict(best_state)
        model.eval()
        measured = benchmark(model, FEATURE_DIMS[mode])
        package = {"config": config, "state_dict": best_state, "mean": mean.tolist(), "std": std.tolist(),
                   "context_frames": 64, "fps": 30, "input_dim": FEATURE_DIMS[mode], "label_source": "weak_kinematic_offline_v2", "num_classes": 17}
        torch.save(package, output / f"{name}.pt")
        entry = {"model": name, "config": config, "training_s": training_s, "history": history, **measured}
        probabilities[name] = {}
        for split in ["validation", "test"]:
            split_records = [r for r in records if r["split"] == split]
            probs, truth = [], []
            for r in split_records:
                x = extract_sequence(project(skeletons[f"{r['id']}_skel"]), mode=mode)
                probs.append(infer(model, x, mean, std, device))
                truth.append(skeletons[f"{r['id']}_labels"])
            phase_prob, phase_truth = np.concatenate(probs), np.concatenate(truth)
            session_probs = [infer(model, extract_sequence(project(s["skel"]), mode=mode), mean, std, device) for s in sessions[split]]
            probabilities[name][split] = {"phase": phase_prob, "truth": phase_truth, "sessions": session_probs}
            entry[split] = {"phase_macro_f1": f1_score(phase_truth, phase_prob.argmax(1), average="macro"),
                            "phase_report": classification_report(phase_truth, phase_prob.argmax(1), output_dict=True, zero_division=0),
                            "action_macro_f1": f1_score(phase_truth // 4, phase_prob.argmax(1) // 4, average="macro"),
                            "confusion_matrix": confusion_matrix(phase_truth, phase_prob.argmax(1), labels=list(range(17))).tolist(),
                            **count_metrics(session_probs, sessions[split])}
        results.append(entry)
        (report / "experiments.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"FINISHED {name}: test F1={entry['test']['phase_macro_f1']:.4f}, MAE={entry['test']['count_mae']:.2f}, CPU={measured['cpu_p50_ms']:.3f}ms", flush=True)
    # Ensemble membership and weights are determined using validation alone.
    ranking = sorted(results, key=lambda r: (r["validation"]["count_mae"], -r["validation"]["phase_macro_f1"]))
    members = [r["model"] for r in ranking[:2]]
    ensemble = {"model": "soft_vote_top2", "members": members,
                "parameters": sum(r["parameters"] for r in ranking[:2]),
                "cpu_p50_ms": sum(r["cpu_p50_ms"] for r in ranking[:2]),
                "cpu_p95_ms": sum(r["cpu_p95_ms"] for r in ranking[:2]),
                "weight_kib": sum(r["weight_kib"] for r in ranking[:2]), "latency_note": "sum of member benchmark latencies"}
    for split in ["validation", "test"]:
        phase = np.mean([probabilities[m][split]["phase"] for m in members], axis=0)
        truth = probabilities[members[0]][split]["truth"]
        merged = [np.mean(p, axis=0) for p in zip(*[probabilities[m][split]["sessions"] for m in members])]
        ensemble[split] = {"phase_macro_f1": f1_score(truth, phase.argmax(1), average="macro"),
                           "action_macro_f1": f1_score(truth // 4, phase.argmax(1) // 4, average="macro"),
                           **count_metrics(merged, sessions[split])}
    results.append(ensemble)
    # Keep candidates within .02 of best phase F1 and .25 count MAE; choose CPU latency.
    best_mae = min(r["validation"]["count_mae"] for r in results)
    candidates = [r for r in results if r["validation"]["count_mae"] <= best_mae + .25]
    best_f1 = max(r["validation"]["phase_macro_f1"] for r in candidates)
    candidates = [r for r in candidates if r["validation"]["phase_macro_f1"] >= best_f1 - .02]
    chosen = min(candidates, key=lambda r: r["cpu_p50_ms"])
    selected_members = chosen.get("members", [chosen["model"]])
    selected = {"api_version": "v1", "model": chosen["model"], "members": [f"{m}.pt" for m in selected_members],
                "pose_model": "yolo11n-pose.pt", "confidence_threshold": 0.0, "min_dwell_s": .067,
                "selection_rule": "validation MAE within .25 of best, phase F1 within .02 of best remaining, minimum CPU p50"}
    (output / "selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (report / "experiments.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (report / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["model", "parameters", "cpu_p50_ms", "cpu_p95_ms", "weight_kib", "val_f1", "val_mae", "test_f1", "test_mae", "test_mape_pct", "test_cycle_recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**{k: r[k] for k in fields[:5]}, "val_f1": r["validation"]["phase_macro_f1"],
                             "val_mae": r["validation"]["count_mae"], "test_f1": r["test"]["phase_macro_f1"],
                             "test_mae": r["test"]["count_mae"], "test_mape_pct": r["test"]["count_mape_pct"],
                             "test_cycle_recall": r["test"]["cycle_recall"]})
    print(f"SELECTED {chosen['model']}", flush=True)


if __name__ == "__main__":
    main()
