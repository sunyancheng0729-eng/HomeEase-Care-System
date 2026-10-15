"""Optimize three strong temporal architectures without changing labels/splits.

Training reads only train/validation arrays. The locked validation winner is
written before --finalize accesses test arrays. Baselines remain archived.
"""
import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.features import extract_sequence, FEATURE_DIMS
from homeease.models import build_model
from scripts.prepare_data import project
from scripts.train_compare import seeds, windows, replay, count_metrics
from scripts.evaluate_select import evaluate, action_predictions

CONFIGS = [
    {"name": "opt_tcn_b4_w48", "kind": "tcn", "width": 48, "blocks": 4, "context": 96},
    {"name": "opt_tcn_b5_w48", "kind": "tcn", "width": 48, "blocks": 5, "context": 128},
    {"name": "opt_gru_b2_w64", "kind": "gru", "width": 64, "blocks": 2, "context": 96},
]
OUT = ROOT / "reports/optimization"


class CausalWindowDataset(Dataset):
    """Keep source sequences once; materialize only the requested batch windows."""
    def __init__(self, sequences, mean, std, context, stride=12):
        self.context = context
        self.sequences = [((x - mean) / std, y) for x, y in sequences]
        self.indices = [(j, i) for j, (x, _) in enumerate(sequences)
                        for i in sorted(set(range(0, len(x), stride)) | {len(x) - 1})]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        j, i = self.indices[index]
        x, y = self.sequences[j]
        xx = x[max(0, i + 1 - self.context):i + 1]
        if len(xx) < self.context:
            xx = np.pad(xx, ((self.context - len(xx), 0), (0, 0)), mode="edge")
        yy = y[max(0, i - 7):i + 1]
        if len(yy) < 8:
            yy = np.pad(yy, (8 - len(yy), 0), mode="edge")
        return torch.from_numpy(xx), torch.from_numpy(yy)


def predict(model, x, mean, std, context, device, stride=1):
    data = windows((x - mean) / std, context)[::stride]
    out = []
    with torch.inference_mode():
        for start in range(0, len(data), 256):
            out.append(model(torch.as_tensor(data[start:start + 256], device=device))[:, -1].softmax(-1).cpu().numpy())
    return np.concatenate(out)


def decoder_grid(probabilities, sessions, expanded=False):
    options = []
    for strategy in ["ordered_posterior", "ordered_transition"]:
        for threshold in [.25, .35, .5]:
            for dwell in ([0, .033, .067, .1] if expanded else [.067, .1]):
                m = count_metrics(probabilities, sessions, dwell, strategy=strategy, threshold=threshold)
                options.append({"strategy": strategy, "threshold": threshold, "dwell_s": dwell,
                                "mae": m["count_mae"], "cycle_recall": m["cycle_recall"]})
    choice = min(options, key=lambda o: (o["mae"], -o["cycle_recall"], -o["dwell_s"], -o["threshold"]))
    return choice, options


def data_for(records, skeletons, split):
    return [(project(skeletons[f"{r['id']}_skel"]), skeletons[f"{r['id']}_labels"])
            for r in records if r["split"] == split]


def sessions_for(records, skeletons, split):
    return sum([replay(skeletons, records, split, m) for m in ["m05", "m01", "m06", "m02", "m07"]], [])


def cpu_benchmark(model, dimension, context):
    model = copy.deepcopy(model).cpu().eval()
    sample = torch.zeros(1, context, dimension)
    elapsed = []
    with torch.inference_mode():
        for _ in range(40):
            model(sample)
        for _ in range(300):
            start = time.perf_counter()
            model(sample)
            elapsed.append((time.perf_counter() - start) * 1000)
    parameters = sum(p.numel() for p in model.parameters())
    return {"cpu_p50_ms": float(np.median(elapsed)), "cpu_p95_ms": float(np.quantile(elapsed, .95)),
            "parameters": parameters, "weight_kib": parameters * 4 / 1024}


def archive_baseline():
    archive = OUT / "baseline"
    if archive.exists():
        return
    archive.mkdir(parents=True)
    for name in ["experiments.json", "comparison.csv", "session_probabilities.npz"]:
        shutil.copy2(ROOT / "reports" / name, archive / name)
    shutil.copy2(ROOT / "artifacts/selected.json", archive / "selected.json")


def train(args, records, skeletons):
    archive_baseline()
    seeds(args.seed)
    training = []
    for r in records:
        if r["split"] != "train":
            continue
        sk, y = skeletons[f"{r['id']}_skel"], skeletons[f"{r['id']}_labels"]
        for angle, speed, mirror in [(25, .85, False), (45, 1., False), (65, 1.15, False), (45, 1., True)]:
            pose = project(sk, angle)
            indices = np.minimum(np.arange(0, len(pose), speed).astype(int), len(pose) - 1)
            pose, labels = pose[indices].copy(), y[indices]
            if mirror:
                pose[:, :, 0] *= -1
                pose = pose[:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]]
            training.append((extract_sequence(pose, mode="stable80"), labels))
    print(f"Prepared {len(training)} augmented clips", flush=True)
    # Train on cross-repetition histories as well as original isolated clips.
    for session in sessions_for(records, skeletons, "train"):
        for angle in [25, 45, 65]:
            training.append((extract_sequence(project(session["skel"], angle), mode="stable80"), session["labels"]))
    print(f"Prepared training clips plus sessions: {len(training)} sequences", flush=True)
    values = np.concatenate([x for x, _ in training])
    mean, std = values.mean(0), np.maximum(values.std(0), .1)
    val = [(extract_sequence(p, mode="stable80"), y) for p, y in data_for(records, skeletons, "validation")]
    sessions = sessions_for(records, skeletons, "validation")
    session_features = [extract_sequence(project(s["skel"]), mode="stable80") for s in sessions]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_labels = np.concatenate([y for _, y in training])
    counts = np.bincount(all_labels, minlength=17)
    phase_weights = torch.tensor((1 / np.maximum(counts[:16].reshape(4, 4).sum(0), 1)) ** .35, device=device, dtype=torch.float32)
    phase_weights /= phase_weights.mean()
    action_weights = torch.tensor((1 / np.bincount(all_labels // 4, minlength=5)) ** .35, device=device, dtype=torch.float32)
    action_weights /= action_weights.mean()
    existing = json.loads((OUT / "validation.json").read_text()) if (OUT / "validation.json").exists() else []
    for cfg in CONFIGS:
        if args.only and cfg["name"] not in args.only:
            continue
        config = {"features": "stable80", "hierarchical": True, "dropout": .15, "seed": args.seed, **cfg}
        seeds(args.seed)
        context = cfg["context"]
        dataset = CausalWindowDataset(training, mean, std, context)
        loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
        print(f"Training {cfg['name']}: {len(dataset)} lazy windows", flush=True)
        model = build_model(config, 80).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00005)
        best_key, best_state, best_decoder, best_epoch = None, None, None, None
        history, stale = [], 0
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                actions, phases = model.components(x + torch.randn_like(x) * .025)
                actions, phases = actions[:, -8:], phases[:, -8:]
                a = y // 4
                action_loss = torch.nn.functional.cross_entropy(actions.flatten(0, 1), a.flatten(), weight=action_weights, label_smoothing=.04)
                mask = a < 4
                target_phase = phases[mask, a[mask]]
                phase_loss = torch.nn.functional.cross_entropy(target_phase, (y % 4)[mask], weight=phase_weights, label_smoothing=.025)
                loss = .7 * action_loss + phase_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            if epoch % 5 == 0 or epoch == args.epochs:
                model.eval()
                probs = [predict(model, x, mean, std, context, device, stride=3) for x, _ in val]
                truth = np.concatenate([y[::3] for _, y in val])
                phase_f1 = f1_score(truth, np.concatenate(probs).argmax(1), average="macro", labels=range(17), zero_division=0)
                action_f1 = f1_score(truth // 4, action_predictions(np.concatenate(probs)), average="macro")
                sp = [predict(model, x, mean, std, context, device) for x in session_features]
                decoder, grid = decoder_grid(sp, sessions)
                key = (decoder["mae"], -phase_f1)
                improved = best_key is None or key < best_key
                if improved:
                    best_key, best_decoder, best_epoch = key, decoder, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                history.append({"epoch": epoch, "loss": total / len(loader), "validation_phase_f1_stride3": phase_f1,
                                "validation_action_f1_stride3": action_f1, "validation_count_mae": decoder["mae"], "decoder": decoder})
                print(f"{cfg['name']} epoch={epoch} loss={total/len(loader):.4f} val_action={action_f1:.4f} val_phase={phase_f1:.4f} val_MAE={decoder['mae']:.2f} best_epoch={best_epoch}", flush=True)
                if stale >= 8:
                    break
        model.load_state_dict(best_state)
        model.eval()
        package = {"config": config, "state_dict": best_state, "mean": mean.tolist(), "std": std.tolist(),
                   "context_frames": context, "fps": 30, "input_dim": 80, "num_classes": 17,
                   "label_source": "weak_kinematic_offline_v2", "best_epoch": best_epoch}
        torch.save(package, ROOT / "artifacts" / f"{cfg['name']}.pt")
        phase = np.concatenate([predict(model, x, mean, std, context, device) for x, _ in val])
        truth = np.concatenate([y for _, y in val])
        sp = [predict(model, x, mean, std, context, device) for x in session_features]
        decoder, grid = decoder_grid(sp, sessions)
        row = {"model": cfg["name"], "config": config, "training_s": time.perf_counter() - started, "best_epoch": best_epoch,
               "history": history, "decoder_dwell_s": decoder["dwell_s"], "decoder_strategy": decoder["strategy"],
               "phase_evidence_threshold": decoder["threshold"], "decoder_validation_grid": grid,
               **cpu_benchmark(model, 80, context),
               "validation": evaluate(phase, truth, sp, sessions, decoder["dwell_s"], decoder["strategy"], decoder["threshold"])}
        existing = [r for r in existing if r["model"] != row["model"]] + [row]
        (OUT / "validation.json").write_text(json.dumps(existing, indent=2), encoding="utf-8")
        np.savez_compressed(OUT / f"{cfg['name']}_validation_probabilities.npz", phase=phase, truth=truth, **{f"session_{i}": p for i, p in enumerate(sp)})
        print(f"FINISHED {cfg['name']} val_MAE={row['validation']['count_mae']:.2f} phase={row['validation']['phase_macro_f1']:.4f} CPU={row['cpu_p50_ms']:.3f}ms", flush=True)
    lock_selection(existing, args.seed, "training_decoder_grid")


def lock_selection(rows, seed, stage):
    best_mae = min(r["validation"]["count_mae"] for r in rows)
    eligible = [r for r in rows if r["validation"]["count_mae"] <= best_mae + .2]
    best_phase = max(r["validation"]["phase_macro_f1"] for r in eligible)
    eligible = [r for r in eligible if r["validation"]["phase_macro_f1"] >= best_phase - .02]
    chosen = min(eligible, key=lambda r: r["cpu_p50_ms"])
    lock = {"model": chosen["model"], "selection_rule": "validation count MAE <= best+.2; joint phase F1 >= eligible best-.02; minimum CPU p50",
            "decoder_strategy": chosen["decoder_strategy"], "min_dwell_s": chosen["decoder_dwell_s"],
            "phase_evidence_threshold": chosen["phase_evidence_threshold"], "seed": seed, "stage": stage,
            "checkpoint_sha256": hashlib.sha256((ROOT / "artifacts" / f"{chosen['model']}.pt").read_bytes()).hexdigest(),
            "locked_at_unix": time.time(), "test_accessed_during_this_training": False}
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print("LOCKED", chosen["model"], flush=True)


def calibrate(records, skeletons):
    rows = json.loads((OUT / "validation.json").read_text())
    sessions = sessions_for(records, skeletons, "validation")
    for row in rows:
        checkpoint = ROOT / "artifacts" / f"{row['model']}.pt"
        package = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = build_model(package["config"], package["input_dim"]).eval()
        model.load_state_dict(package["state_dict"])
        cached = np.load(OUT / f"{row['model']}_validation_probabilities.npz")
        if package["config"]["kind"] == "tcn":
            receptive_field = 1 + 4 * (2 ** package["config"]["blocks"] - 1)
            minimal_context = 2 ** int(np.ceil(np.log2(receptive_field)))
            if minimal_context < package["context_frames"]:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                mean, std = np.asarray(package["mean"], dtype=np.float32), np.asarray(package["std"], dtype=np.float32)
                def run(pose):
                    return predict(model, extract_sequence(pose, mode=package["config"]["features"]), mean, std, minimal_context, device)
                data = data_for(records, skeletons, "validation")
                phase = np.concatenate([run(pose) for pose, _ in data])
                sp = [run(project(s["skel"])) for s in sessions]
                error = max([float(np.max(np.abs(phase - cached["phase"])))] +
                            [float(np.max(np.abs(p - cached[f"session_{i}"]))) for i, p in enumerate(sp)])
                same_phase = bool(np.array_equal(phase.argmax(1), cached["phase"].argmax(1)))
                same_action = bool(np.array_equal(action_predictions(phase), action_predictions(cached["phase"])))
                if error > 2e-5 or not same_phase or not same_action:
                    raise ValueError("TCN context contraction changed validation predictions")
                backup = OUT / "training_checkpoints"
                backup.mkdir(exist_ok=True)
                shutil.copy2(checkpoint, backup / checkpoint.name)
                old_context = package["context_frames"]
                package["context_frames"] = minimal_context
                package["config"].update({"training_context": old_context, "context": minimal_context})
                row["config"] = package["config"]
                row["context_contraction"] = {"training_context": old_context, "deployment_context": minimal_context,
                                              "receptive_field": receptive_field, "max_abs_probability_error": error,
                                              "same_phase_predictions": same_phase, "same_action_predictions": same_action}
                torch.save(package, checkpoint)
                truth = cached["truth"].copy()
                cached.close()
                np.savez_compressed(OUT / f"{row['model']}_validation_probabilities.npz", phase=phase, truth=truth,
                                    **{f"session_{i}": p for i, p in enumerate(sp)})
                cached = np.load(OUT / f"{row['model']}_validation_probabilities.npz")
                print(f"CONTRACTED {row['model']} {old_context}->{minimal_context} frames; max probability error={error:.3g}", flush=True)
        row["training_time_cpu_measurement"] = {k: row[k] for k in ["cpu_p50_ms", "cpu_p95_ms"]}
        row.update(cpu_benchmark(model, package["input_dim"], package["context_frames"]))
        sp = [cached[f"session_{i}"] for i in range(len(sessions))]
        choice, grid = decoder_grid(sp, sessions, expanded=True)
        row["checkpoint_decoder_validation_grid"] = row["decoder_validation_grid"]
        row["decoder_validation_grid"] = grid
        row["decoder_dwell_s"], row["decoder_strategy"], row["phase_evidence_threshold"] = choice["dwell_s"], choice["strategy"], choice["threshold"]
        row["validation"] = evaluate(cached["phase"], cached["truth"], sp, sessions, choice["dwell_s"], choice["strategy"], choice["threshold"])
        print(f"CALIBRATED {row['model']} MAE={choice['mae']:.2f} dwell={choice['dwell_s']} strategy={choice['strategy']}", flush=True)
    (OUT / "validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lock_selection(rows, rows[0]["config"]["seed"], "full_validation_decoder_calibration")


def finalize(records, skeletons):
    lock = json.loads((OUT / "selection_lock.json").read_text())
    if lock.get("stage") != "full_validation_decoder_calibration":
        raise ValueError("Run --calibrate before --finalize")
    rows = json.loads((OUT / "validation.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sessions = sessions_for(records, skeletons, "test")
    for row in rows:
        path = ROOT / "artifacts" / f"{row['model']}.pt"
        if row["model"] == lock["model"] and hashlib.sha256(path.read_bytes()).hexdigest() != lock["checkpoint_sha256"]:
            raise ValueError("Locked checkpoint changed")
        p = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(p["config"], p["input_dim"]).to(device).eval()
        model.load_state_dict(p["state_dict"])
        mean, std = np.asarray(p["mean"], dtype=np.float32), np.asarray(p["std"], dtype=np.float32)
        def run(pose):
            return predict(model, extract_sequence(pose, mode=p["config"]["features"]), mean, std, p["context_frames"], device)
        data = data_for(records, skeletons, "test")
        phase, truth = np.concatenate([run(pose) for pose, _ in data]), np.concatenate([y for _, y in data])
        sp = [run(project(s["skel"])) for s in sessions]
        row["test"] = evaluate(phase, truth, sp, sessions, row["decoder_dwell_s"], row["decoder_strategy"], row["phase_evidence_threshold"])
        np.savez_compressed(OUT / f"{row['model']}_test_probabilities.npz", phase=phase, truth=truth, **{f"session_{i}": v for i, v in enumerate(sp)})
        print(f"TEST {row['model']} action={row['test']['action_macro_f1']:.4f} phase={row['test']['phase_macro_f1']:.4f} MAE={row['test']['count_mae']:.2f} MAPE={row['test']['count_mape_pct']:.2f}%", flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    baseline = json.loads((OUT / "baseline/experiments.json").read_text())
    (ROOT / "reports/experiments.json").write_text(json.dumps(baseline + rows, indent=2), encoding="utf-8")
    fields = ["model", "parameters", "cpu_p50_ms", "cpu_p95_ms", "weight_kib", "val_phase_f1", "val_mae", "test_action_f1", "test_phase_f1", "test_mae", "test_mape_pct", "test_cycle_recall"]
    with (ROOT / "reports/comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in baseline + rows:
            writer.writerow({**{k: r[k] for k in fields[:5]}, "val_phase_f1": r["validation"]["phase_macro_f1"],
                             "val_mae": r["validation"]["count_mae"], "test_action_f1": r["test"]["action_macro_f1"],
                             "test_phase_f1": r["test"]["phase_macro_f1"], "test_mae": r["test"]["count_mae"],
                             "test_mape_pct": r["test"]["count_mape_pct"], "test_cycle_recall": r["test"]["cycle_recall"]})
    selected = json.loads((OUT / "baseline/selected.json").read_text())
    selected.update({k: lock[k] for k in ["model", "selection_rule", "decoder_strategy", "min_dwell_s", "phase_evidence_threshold"]})
    selected["members"] = [f"{lock['model']}.pt"]
    selected["optimization_selection_lock"] = "reports/optimization/selection_lock.json"
    (ROOT / "artifacts/selected.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    if args.finalize:
        finalize(records, skeletons)
    elif args.calibrate:
        calibrate(records, skeletons)
    else:
        train(args, records, skeletons)


if __name__ == "__main__":
    main()
