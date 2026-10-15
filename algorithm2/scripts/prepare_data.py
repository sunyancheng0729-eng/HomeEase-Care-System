"""Reconstruct original skeletons and export explicit weak phase annotations."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.features import extract_sequence
from homeease.exercises import MOVEMENT_TO_INDEX

PARENTS = [-1, 0, 1, 2, 3, 4, 2, 6, 7, 8, 2, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20]
NODES = [7, 11, 8, 12, 9, 13, 14, 18, 15, 19, 16, 20]


def reconstruct(position_path):
    p = np.loadtxt(position_path, delimiter=",").reshape(-1, 22, 3)
    a = np.loadtxt(str(position_path).replace("positions", "angles"), delimiter=",").reshape(-1, 22, 3)
    rotations = Rotation.from_euler("xyz", a.reshape(-1, 3), degrees=True).as_matrix().reshape(-1, 22, 3, 3)
    output, global_r = np.empty_like(p), np.empty_like(rotations)
    for joint, parent in enumerate(PARENTS):
        if parent < 0:
            output[:, joint], global_r[:, joint] = p[:, joint], rotations[:, joint]
        else:
            output[:, joint] = output[:, parent] + np.einsum("tij,tj->ti", global_r[:, parent], p[:, joint])
            global_r[:, joint] = global_r[:, parent] @ rotations[:, joint]
    return output


def project(skeleton, angle=45):
    # Oblique views expose knee flexion; y-up mocap becomes y-down camera space.
    theta = np.deg2rad(angle)
    p = skeleton[:, NODES]
    xy = np.stack([p[:, :, 0] * np.cos(theta) + p[:, :, 2] * np.sin(theta), -p[:, :, 1]], axis=-1)
    return np.concatenate([xy, np.ones((*xy.shape[:2], 1))], axis=-1).astype(np.float32)


def motion_signal(skeleton, movement):
    if movement in ["m01", "m05"]:
        return skeleton[:, 0, 1]
    nodes = [15, 19] if movement == "m02" else [16, 20]
    return np.max(skeleton[:, nodes, 1], axis=1) - skeleton[:, 0, 1]


def weak_labels(skeleton, movement):
    if movement not in MOVEMENT_TO_INDEX:
        return np.full(len(skeleton), 16, dtype=np.int64)
    # Offline annotation heuristic, never used as an inference input or count oracle.
    h = gaussian_filter1d(motion_signal(skeleton, movement), 1.3)
    lo, hi = np.quantile(h, [.05, .95])
    relative = (h - lo) / max(hi - lo, 1)
    velocity = np.gradient(relative)
    if movement == "m01":
        relative, velocity = 1 - relative, -velocity
    labels = np.where(relative < .15, 0, np.where(relative > .85, 2, np.where(velocity > 0, 1, 3)))
    return (labels + 4 * MOVEMENT_TO_INDEX[movement]).astype(np.int64)


def main():
    records = []
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for path in sorted((ROOT / "data/raw/uiprmd").glob("*positions*.txt")):
        match = re.match(r"(m\d+)_s(\d+)_e(\d+)_positions(_inc)?", path.stem)
        movement, subject, episode, incorrect = match.groups()
        skeleton = reconstruct(path)
        labels = weak_labels(skeleton, movement)
        key = path.stem.replace("_positions", "")
        arrays[f"{key}_skel"] = skeleton.astype(np.float32)
        arrays[f"{key}_labels"] = labels
        split = "train" if int(subject) <= 6 else "validation" if int(subject) <= 8 else "test"
        records.append({"id": key, "subject": int(subject), "episode": int(episode), "movement": movement,
                        "incorrect": bool(incorrect), "split": split, "frames": len(skeleton),
                        "phase_label_source": "weak_kinematic_offline_v2", "source_file": path.name})
    np.savez_compressed(output / "skeletons.npz", **arrays)
    (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print({s: sum(r["split"] == s for r in records) for s in ["train", "validation", "test"]})
    # A human-readable audit of all correct sit-to-stand clips and phase boundaries.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    audit = ROOT / "reports/figures"
    audit.mkdir(parents=True, exist_ok=True)
    for movement in MOVEMENT_TO_INDEX:
        for subject in range(1, 11):
            fig, axes = plt.subplots(2, 5, figsize=(16, 6))
            for episode, ax in enumerate(axes.flat, 1):
                key = f"{movement}_s{subject:02}_e{episode:02}"
                h = motion_signal(arrays[f"{key}_skel"], movement)
                ax.plot(np.arange(len(h)) / 30, h)
                ax.set_title(key)
                ax.set_xlabel("seconds")
                ax.set_ylabel("annotation signal (source units)")
            fig.tight_layout()
            fig.savefig(audit / f"source_audit_{movement}_s{subject:02}.png", dpi=120)
            plt.close(fig)


if __name__ == "__main__":
    main()
