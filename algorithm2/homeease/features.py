"""Shared camera-space features: COCO joints 5..16, x right and y down."""
import numpy as np

JOINT_NAMES = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle"]
FEATURE_DIMS = {"coords": 24, "kinematics": 16, "hybrid": 64, "stable80": 80}


def cosine(a, b, c):
    u, v = a - b, c - b
    return np.sum(u * v, axis=-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-6)


def frame_features(pose):
    p = np.asarray(pose, dtype=np.float32)[:, :2]
    hips = (p[6] + p[7]) * .5
    ankles = (p[10] + p[11]) * .5
    shoulders = (p[0] + p[1]) * .5
    leg = .5 * (np.linalg.norm(p[6] - p[8]) + np.linalg.norm(p[8] - p[10])
                + np.linalg.norm(p[7] - p[9]) + np.linalg.norm(p[9] - p[11]))
    scale = max(float(leg), 1e-4)
    coords = ((p - hips) / scale).reshape(-1)
    torso = shoulders - hips
    kin = np.asarray([cosine(p[6], p[8], p[10]), cosine(p[7], p[9], p[11]),
                      cosine(p[0], p[6], p[8]), cosine(p[1], p[7], p[9]),
                      (ankles[1] - hips[1]) / scale,
                      torso[0] / max(np.linalg.norm(torso), 1e-4),
                      -torso[1] / max(np.linalg.norm(torso), 1e-4),
                      np.linalg.norm(p[10] - p[11]) / scale], dtype=np.float32)
    return coords, kin


def stable_features(pose):
    """Torso scaling avoids shrinking the denominator when a leg flexes."""
    p = np.asarray(pose, dtype=np.float32)[:, :2]
    hips = (p[6] + p[7]) * .5
    shoulders = (p[0] + p[1]) * .5
    scale = max(float(np.linalg.norm(shoulders - hips)) * 2, 1e-4)
    q = (p - hips) / scale
    _, angles = frame_features(pose)
    extra = np.asarray([
        q[8, 1], q[9, 1], q[10, 1], q[11, 1],
        min(q[8, 1], q[9, 1]), min(q[10, 1], q[11, 1]),
        q[8, 1] - q[9, 1], q[10, 1] - q[11, 1],
    ], dtype=np.float32)
    # Angles are dimensionless; vertical limb signals retain unilateral motion.
    return np.concatenate([q.reshape(-1), angles, extra])


class FeatureExtractor:
    """Causal EMA and backward difference, identical in training and serving."""
    def __init__(self, mode="hybrid", alpha=.65):
        self.mode, self.alpha = mode, alpha
        self.previous = None
        self.timestamp = None

    def reset(self):
        self.previous = self.timestamp = None

    def update(self, pose, timestamp):
        if self.mode == "stable80":
            current = stable_features(pose)
        else:
            coords, kin = frame_features(pose)
            current = np.concatenate([coords, kin])
        if self.previous is None:
            filtered, velocity = current, np.zeros_like(current)
        else:
            dt = max(timestamp - self.timestamp, 1 / 120)
            filtered = self.alpha * current + (1 - self.alpha) * self.previous
            velocity = np.clip((filtered - self.previous) / dt, -8, 8)
        self.previous, self.timestamp = filtered, timestamp
        if self.mode == "stable80":
            return np.concatenate([filtered, velocity])
        if self.mode == "coords":
            return filtered[:24]
        if self.mode == "kinematics":
            return np.concatenate([filtered[24:], velocity[24:]])
        return np.concatenate([filtered, velocity[:24], velocity[24:]])


def extract_sequence(poses, fps=30, mode="hybrid"):
    extractor = FeatureExtractor(mode)
    return np.asarray([extractor.update(pose, i / fps) for i, pose in enumerate(poses)], dtype=np.float32)
