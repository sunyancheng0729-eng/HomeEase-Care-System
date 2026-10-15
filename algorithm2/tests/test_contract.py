import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.api import create_app
from homeease.counter import CycleCounter
from homeease.engine import ExerciseSession
from homeease.exercises import EXERCISE_NAMES
from homeease.models import PhaseModel, HierarchicalPhaseModel


class StubStream:
    def reset(self):
        pass

    def update(self, pose, timestamp):
        result = np.full(17, .0001)
        result[int(pose[0, 0])] = .9984
        return result


class StubPredictor:
    selected = {"min_dwell_s": .033, "confidence_threshold": 0, "model": "contract-test"}

    def stream(self):
        return StubStream()


def frame(label, timestamp, confidence=1, person="primary"):
    return {"timestamp_ms": timestamp, "person_id": person, "keypoints": [[float(label), float(i), confidence] for i in range(17)]}


def cycle(offset=0, start=0):
    return [frame(offset + phase, start + i * 100) for i, phase in enumerate([0, 0, 1, 1, 2, 2, 3, 3, 0, 0])]


@pytest.mark.parametrize("exercise", EXERCISE_NAMES)
def test_four_exercises_complete_cycles(exercise):
    session = ExerciseSession(StubPredictor(), exercise=exercise)
    for f in cycle(EXERCISE_NAMES.index(exercise) * 4):
        session.update(f["timestamp_ms"], f["keypoints"])
    assert session.snapshot()["count"] == 1


def test_partial_and_backwards_jitter_do_not_duplicate():
    counter = CycleCounter(min_dwell_s=.033)
    phases = ["rest", "outbound", "hold", "outbound", "hold", "return", "hold", "return", "rest"]
    for i, p in enumerate(phases):
        counter.update(p, i * .2)
        counter.update(p, i * .2 + .1)
    assert counter.count == 1
    counter.update("rest", 10)
    assert counter.count == 1


@pytest.mark.parametrize("reason", ["pause", "occlusion", "gap", "person"])
def test_interruption_cannot_complete_open_cycle(reason):
    session = ExerciseSession(StubPredictor())
    for f in cycle()[:6]:
        session.update(f["timestamp_ms"], f["keypoints"])
    if reason == "pause":
        session.control("pause")
        session.control("resume")
    elif reason == "occlusion":
        session.update(550, frame(2, 550, 0)["keypoints"])
    elif reason == "person":
        session.update(550, frame(2, 550)["keypoints"], "new-person")
    start = 1000 if reason == "gap" else 600
    for i, phase in enumerate([3, 3, 0, 0]):
        session.update(start + i * 100, frame(phase, 0)["keypoints"], "new-person" if reason == "person" else "primary")
    assert session.counter.count == 0


@pytest.mark.parametrize("kind", ["tcn", "gru", "transformer"])
def test_model_cannot_read_future(kind):
    torch.manual_seed(42)
    model = PhaseModel(16, kind=kind, width=16, blocks=2).eval()
    x = torch.randn(1, 20, 16)
    changed = x.clone()
    changed[:, 10:] = torch.randn_like(changed[:, 10:]) * 100
    with torch.inference_mode():
        torch.testing.assert_close(model(x)[:, :10], model(changed)[:, :10], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("kind", ["tcn", "gru"])
def test_hierarchical_model_is_causal_and_preserves_action_probabilities(kind):
    model = HierarchicalPhaseModel(80, kind=kind, width=16, blocks=2).eval()
    x = torch.randn(2, 96, 80)
    changed = x.clone()
    changed[:, 48:] *= -100
    with torch.inference_mode():
        logits = model(x)
        torch.testing.assert_close(logits[:, :48], model(changed)[:, :48], atol=1e-5, rtol=1e-5)
        actions, _ = model.components(x)
        joint = logits.softmax(-1)
        marginal = torch.cat([joint[..., :16].reshape(2, 96, 4, 4).sum(-1), joint[..., 16:]], -1)
        torch.testing.assert_close(marginal, actions.softmax(-1))


def test_stable_features_are_translation_and_scale_invariant():
    from homeease.features import extract_sequence
    rng = np.random.default_rng(14)
    pose = rng.normal(size=(40, 12, 3)).astype(np.float32)
    pose[:, :, 2] = 1
    transformed = pose.copy()
    transformed[:, :, :2] = pose[:, :, :2] * 37 + [400, 200]
    np.testing.assert_allclose(extract_sequence(pose, mode="stable80"),
                               extract_sequence(transformed, mode="stable80"), atol=1e-4, rtol=1e-4)


def test_lazy_training_windows_match_causal_reference():
    from scripts.optimize_top import CausalWindowDataset
    from scripts.train_compare import windows
    x = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    y = np.arange(20, dtype=np.int64) % 17
    mean, std = x.mean(0), np.maximum(x.std(0), .1)
    data = CausalWindowDataset([(x, y)], mean, std, 12, stride=1)
    expected = windows((x - mean) / std, 12)
    expected_y = windows(y[:, None], 12).squeeze(-1)[:, -8:]
    for i in range(len(data)):
        xx, yy = data[i]
        np.testing.assert_array_equal(xx.numpy(), expected[i])
        np.testing.assert_array_equal(yy.numpy(), expected_y[i])


def test_http_idempotency_validation_and_lifecycle():
    with TestClient(create_app(StubPredictor())) as client:
        assert client.get("/health").json()["status"] == "ready"
        sid = client.post("/v1/sessions", json={"exercise": "squat"}).json()["session_id"]
        path = f"/v1/sessions/{sid}"
        body = {"request_id": "a", "frames": cycle(4)}
        response = client.post(path + "/frames", json=body)
        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert client.post(path + "/frames", json=body).json() == response.json()
        altered = {"request_id": "a", "frames": [frame(4, 1100)]}
        assert client.post(path + "/frames", json=altered).status_code == 409
        assert client.post(path + "/frames", json={"request_id": "b", "frames": [frame(4, 500)]}).status_code == 409
        invalid = {"request_id": "b", "frames": [frame(4, 1100), frame(4, 1200, 2)]}
        assert client.post(path + "/frames", json=invalid).status_code == 422
        too_large = frame(4, 1200)
        too_large["keypoints"][0][0] = 1e100
        assert client.post(path + "/frames", json={"request_id": "huge", "frames": [frame(4, 1100), too_large]}).status_code == 422
        assert client.post(path + "/frames", json={"request_id": "huge-time", "frames": [frame(4, 1e100)]}).status_code == 422
        assert client.get(path).json()["last_timestamp_ms"] == 900
        client.post(path + "/control", json={"action": "pause"}).raise_for_status()
        assert client.post(path + "/frames", json=altered).status_code == 409
        client.post(path + "/control", json={"action": "end"}).raise_for_status()
        assert client.post(path + "/control", json={"action": "resume"}).status_code == 409
        assert client.delete(path).status_code == 204
        assert client.get(path).status_code == 404


def test_splits_are_subject_disjoint():
    import json
    records = json.loads((ROOT / "data/processed/records.json").read_text())
    subjects = {s: {r["subject"] for r in records if r["split"] == s} for s in ["train", "validation", "test"]}
    assert subjects == {"train": set(range(1, 7)), "validation": {7, 8}, "test": {9, 10}}


def test_pause_time_is_not_active_time():
    session = ExerciseSession(StubPredictor())
    session.update(0, frame(0, 0)["keypoints"])
    session.update(100, frame(0, 100)["keypoints"])
    session.control("pause")
    session.control("resume")
    session.update(200, frame(0, 200)["keypoints"])
    assert session.active_duration_s == pytest.approx(.1)


def test_ordered_posterior_requires_all_phases():
    counter = CycleCounter(min_dwell_s=.033)
    # A short peak need not be the most likely class, but still supplies evidence.
    samples = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, .6, .4, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]]
    for i, p in enumerate(samples):
        counter.update_probabilities(p, i * .3, .35)
        counter.update_probabilities(p, i * .3 + .1, .35)
    assert counter.count == 1
    missing_peak = CycleCounter(min_dwell_s=.033)
    for i, p in enumerate([samples[0], samples[1], samples[3], samples[4]]):
        missing_peak.update_probabilities(p, i * .3, .35)
        missing_peak.update_probabilities(p, i * .3 + .1, .35)
    assert missing_peak.count == 0


def test_degenerate_high_confidence_pose_abstains():
    session = ExerciseSession(StubPredictor())
    state = session.update(0, [[0, 0, 1]] * 17)
    assert state["quality"] == "degenerate_pose"
    assert state["count"] == 0


def test_inferred_return_still_requires_peak_and_visible_baseline():
    for phases, expected in [([0, 1, 2, 0], 1), ([0, 1, 0], 0), ([0, 1, 2], 0)]:
        counter = CycleCounter(min_dwell_s=.033)
        t = 0
        for phase in phases:
            for _ in range(6):
                counter.update_probabilities(np.eye(5)[phase], t, .35, infer_return=True)
                t += .1
        assert counter.count == expected
