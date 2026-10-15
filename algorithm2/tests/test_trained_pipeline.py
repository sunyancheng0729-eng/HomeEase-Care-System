"""Deployment parity checks use real selected checkpoints, never mock logits."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from homeease.api import create_app
from homeease.engine import ExerciseSession, Predictor
from homeease.exercises import EXERCISES
from homeease.features import extract_sequence
from scripts.prepare_data import project
from scripts.replay_demo import coco_points
from scripts.train_compare import infer

pytestmark = pytest.mark.skipif(not (ROOT / "artifacts/selected.json").exists(), reason="Training must finish first")


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def test_real_checkpoint_stream_matches_offline_windows(predictor):
    skeletons = np.load(ROOT / "data/processed/skeletons.npz")
    poses = project(skeletons["m05_s09_e01_skel"])
    stream = predictor.stream()
    actual = np.array([stream.update(p, i / 30) for i, p in enumerate(poses)])
    expected = []
    for member in predictor.members:
        features = extract_sequence(poses, mode=member["mode"])
        expected.append(infer(member["model"], features, member["mean"], member["std"], "cpu", context=member["context"]))
    np.testing.assert_allclose(actual, np.mean(expected, axis=0), atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("exercise", list(EXERCISES))
def test_real_http_and_sdk_are_identical(predictor, exercise):
    source = json.loads((ROOT / "examples" / f"{exercise}_replay.json").read_text())
    frames = source["frames"][:150]
    sdk = ExerciseSession(predictor, exercise=exercise)
    for f in frames:
        sdk.update(f["timestamp_ms"], f["keypoints"], f["person_id"])
    with TestClient(create_app(predictor)) as client:
        created = client.post("/v1/sessions", json={"exercise": exercise})
        created.raise_for_status()
        sid = created.json()["session_id"]
        for start in range(0, len(frames), 30):
            response = client.post(f"/v1/sessions/{sid}/frames", json={"request_id": str(start), "frames": frames[start:start + 30]})
            response.raise_for_status()
        assert response.json()["count"] == sdk.counter.count
        assert response.json()["phase"] == sdk.phase
        assert response.json()["cycles"] == sdk.counter.events


def test_openapi_has_typed_outputs():
    schema = create_app().openapi()
    assert "BatchResult" in schema["components"]["schemas"]
    assert "cycles" in schema["components"]["schemas"]["BatchResult"]["properties"]
