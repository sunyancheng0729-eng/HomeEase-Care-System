"""Sound-event model packages: ``models/audio_sound/<model_id>/<version>/`` (see ml/registry.py).

``save_sound_package`` is the training-side helper (the next work package trains a real model and
calls it); ``load_sound_classifier`` is what the online monitor uses. A package is refused, never
silently adapted, when its layout does not match what this code can run.

meta.json (schema "audio_sound/1"), written next to the weights::

    {
      "schema": "audio_sound/1", "task": "audio_sound", "model_id": "sound_events", "version": "20261010-120000",
      "kind": "sklearn" | "torch", "weights": "model.joblib" | "model.pt", "sha256": "<weights file>",
      "classes": ["HELP", "IMPACT", "ABNORMAL", "DAILY"],          # column order of predict_proba (never UNKNOWN)
      "input": {"sample_rate": 16000, "window_s": 2.0,
                "representation": "clip_features" | "logmel_patch" | "yamnet_embedding",
                "patch_frames": 256,                                 # logmel_patch only (64 mel bands, fixed by sound.logmel_patch)
                "normalization": "none" | "peak" | "rms"},           # level handling applied to every live window before features
      "thresholds": {"min_conf": 0.55, "chosen_on": "dev", ...},     # chosen on development data, never on the test fold
                                                                     # optional "per_class": {"HELP": 0.7, ...} cut-offs below which a window is UNKNOWN
      "calibration": null | {"method": "temperature", "T": 1.4},     # torch models only
      "training": {...},                                              # datasets, split description, sample sizes, seed, metrics reference
      "code_version": "...", "created_at": "..."
    }

kind=torch stores ``{"state_dict", "cfg"}`` of ``homeease.ml.audio.sound.SmallCNN``; kind=sklearn a
joblib pipeline fitted on ``sound.clip_features`` vectors, or (representation ``yamnet_embedding``) on the
YAMNet window embedding of ml/audio/embed.py: then ``input.embedder`` = "yamnet" and ``input.embedder_sha256``
pin the ONNX file that must be present under models/audio/yamnet/. No audio, spectrogram or embedding of a
recording is ever stored in a package.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from homeease.core.paths import models_dir
from homeease.core.util import code_version, dumps, get_logger, sha256_file
from homeease.ml.audio.monitor import CLASS_NAMES, SR
from homeease.ml.registry import ModelLoadError, find_package

log = get_logger(__name__)

TASK = "audio_sound"
SCHEMA = "audio_sound/1"
REPRESENTATIONS = ("clip_features", "logmel_patch", "yamnet_embedding")
KINDS = ("sklearn", "torch")
NORMALIZATIONS = ("none", "peak", "rms")


def save_sound_package(
    model: Any,
    kind: str,
    classes: Sequence[str],
    input_spec: dict[str, Any],
    model_id: str,
    version: str | None = None,
    thresholds: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Write weights + meta.json and return the package directory. ``root`` defaults to ``models/``."""
    version = version or time.strftime("%Y%m%d-%H%M%S")
    meta: dict[str, Any] = {
        "schema": SCHEMA, "task": TASK, "model_id": model_id, "version": version, "kind": kind, "classes": [str(c) for c in classes],
        "input": {"sample_rate": SR, **input_spec}, "thresholds": thresholds or {}, "calibration": calibration, "training": training or {},
    }
    _check_meta(meta)
    d = Path(root or models_dir()) / TASK / model_id / version
    if (d / "meta.json").exists():
        raise FileExistsError(f"sound package {model_id}@{version} already exists: {d} (pick a new version, packages are immutable)")
    d.mkdir(parents=True, exist_ok=True)
    if kind == "sklearn":
        import joblib

        wp = d / "model.joblib"
        joblib.dump(model, wp)
    else:
        import torch

        wp = d / "model.pt"
        torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, wp)
    meta.update(weights=wp.name, sha256=sha256_file(wp), code_version=code_version(), created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (d / "meta.json").write_text(dumps(meta, indent=2), encoding="utf-8")
    return d


def _check_meta(meta: dict[str, Any]) -> None:
    if meta.get("task") != TASK:
        raise ModelLoadError(f"package task {meta.get('task')!r} is not {TASK}")
    if meta.get("kind") not in KINDS:
        raise ModelLoadError(f"unsupported package kind {meta.get('kind')!r} (supported: {KINDS})")
    classes = meta.get("classes")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes) or any(c not in CLASS_NAMES for c in classes):
        raise ModelLoadError(f"package classes must be distinct members of {CLASS_NAMES} (UNKNOWN is never a class): {classes}")
    inp = meta.get("input") or {}
    if int(inp.get("sample_rate", 0)) != SR:
        raise ModelLoadError(f"package expects {inp.get('sample_rate')} Hz audio; the pipeline delivers {SR} Hz")
    if inp.get("representation") not in REPRESENTATIONS:
        raise ModelLoadError(f"unsupported input representation {inp.get('representation')!r} (supported: {REPRESENTATIONS})")
    if not float(inp.get("window_s", 0)) > 0:
        raise ModelLoadError("package must declare input.window_s > 0")
    if inp.get("normalization", "none") not in NORMALIZATIONS:
        raise ModelLoadError(f"unsupported input normalization {inp.get('normalization')!r} (supported: {NORMALIZATIONS})")
    if inp["representation"] == "logmel_patch" and int(inp.get("n_mels", 64)) != 64:
        raise ModelLoadError("logmel_patch uses 64 mel bands")
    if inp["representation"] == "yamnet_embedding" and (meta["kind"] != "sklearn" or inp.get("embedder") != "yamnet" or not inp.get("embedder_sha256")):
        raise ModelLoadError("yamnet_embedding packages are sklearn models that declare input.embedder='yamnet' and input.embedder_sha256")
    cal = meta.get("calibration")
    if cal and (cal.get("method") != "temperature" or meta["kind"] != "torch" or not float(cal.get("T", 0)) > 0):
        raise ModelLoadError(f"unsupported calibration {cal} (only temperature scaling of torch models)")


def normalize_window(x: np.ndarray, mode: str) -> np.ndarray:
    """Level handling shared by training (ml/audio/sound_train.py) and the live classifier, so both see the same input."""
    if mode == "peak":
        pk = float(np.max(np.abs(x))) if len(x) else 0.0
        return x * (0.5 / pk) if pk > 1e-4 else x
    if mode == "rms":
        r = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
        return (x * (0.1 / r)).astype(np.float32) if r > 1e-5 else x
    return x


class ModelSoundClassifier:
    """A loaded package behind the ``SoundClassifier`` interface of monitor.py."""

    def __init__(self, model: Any, meta: dict[str, Any], embedder: Any = None) -> None:
        self.model, self.meta, self.embedder = model, meta, embedder
        self.model_id, self.model_version = meta["model_id"], meta["version"]
        self.classes: list[str] = list(meta["classes"])
        inp = meta["input"]
        self.window_s = float(inp["window_s"])
        self.representation = inp["representation"]
        self.patch_frames = int(inp.get("patch_frames", 256))
        self.normalization = inp.get("normalization", "none")
        th = (meta.get("thresholds") or {}).get("min_conf")
        self.min_conf = float(th) if th is not None else None
        per = (meta.get("thresholds") or {}).get("per_class") or {}
        self.min_conf_by_class: dict[str, float] = {str(c): float(v) for c, v in per.items() if str(c) in self.classes}
        self.temperature = float(meta["calibration"]["T"]) if meta.get("calibration") else 1.0
        self.kind = meta["kind"]

    def describe(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "version": self.model_version, "kind": self.kind, "classes": self.classes, "window_s": self.window_s,
                "representation": self.representation, "normalization": self.normalization, "min_conf": self.min_conf, "min_conf_by_class": self.min_conf_by_class, "sha256": self.meta.get("sha256")}

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Live windows arrive at raw microphone level; the package declares how training handled level."""
        return normalize_window(x, self.normalization)

    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        n = int(round(self.window_s * SR))
        x = np.asarray(window, dtype=np.float32).reshape(-1)
        x = self._normalize(x[:n] if len(x) >= n else np.pad(x, (0, n - len(x))))
        if self.kind == "sklearn":
            if self.representation == "yamnet_embedding":
                feats = self.embedder.window_embedding(x)
            else:
                from homeease.ml.audio.sound import clip_features

                feats = clip_features(x)
            return self._order(self.model.predict_proba(feats[None])[0], getattr(self.model, "classes_", None))
        import torch

        from homeease.ml.audio.sound import logmel_patch

        with torch.no_grad():
            logits = self.model(torch.from_numpy(logmel_patch(x, self.patch_frames)[None].astype(np.float32)))
            return torch.softmax(logits / self.temperature, dim=-1)[0].numpy().astype(np.float64)

    def _order(self, p: np.ndarray, model_classes: Any) -> np.ndarray:
        """Return probabilities in meta['classes'] order whatever labels the sklearn model was fitted with."""
        p = np.asarray(p, dtype=np.float64)
        if model_classes is None or len(model_classes) != len(p):
            raise ValueError("classifier output does not match its class list")
        out = np.zeros(len(self.classes))
        for col, c in enumerate(model_classes):
            idx = self.classes.index(str(c)) if str(c) in self.classes else int(c) if str(c).lstrip("-").isdigit() else -1
            if not 0 <= idx < len(out):
                raise ValueError(f"unknown class label {c!r} in the model")
            out[idx] = p[col]
        return out


def load_sound_classifier(model_ref: str, embedder: Any = None, expected_sha256: str | None = None) -> ModelSoundClassifier:
    """Resolve ``<model_id>``, ``<model_id>@<version>`` or a package path; verify and load it.

    ``expected_sha256`` pins the weights file from OUTSIDE the package (``audio.sound_monitor.model_sha256``). The checksum
    in meta.json sits in the same folder as the weights, so it detects a damaged file, not a replaced one; a scikit-learn
    package is a pickle and executing it is as trusted as the folder it lives in (``models/`` is a trust boundary).

    Raises ``ModelLoadError`` (missing, incompatible or corrupt package); callers turn that into
    ``degraded:model_missing`` instead of a crash."""
    path = find_package(model_ref, TASK)
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ModelLoadError(f"unreadable meta.json in {path}: {e}") from e
    _check_meta(meta)
    wp = path / str(meta.get("weights", ""))
    if not wp.is_file():
        raise ModelLoadError(f"weights file missing: {wp}")
    actual = sha256_file(wp)
    if actual != meta.get("sha256"):
        raise ModelLoadError(f"weights checksum mismatch for {path}")
    if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
        raise ModelLoadError(f"weights of {path} do not match the sha256 pinned in the configuration")
    try:
        if meta["kind"] == "sklearn":
            import joblib

            if expected_sha256 is None:
                log.info("sklearn sound model %s loaded without a pinned hash (audio.sound_monitor.model_sha256): models/ is trusted as it is", path.name)

            model = joblib.load(wp)
        else:
            import torch

            from homeease.ml.audio.sound import SmallCNN

            ck = torch.load(wp, map_location="cpu", weights_only=True)
            if ck["cfg"].get("arch") != "small_cnn":
                raise ModelLoadError(f"unsupported network {ck['cfg'].get('arch')!r} (supported: small_cnn)")
            model = SmallCNN(int(ck["cfg"]["n_out"]))
            model.load_state_dict(ck["state_dict"])
            model.eval()
            if int(ck["cfg"]["n_out"]) != len(meta["classes"]):
                raise ModelLoadError("network output size does not match the package class list")
    except ModelLoadError:
        raise
    except Exception as e:
        raise ModelLoadError(f"could not load weights of {path}: {type(e).__name__}: {e}") from e
    if meta["input"]["representation"] == "yamnet_embedding":
        # the package only holds the small classifier on top of the embedding: the embedder file must be the one it was trained with
        try:
            if embedder is None:
                from homeease.ml.audio.embed import shared_embedder

                embedder = shared_embedder()
        except Exception as e:  # missing file / onnxruntime
            raise ModelLoadError(f"embedder for {path} unavailable: {type(e).__name__}: {e}") from e
        if embedder.sha256 != meta["input"]["embedder_sha256"]:
            raise ModelLoadError(f"embedder checksum mismatch for {path}: package expects {meta['input']['embedder_sha256'][:12]}, found {str(embedder.sha256)[:12]}")
    log.info("sound model loaded: %s@%s (%s)", meta["model_id"], meta["version"], meta["kind"])
    return ModelSoundClassifier(model, meta, embedder)
