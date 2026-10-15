"""Model registry: resolve configured model ids to scorers / classifiers.

``rule`` selects the transparent baselines. Any other id refers to a model
package directory ``models/<task>/<model_id>/<version>/`` containing
``meta.json`` (feature schema, window parameters, layout, calibration,
training provenance) plus the weights. Packages are checked for layout and
schema compatibility at load time; a mismatched package is refused, never
silently adapted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from homeease.core.config import AppConfig
from homeease.core.paths import models_dir
from homeease.core.util import get_logger

log = get_logger(__name__)


class ModelLoadError(RuntimeError):
    pass


def registry_index() -> list[dict[str, Any]]:
    out = []
    root = models_dir()
    for meta in sorted(root.glob("*/*/*/meta.json")):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            m["path"] = str(meta.parent)
            out.append(m)
        except (OSError, json.JSONDecodeError):
            log.warning("unreadable model meta %s", meta)
    return out


def find_package(model_ref: str, task: str | None = None) -> Path:
    """model_ref: '<model_id>' (latest version) or '<model_id>@<version>' or a path."""
    p = Path(model_ref)
    if (p / "meta.json").is_file():
        return p
    mid, _, ver = model_ref.partition("@")
    cands = [m for m in registry_index() if m.get("model_id") == mid and (not task or m.get("task") == task)]
    if ver:
        cands = [m for m in cands if m.get("version") == ver]
    if not cands:
        raise ModelLoadError(f"model package not found: {model_ref} (task={task})")
    cands.sort(key=lambda m: m.get("created_at", ""))
    return Path(cands[-1]["path"])


def package_kind(path: Path) -> str | None:
    return json.loads((Path(path) / "meta.json").read_text(encoding="utf-8")).get("kind")


def load_fall_scorer(model_ref: str, task: str, cfg: AppConfig) -> Any:
    """A trained window scorer: the team's packages (gxh_signal, gxh_a4) or a HomeEase package."""
    path = find_package(model_ref, task)
    kind = package_kind(path)
    if kind == "gxh_signal":
        from homeease.ml.fall.accel_rfxgb import AccelRfXgbScorer

        return AccelRfXgbScorer(path, cfg)
    if kind == "gxh_a4":
        from homeease.ml.fall.vision_a4 import VisionA4Scorer

        return VisionA4Scorer(path, cfg)
    from homeease.ml.fall.models import load_window_scorer

    return load_window_scorer(path, cfg)


def build_fall_components(cfg: AppConfig) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Return (accel_scorer, vision_scorer, fusion_kwargs, info)."""
    from homeease.ml.fall.accel import AccelRuleScorer
    from homeease.ml.fall.vision import VisionRuleScorer

    info: dict[str, Any] = {}
    q = cfg.quality
    if cfg.fall.accel_model == "rule":
        accel = AccelRuleScorer(cfg.fall.accel, cfg.acquisition.accel_target_hz, q.max_clock_uncertainty_ms, q.saturation_g)
    else:
        accel = load_fall_scorer(cfg.fall.accel_model, "fall_accel", cfg)
    if cfg.fall.vision_model == "rule":
        vision = VisionRuleScorer(cfg.fall.vision)
    else:
        vision = load_fall_scorer(cfg.fall.vision_model, "fall_vision", cfg)
    info["accel"] = {"model": f"{accel.model_id}@{accel.model_version}", "score_semantics": accel.score_semantics.value}
    info["vision"] = {"model": f"{vision.model_id}@{vision.model_version}", "score_semantics": vision.score_semantics.value}
    fusion_kwargs: dict[str, Any] = {"fusion_backend": cfg.fall.fusion.backend}
    if cfg.fall.fusion.backend == "late_trained":
        from homeease.ml.fall.models import load_trained_fusion

        if not cfg.fall.fusion.trained_model:
            raise ModelLoadError("fall.fusion.trained_model must be set for late_trained")
        fusion_kwargs["trained_fusion"] = load_trained_fusion(find_package(cfg.fall.fusion.trained_model, "fall_fusion"))
    elif cfg.fall.fusion.backend == "feature_fusion":
        from homeease.ml.fall.feature_fusion import build_feature_fusion_engine

        if not cfg.fall.fusion.trained_model:
            raise ModelLoadError("fall.fusion.trained_model must be set for feature_fusion")
        fusion_kwargs = {"fusion_backend": "feature_fusion", "fusion_engine_factory": build_feature_fusion_engine, "fusion_package": str(find_package(cfg.fall.fusion.trained_model, "fall_feature_fusion"))}
    info["fusion"] = {"backend": cfg.fall.fusion.backend, "model": cfg.fall.fusion.trained_model}
    return accel, vision, fusion_kwargs, info


def build_phase_classifier(cfg: AppConfig) -> Any:
    from homeease.ml.exercise.phase import RulePhaseClassifier

    if cfg.exercise.phase_model == "rule":
        return RulePhaseClassifier(cfg.exercise)
    from homeease.ml.exercise.models import load_phase_classifier

    return load_phase_classifier(cfg.exercise.phase_model, cfg)


_MULTI_MODELS: dict[str, Any] = {}


def multi_exercise_package(cfg: AppConfig) -> Path | None:
    """The configured Algorithm 2 package (several exercises), or None for sit-to-stand phase models."""
    if cfg.exercise.phase_model == "rule":
        return None
    try:
        path = find_package(cfg.exercise.phase_model, "exercise_phase_multi")
    except ModelLoadError:
        return None
    return path


def supported_activities(cfg: AppConfig) -> list[str]:
    if multi_exercise_package(cfg) is not None:
        from homeease.ml.exercise.multi import ACTIVITIES

        return list(ACTIVITIES)
    return ["chair_sit_to_stand"]


def build_exercise_tracker(cfg: AppConfig, session_id: str, target: int, definition: str | None = None, activity: str = "chair_sit_to_stand") -> Any:
    """The configured tracker: Algorithm 2 (all four exercises) or the sit-to-stand phase classifier + counter."""
    from homeease.ml.exercise.multi import canonical_activity

    activity = canonical_activity(activity)
    path = multi_exercise_package(cfg)
    if path is not None:
        from homeease.ml.exercise.multi import MultiExerciseModel, MultiExerciseTracker

        key = str(path)
        if key not in _MULTI_MODELS:
            _MULTI_MODELS[key] = MultiExerciseModel(path)  # read-only weights, shared by sessions
        return MultiExerciseTracker(cfg.exercise, session_id, target, _MULTI_MODELS[key], activity)
    if activity != "chair_sit_to_stand":
        raise ModelLoadError(f"exercise {activity} needs the Algorithm 2 package (exercise.phase_model)")
    from homeease.ml.exercise.tracker import ExerciseTracker

    return ExerciseTracker(cfg.exercise, session_id, target, definition, build_phase_classifier(cfg), cfg.acquisition.pose.min_keypoint_score)
