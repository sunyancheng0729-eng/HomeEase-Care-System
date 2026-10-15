"""Typed configuration.

Defaults live in these models; ``config/default.yaml`` mirrors them for
documentation (a test asserts the two do not drift). Overlay order:
model defaults < YAML file < environment variables ``HOMEEASE__SECTION__KEY``.
Secrets never live in YAML; they come from ``var/secrets`` or the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from homeease.core.paths import project_root

SourceModeLit = Literal["live", "replay", "synthetic"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeCfg(_M):
    mode: SourceModeLit = "live"
    timezone: str = "Asia/Singapore"
    synthetic_fallback: bool = False
    tick_ms: int = Field(100, ge=20, le=1000)
    subject_id: str = "elder-01"
    room_id: str = "living-room"
    replay_recording: str | None = None
    replay_speed: float = Field(1.0, gt=0, le=50)
    replay_loop: bool = False

    @model_validator(mode="after")
    def _no_silent_synthetic(self) -> "RuntimeCfg":
        if self.mode == "live" and self.synthetic_fallback:
            raise ValueError(
                "runtime.synthetic_fallback must stay false in live mode: real inputs "
                "must never be silently replaced by simulated data"
            )
        return self


class ServerCfg(_M):
    host: str = "0.0.0.0"
    port: int = 8443
    https: bool = True
    certfile: str = "var/certs/server.crt"
    keyfile: str = "var/certs/server.key"
    database_url: str = "sqlite:///var/homeease.db"
    session_ttl_hours: int = 72
    cors_origins: list[str] = Field(default_factory=list)
    max_body_bytes: int = 262_144
    login_attempts_per_minute: int = 10


class DemoCfg(_M):
    """Demonstration mode: no accounts or passwords; the pages offer tabs that switch between the
    older adult, caregiver and technical views. For demos on the node itself, not for deployment."""

    enabled: bool = False
    loopback_only: bool = True  # role switching is accepted only from a browser on the node itself


class DebugCfg(_M):
    enabled: bool = True
    allow_injection: bool = True
    skeleton_preview: Literal["off", "local_only"] = "local_only"


class CameraCfg(_M):
    enabled: bool = True
    source: str | int = 0
    # 15 fps throughout: the A4 vision model scores 2 s at 15 fps; at 5 fps idle its emergency recall on
    # UR Fall fell from 0.233 to 0.167 (docs/results.md). Lower only with the rule vision model.
    base_fps: float = Field(15.0, gt=0, le=60)
    event_fps: float = Field(15.0, gt=0, le=60)
    exercise_fps: float = Field(15.0, gt=0, le=60)
    event_boost_s: float = 10.0
    width: int = 640
    height: int = 480
    read_timeout_ms: int = 2000
    reconnect_backoff_ms: list[int] = Field(default_factory=lambda: [500, 1000, 2000, 5000])


class PoseCfg(_M):
    backend: Literal["mediapipe", "rtmpose", "gxh_p1", "none"] = "mediapipe"
    mediapipe_model: str = "models/pose/mediapipe/pose_landmarker_lite.task"
    mediapipe_delegate: Literal["gpu", "cpu"] = "gpu"
    rtmpose_det: str = "models/pose/rtmpose/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
    rtmpose_pose: str = "models/pose/rtmpose/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx"
    # the team's video front end (Algorithm 1): YOLOv8n person box + P1 U-Net keypoints
    gxh_yolo: str = "models/pose/gxh_p1/yolov8n-pose.onnx"
    gxh_pose: str = "models/pose/gxh_p1/pose_p1.pt"
    min_keypoint_score: float = Field(0.3, ge=0, le=1)
    max_people: int = Field(3, ge=1, le=6)
    # Keypoint scores are not on the same scale across backends (BlazePose "visibility" vs
    # RTMPose SimCC peak response). Scores are multiplied by this per-family factor (then
    # clipped to 1) before the shared validity threshold. Raw scores stay in the files.
    score_scale: dict[str, float] = Field(default_factory=lambda: {"mediapipe": 1.0, "rtmpose": 1.43, "gxh": 1.0, "synthetic": 1.0})


class AcquisitionCfg(_M):
    accel_target_hz: float = 50.0
    accel_batch_ms: int = 500
    accel_max_batch_samples: int = 600
    accel_buffer_seconds: float = 60.0
    late_batch_ms: int = 5000
    camera: CameraCfg = Field(default_factory=CameraCfg)
    pose: PoseCfg = Field(default_factory=PoseCfg)
    zone: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])

    @field_validator("zone")
    @classmethod
    def _zone(cls, v: list[float]) -> list[float]:
        if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
            raise ValueError("zone must be [x0, y0, x1, y1] normalised with x0<x1, y0<y1")
        return v


class QualityCfg(_M):
    max_clock_uncertainty_ms: float = 150.0
    clock_resync_interval_s: float = 60.0
    clock_sync_max_age_s: float = 300.0
    fusion_time_tolerance_ms: float = 1000.0
    accel_stale_ms: float = 3000.0
    camera_stale_ms: float = 2000.0
    accel_min_rate_hz: float = 20.0
    recovery_stable_ms: float = 1500.0
    accel_gap_interp_max_ms: float = 120.0
    saturation_g: float = 7.8
    pose_min_valid_core: int = 6
    carriage_still_minutes: float = 15.0
    max_model_latency_ms: float = 500.0
    max_queue_backlog: int = 200


class AccelRuleCfg(_M):
    impact_threshold_g: float = 2.0
    post_start_s: float = 0.5
    post_min_s: float = 1.5
    post_window_s: float = 2.5
    still_std_g: float = 0.18
    strong_score: float = 0.6
    filter: Literal["none", "lowpass"] = "lowpass"
    lowpass_hz: float = 15.0
    window_s: float = 4.0
    step_ms: int = 250


class VisionRuleCfg(_M):
    descent_tl: float = 0.9
    descent_with_tilt_tl: float = 0.55
    descent_max_s: float = 1.6
    min_speed_tl_s: float = 0.8
    low_hip_tl: float = 1.0
    floor_hip_tl: float = 1.45
    lying_torso_deg: float = 55.0
    persist_min_s: float = 1.2
    persist_window_s: float = 2.5
    persist_fraction: float = 0.6
    recovery_tl: float = 0.4
    recovery_s: float = 0.5
    strong_score: float = 0.6
    window_s: float = 6.0
    step_ms: int = 200
    fps: float = 15.0


class FusionCfg(_M):
    backend: Literal["late_rule_basic", "late_quality_gated", "late_trained", "feature_fusion"] = (
        "late_quality_gated"
    )
    emergency_requires_two_valid_modalities: bool = True
    min_quality: float = 0.5
    pending_min_score: float = 0.5
    fusion_wait_ms: float = 4500.0
    weights: dict[str, float] = Field(default_factory=lambda: {"accel": 1.0, "vision": 1.0})
    trained_model: str | None = None

    @model_validator(mode="after")
    def _gate(self) -> "FusionCfg":
        if not self.emergency_requires_two_valid_modalities:
            raise ValueError(
                "fall.fusion.emergency_requires_two_valid_modalities is a product rule "
                "from the proposal (dual-channel emergency gate) and cannot be disabled here"
            )
        return self


class FallCfg(_M):
    accel_model: str = "gxh-signal-rfxgb"  # Algorithm 1 accelerometer (team package); "rule" = transparent baseline
    vision_model: str = "gxh-video-a4"  # Algorithm 1 vision A4 ensemble (team package); "rule" = baseline
    # A4 detection is confirmed only if the person does not get back up within this time (0 = off):
    # squats, sitting down and picking things up recover, a fall stays down.
    vision_verify_s: float = Field(1.5, ge=0.0, le=4.0)
    accel: AccelRuleCfg = Field(default_factory=AccelRuleCfg)
    vision: VisionRuleCfg = Field(default_factory=VisionRuleCfg)
    fusion: FusionCfg = Field(default_factory=FusionCfg)
    merge_gap_s: float = 60.0
    cooldown_s: float = 120.0


class ExerciseCfg(_M):
    repetition_definition: Literal["full_cycle", "stand_complete"] = "full_cycle"
    phase_model: str = "ag2-opt-tcn-b4-w48"  # Algorithm 2, four exercises (team package); "rule" = sit-to-stand baseline
    sit_enter: float = 0.35
    sit_exit: float = 0.45
    stand_enter: float = 0.75
    stand_exit: float = 0.65
    velocity_tl_s: float = 0.15
    max_torso_deg: float = 50.0
    min_phase_s: float = 0.25
    min_cycle_s: float = 1.2
    max_cycle_s: float = 15.0
    max_unknown_s: float = 0.7
    quality_pause_s: float = 1.0
    resume_stable_s: float = 0.6
    default_target: int = 10


class EscalationCfg(_M):
    emergency_ack_timeout_s: float = 30.0
    pending_ack_timeout_s: float = 120.0
    handling_reminder_s: float = 120.0
    reminder_interval_s: float = 120.0
    max_reminders: int = 10
    notification_ttl_s: float = 6 * 3600.0
    delayed_delivery_threshold_s: float = 60.0
    retry_base_s: float = 2.0
    retry_max_s: float = 60.0
    retry_max_attempts: int = 8


class ChannelToggle(_M):
    enabled: bool = False


class WebPushCfg(_M):
    enabled: bool = False
    vapid_subject: str = "mailto:admin@homeease.local"


class WebhookCfg(_M):
    enabled: bool = False
    url: str | None = None
    timeout_s: float = 5.0


class NotificationsCfg(_M):
    inapp: ChannelToggle = Field(default_factory=lambda: ChannelToggle(enabled=True))
    webpush: WebPushCfg = Field(default_factory=WebPushCfg)
    webhook: WebhookCfg = Field(default_factory=WebhookCfg)
    test_sink: ChannelToggle = Field(default_factory=ChannelToggle)
    test_events_external: bool = False


class ActivityGateCfg(_M):
    """Shared audio front end (see ``homeease.ml.audio.pipeline``): the speech VAD gates the
    VOICE branch only; the SOUND branch is gated by an energy-onset trigger plus a slow periodic
    window, because a speech VAD would discard impacts and breaking glass."""

    # speech VAD (voice branch)
    vad_window_s: float = Field(1.5, ge=0.5, le=10.0)  # context the VAD sees on each (re)run
    vad_hop_s: float = Field(0.25, ge=0.05, le=2.0)  # how often it is re-run on the live stream
    speech_min_s: float = Field(0.25, ge=0.05, le=5.0)  # shorter bursts (clicks) are not speech segments
    speech_end_silence_s: float = Field(0.6, ge=0.1, le=5.0)
    speech_max_s: float = Field(12.0, ge=2.0, le=60.0)  # long speech is cut into segments of at most this length
    # sound trigger (sound branch)
    onset_db: float = Field(12.0, ge=3.0, le=40.0)  # frame level above the tracked noise floor
    hold_s: float = Field(0.3, ge=0.0, le=5.0)  # gaps shorter than this inside one sound do not end its activity span
    min_level_dbfs: float = Field(-60.0, ge=-100.0, le=-20.0)  # nothing below this absolute level triggers
    floor_rise_db_per_s: float = Field(1.0, gt=0.0, le=20.0)  # the floor follows quiet instantly, loud only slowly
    periodic_s: float = Field(60.0, ge=5.0, le=3600.0)  # slow window that catches sounds the onset trigger has absorbed
    gap_tolerance_ms: float = Field(200.0, ge=20.0, le=5000.0)  # timestamp jump that restarts the timeline


class SoundMonitorCfg(_M):
    """Daily sound monitoring (proposal 1.5.2). Off by default; stores labels and statistics only."""

    enabled: bool = False
    model_id: str = "sound_events"  # model package models/audio_sound/<model_id>/<version>/
    window_s: float = Field(2.0, ge=0.5, le=10.0)  # a model package may declare its own window length
    hop_s: float = Field(0.5, ge=0.1, le=5.0)
    min_conf: float | None = Field(None, ge=0.0, le=1.0)  # None: threshold chosen on dev data and stored in the package, else 0.5
    confirm_windows: int = Field(2, ge=1, le=10)  # consecutive windows with the same label before an event exists
    cooldown_s: float = Field(20.0, ge=0.0, le=3600.0)  # same-category detections closer than this are merged into one event
    # sha256 of the model weights file, pinned HERE (outside models/): a scikit-learn package is a pickle, and the checksum inside the
    # package folder only detects corruption, not replacement. Set it for a deployment (value: "sha256" in the package's meta.json).
    model_sha256: str | None = Field(None, pattern=r"^[0-9a-fA-F]{64}$")
    # Demonstrations only: in replay/synthetic mode with demo mode on, listen to held-out ESC-50 clips from this
    # folder (fold 5, never trained on) instead of a microphone. `homeease serve --demo` sets data/raw/esc50.
    replay_clips: str | None = None

    @model_validator(mode="after")
    def _hop(self) -> "SoundMonitorCfg":
        if self.hop_s > self.window_s:
            raise ValueError("audio.sound_monitor.hop_s must not exceed window_s (windows would leave gaps)")
        return self


class VoiceprintCfg(_M):
    """Cancel-eligibility EVIDENCE from a voiceprint match (``homeease.ml.audio.cancel_gate``). Evidence only: it never changes an
    event (``audio.auto_cancel_emergency`` stays impossible). Without an enrollment or a threshold the evidence is simply False."""

    speaker_model: str = "voxceleb_resnet34_LM.onnx"  # file under models/audio/wespeaker/
    # cosine threshold on the speaker score, calibrated on DEV speakers (``calibrate_threshold``; experiment A6). None = not calibrated: never eligible.
    threshold: float | None = Field(None, ge=-1.0, le=1.0)
    min_speech_s: float = Field(1.0, ge=0.3, le=10.0)  # seconds of VAD speech needed before voices are compared
    min_asr_logprob: float = Field(-1.0, ge=-5.0, le=0.0)  # faster-whisper avg_logprob a REASSURANCE reply needs


class HelpListenerCfg(_M):
    """Continuous listening for a spontaneous spoken call for help (``homeease.ml.audio.help_listener``). Off by default, and it only ever
    runs inside an enabled audio layer with the subject's audio consent. Speech segments are analysed in memory and dropped; no audio and
    (``privacy.persist_voice_transcript``) no transcript is stored."""

    enabled: bool = False
    cooldown_s: float = Field(30.0, ge=0.0, le=3600.0)  # after one call for help, further calls inside this time are ignored (a TV or one long cry must not raise a second event)
    min_utterance_s: float = Field(0.3, ge=0.1, le=5.0)  # shorter speech segments are noise, not a call
    max_utterance_s: float = Field(6.0, ge=0.5, le=30.0)  # longer speech is a conversation or a broadcast, not a call for help
    min_asr_logprob: float = Field(-1.0, ge=-5.0, le=0.0)  # faster-whisper avg_logprob a recognised call needs

    @model_validator(mode="after")
    def _utterance_bounds(self) -> "HelpListenerCfg":
        if self.min_utterance_s >= self.max_utterance_s:
            raise ValueError("audio.help_listener.min_utterance_s must be shorter than max_utterance_s")
        return self


class AudioCfg(_M):
    enabled: bool = False
    auto_cancel_emergency: bool = False
    inquiry_response_window_s: float = 10.0
    # After a voice inquiry that was not cancelled (silence, unclear or unrelated reply, an untrusted "I'm fine") a PENDING event is raised to
    # EMERGENCY: the voice check becomes the second channel. A valid cancel only stops this escalation; it never closes the event.
    escalate_on_no_cancel: bool = True
    # system = the platform's native engine (macOS say, Windows SAPI); system_say is the historic alias of system
    tts: Literal["system", "system_say", "browser", "none"] = "browser"
    asr_temperature: float | None = Field(None, ge=0.0, le=1.0)  # None: faster-whisper default (random fallback sampling); 0.0: deterministic decoding
    asr_model: str = "tiny"
    # reply-intent method of the voice inquiry: rule_v2 (script normalisation, negation scope; chosen in experiment A3b) or the original rule
    intent_method: Literal["rule_v2", "rule"] = "rule_v2"
    vad: Literal["energy", "silero", "webrtc"] = "silero"
    languages: list[str] = Field(default_factory=lambda: ["zh", "en"])  # the order the voice prompt is spoken and shown in
    buffer_seconds: float = 15.0
    gate: ActivityGateCfg = Field(default_factory=ActivityGateCfg)
    sound_monitor: SoundMonitorCfg = Field(default_factory=SoundMonitorCfg)
    voiceprint: VoiceprintCfg = Field(default_factory=VoiceprintCfg)
    help_listener: HelpListenerCfg = Field(default_factory=HelpListenerCfg)

    @field_validator("auto_cancel_emergency")
    @classmethod
    def _never_auto_cancel(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "audio.auto_cancel_emergency cannot be enabled in the MVP: a spoken reply "
                "is supporting evidence only (proposal 6.3, revised slide 1)"
            )
        return v

    @model_validator(mode="after")
    def _listener_can_hear_something(self) -> "AudioCfg":
        # the listener only judges speech segments that are long enough for min_utterance_s and short enough for max_utterance_s AND for the
        # speech gate's own cut (speech_max_s): when no length can pass, it would run, report "ok" and never hear a call
        lc = self.help_listener
        if lc.enabled and not lc.min_utterance_s < min(lc.max_utterance_s, self.gate.speech_max_s):
            raise ValueError(
                f"audio.help_listener.min_utterance_s ({lc.min_utterance_s}) must be shorter than both max_utterance_s and audio.gate.speech_max_s "
                f"({min(lc.max_utterance_s, self.gate.speech_max_s)}): no speech segment could be judged"
            )
        return self

    @model_validator(mode="after")
    def _buffer_covers_windows(self) -> "AudioCfg":
        g = self.gate
        # a speech segment is cut from the buffer only after the VAD has seen its end (end silence + look-ahead + one hop) and gets a little context
        need = max(self.sound_monitor.window_s + self.sound_monitor.hop_s, g.speech_max_s + g.speech_end_silence_s + g.vad_hop_s + 1.0)
        if self.buffer_seconds < need:
            raise ValueError(f"audio.buffer_seconds ({self.buffer_seconds}) must cover the sound window plus hop and the longest speech segment with its detection latency ({need:.1f} s)")
        return self


class PrivacyCfg(_M):
    persist_raw_video: bool = False
    persist_raw_audio: bool = False
    # the text of a spoken reply is personal content: by default only intent, reason and cues are stored (never in the audit transitions)
    persist_voice_transcript: bool = False
    research_recording_dir: str = "data/captures"
    retention_days_events: int = 365
    retention_days_health: int = 90
    retention_days_research: int = 30

    @field_validator("persist_raw_audio")
    @classmethod
    def _never_persist_audio(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "privacy.persist_raw_audio cannot be enabled in the MVP: audio is analysed in memory and only "
                "labels, timestamps and summary statistics are stored (proposal 1.5.2, no raw audio storage)"
            )
        return v


class AppConfig(_M):
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    server: ServerCfg = Field(default_factory=ServerCfg)
    debug: DebugCfg = Field(default_factory=DebugCfg)
    acquisition: AcquisitionCfg = Field(default_factory=AcquisitionCfg)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    fall: FallCfg = Field(default_factory=FallCfg)
    exercise: ExerciseCfg = Field(default_factory=ExerciseCfg)
    escalation: EscalationCfg = Field(default_factory=EscalationCfg)
    notifications: NotificationsCfg = Field(default_factory=NotificationsCfg)
    audio: AudioCfg = Field(default_factory=AudioCfg)
    privacy: PrivacyCfg = Field(default_factory=PrivacyCfg)
    demo: DemoCfg = Field(default_factory=DemoCfg)

    @model_validator(mode="after")
    def _demo_stays_local(self) -> "AppConfig":
        if self.demo.enabled and not self.demo.loopback_only and self.runtime.mode == "live":
            raise ValueError("demo mode without login must stay loopback-only when the node runs in live mode")
        return self

    @model_validator(mode="after")
    def _live_fusion_policy(self) -> "AppConfig":
        if self.runtime.mode == "live" and self.fall.fusion.backend == "late_rule_basic":
            raise ValueError(
                "fall.fusion.backend=late_rule_basic is an ablation baseline without the quality/time/target "
                "emergency gate; live operation must use late_quality_gated, late_trained or feature_fusion"
            )
        return self

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides(prefix: str = "HOMEEASE__") -> dict[str, Any]:
    over: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = [p.lower() for p in key[len(prefix):].split("__") if p]
        if not parts:
            continue
        try:
            value: Any = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cur = over
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return over


def load_config(path: str | os.PathLike[str] | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    data: dict[str, Any] = {}
    if path is None:
        default = project_root() / "config" / "default.yaml"
        path = default if default.is_file() else None
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = project_root() / p
        with open(p, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {p} must contain a mapping")
        data = loaded
    data = _deep_merge(data, _env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig.model_validate(data)
