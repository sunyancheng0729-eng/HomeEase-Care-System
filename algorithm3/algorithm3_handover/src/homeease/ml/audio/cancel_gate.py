"""Voiceprint matching and cancel-eligibility EVIDENCE (proposal Figure 3-3, branch A, second half).

The figure draws "person responds and voiceprint matches -> auto-cancel false alarm". The approved product policy is
the proposal TEXT (sections 2.4, 3.3, 6.3, Table 9-1): a spoken reply is supporting evidence only. So this module only
produces EVIDENCE: ``cancel_eligible`` says "had automatic cancellation been allowed, every condition for it would hold".
Nothing here, nor any caller, changes the level or the lifecycle status of an event (``audio.auto_cancel_emergency`` is
fixed to false by a config validator, ``EventService.voice_response`` never reads ``cancel_eligible``, experiment A5/A6 and
tests/test_cancel_gate.py check both). The false-cancel rate of this evidence is what experiment A6 measures, so that the
auto-cancel box of the figure can be judged on numbers if the policy is ever revisited (documented as a future extension).

``cancel_eligible`` is True only if ALL of these hold; otherwise it is False with explicit reason codes:

1. a voiceprint enrollment exists and was made with the same embedding model            (no_enrollment, enrollment_model_mismatch)
2. the speaker model can be loaded and a decision threshold is configured              (model_missing, no_threshold)
3. there is enough speech to compare voices: at least ``min_speech_s`` seconds of VAD speech   (too_short)
4. the reply intent is REASSURANCE                                                          (intent_not_reassurance)
5. the ASR confidence (faster-whisper avg_logprob) is at least ``min_asr_logprob``          (low_asr_confidence)
6. the speaker score (cosine to the enrolled voiceprint) is at least the calibrated threshold (speaker_mismatch)
   a failing embedding call is ``embedding_error``. Anything that could not be evaluated counts as NOT eligible.

Privacy. The enrollment keeps one L2-normalised embedding and a few counters, never audio. An embedding is still biometric
data (it identifies a speaker, and it is not a proof of anonymity), so storing it is a privacy decision of the household:
this module offers an in-memory object and a JSON round trip (``to_json`` / ``from_json``) and nothing that writes a file.
Audio passed to ``evaluate`` is only read; the speech crop that goes into the embedder is a private copy that is zeroed
before the call returns.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from homeease.core.util import get_logger
from homeease.ml.audio.features import SR
from homeease.ml.audio.speaker import tar_at_far

log = get_logger(__name__)

ENROLLMENT_SCHEMA = 1
MIN_ENROLL_UTTERANCES = 3
MIN_ENROLL_UTT_S = 1.5  # seconds of VAD speech per enrollment utterance (a 0.5 s "okay" is a poor voice sample)
MIN_ENROLL_CONSISTENCY = 0.25  # unvalidated sanity guard: an utterance far from the mean voice is probably another person
DEFAULT_MIN_SPEECH_S = 1.0  # below about one second of speech a speaker embedding is unreliable (A6 reports the dev EER per length)
DEFAULT_MIN_ASR_LOGPROB = -1.0  # faster-whisper avg_logprob; stricter than the -1.2 "unclear speech" cut of the intent rule

# reason codes (stable strings: they are stored in the voice-inquiry record and read by reports)
NO_ENROLLMENT = "no_enrollment"
ENROLLMENT_MODEL_MISMATCH = "enrollment_model_mismatch"
MODEL_MISSING = "model_missing"
NO_THRESHOLD = "no_threshold"
TOO_SHORT = "too_short"
INTENT_NOT_REASSURANCE = "intent_not_reassurance"
INTENT_UNAVAILABLE = "intent_unavailable"
LOW_ASR_CONFIDENCE = "low_asr_confidence"
SPEAKER_MISMATCH = "speaker_mismatch"
EMBEDDING_ERROR = "embedding_error"
REASON_CODES = (NO_ENROLLMENT, ENROLLMENT_MODEL_MISMATCH, MODEL_MISSING, NO_THRESHOLD, TOO_SHORT, INTENT_NOT_REASSURANCE, INTENT_UNAVAILABLE, LOW_ASR_CONFIDENCE, SPEAKER_MISMATCH, EMBEDDING_ERROR)
# codes the inquiry service itself sets when the gate was never reached
NO_SPEECH = "no_speech"
AUDIO_UNAVAILABLE = "audio_unavailable"
ASR_FAILED = "asr_failed"
GATE_ERROR = "gate_error"
POLICY = "evidence_only"


class EnrollmentError(ValueError):
    """The enrollment utterances are unusable (too few, too short, inconsistent, degenerate)."""


class Embedder(Protocol):
    model_id: str

    def embed(self, audio16k: np.ndarray) -> np.ndarray: ...


# ----------------------------------------------------------------------------- speech crop
def speech_crop(audio: np.ndarray, vad: Any = None, context_s: float = 0.1, sr: int = SR) -> tuple[np.ndarray, float]:
    """(private copy of the VAD-speech samples with a little context, seconds of speech). Silence and non-speech are dropped,
    so a 10 s response window with 2 s of speech is embedded as those 2 s. The caller must zero the copy after use."""
    if vad is None:
        from homeease.ml.audio.vad import EnergyVAD

        vad = EnergyVAD()
    x = np.asarray(audio, dtype=np.float32)
    fr = np.asarray(vad.frames(x, sr)).astype(bool)
    hop = sr // 100
    if not fr.any():
        return np.zeros(0, dtype=np.float32), 0.0
    ctx = int(round(context_s * 100))
    keep = np.convolve(fr.astype(np.int8), np.ones(2 * ctx + 1, dtype=np.int8), mode="same") > 0 if ctx > 0 else fr
    mask = np.repeat(keep, hop)
    mask = np.concatenate([mask, np.zeros(max(0, len(x) - len(mask)), dtype=bool)])[: len(x)]
    return x[mask].copy(), float(fr.sum()) / 100.0


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-9:
        raise EnrollmentError("degenerate embedding (zero or non-finite norm)")
    return (v / n).astype(np.float32)


# ----------------------------------------------------------------------------- enrollment
@dataclass(frozen=True, eq=False)
class VoiceprintEnrollment:
    """The enrolled voice: the L2-normalised mean of the embeddings of >= 3 enrollment utterances. No audio is kept."""

    embedding: np.ndarray = field(repr=False)  # float32, unit norm
    model_id: str
    n_utterances: int
    speech_s: float  # total VAD speech used
    min_consistency: float | None  # lowest cosine between one utterance embedding and the mean (1 = identical); None when built from a bare mean
    created_at: str | None = None

    def __post_init__(self) -> None:
        e = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        if e.size == 0 or not np.all(np.isfinite(e)) or abs(float(np.linalg.norm(e)) - 1.0) > 1e-3:
            raise EnrollmentError("enrollment embedding must be finite and L2-normalised")
        if self.n_utterances < MIN_ENROLL_UTTERANCES:
            raise EnrollmentError(f"an enrollment needs at least {MIN_ENROLL_UTTERANCES} utterances, got {self.n_utterances}")
        object.__setattr__(self, "embedding", e)

    @classmethod
    def from_embeddings(cls, embeddings: Sequence[np.ndarray], model_id: str, speech_s: float = 0.0, created_at: str | None = None,
                        min_consistency: float | None = MIN_ENROLL_CONSISTENCY) -> "VoiceprintEnrollment":
        """Mean of the (individually normalised) embeddings, normalised again. `min_consistency` refuses an enrollment whose
        utterances do not look like one voice (None disables the guard)."""
        if len(embeddings) < MIN_ENROLL_UTTERANCES:
            raise EnrollmentError(f"an enrollment needs at least {MIN_ENROLL_UTTERANCES} utterances, got {len(embeddings)}")
        es = [_l2(e) for e in embeddings]
        if len({e.shape for e in es}) != 1:
            raise EnrollmentError("embeddings have different dimensions")
        mean = _l2(np.mean(np.stack(es), axis=0))
        sims = [float(np.dot(e, mean)) for e in es]
        if min_consistency is not None and min(sims) < min_consistency:
            raise EnrollmentError(f"enrollment utterances are inconsistent (lowest cosine to the mean voice {min(sims):.2f} < {min_consistency}): was more than one person speaking?")
        return cls(mean, model_id, len(es), float(speech_s), min(sims), created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @classmethod
    def from_audio(cls, embedder: Embedder, utterances: Sequence[np.ndarray], vad: Any = None, min_utt_s: float = MIN_ENROLL_UTT_S,
                   min_consistency: float | None = MIN_ENROLL_CONSISTENCY) -> "VoiceprintEnrollment":
        """Embed each utterance (VAD speech only; the crop is zeroed after use) and build the enrollment. The audio itself is not kept."""
        embs: list[np.ndarray] = []
        total = 0.0
        for k, u in enumerate(utterances):
            crop, sp = speech_crop(u, vad)
            try:
                if sp < min_utt_s:
                    raise EnrollmentError(f"enrollment utterance {k + 1} has {sp:.1f} s of speech, at least {min_utt_s:.1f} s are needed")
                embs.append(np.asarray(embedder.embed(crop), dtype=np.float32))
            finally:
                crop[:] = 0.0
            total += sp
        return cls.from_embeddings(embs, embedder.model_id, total, min_consistency=min_consistency)

    def score(self, embedding: np.ndarray) -> float:
        """Cosine similarity between the enrolled voice and one embedding."""
        e = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if e.shape != self.embedding.shape:
            raise ValueError(f"embedding dimension {e.shape[0]} != enrolled {self.embedding.shape[0]}")
        return float(np.dot(self.embedding.astype(np.float64), e) / (np.linalg.norm(e) + 1e-12))

    # ---- JSON round trip only (persisting it is a privacy decision of the household; nothing here writes a file)
    def to_dict(self) -> dict[str, Any]:
        return {"schema": ENROLLMENT_SCHEMA, "kind": "voiceprint_enrollment", "model_id": self.model_id, "n_utterances": self.n_utterances, "speech_s": round(self.speech_s, 3),
                "min_consistency": None if self.min_consistency is None else round(self.min_consistency, 4), "created_at": self.created_at,
                "dim": int(self.embedding.size), "embedding": [float(v) for v in self.embedding]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "VoiceprintEnrollment":
        if not isinstance(d, Mapping):
            raise EnrollmentError("an enrollment is a JSON object")
        if d.get("kind") != "voiceprint_enrollment" or d.get("schema") != ENROLLMENT_SCHEMA:
            raise EnrollmentError("not a voiceprint enrollment of schema 1")
        emb = np.asarray(d.get("embedding"), dtype=np.float32)
        if emb.ndim != 1 or emb.size != int(d.get("dim", -1)):
            raise EnrollmentError("embedding length does not match 'dim'")
        mc = d.get("min_consistency")
        return cls(emb, str(d["model_id"]), int(d["n_utterances"]), float(d.get("speech_s", 0.0)), None if mc is None else float(mc), d.get("created_at"))

    @classmethod
    def from_json(cls, text: str) -> "VoiceprintEnrollment":
        try:
            return cls.from_dict(json.loads(text))
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, EnrollmentError):
                raise
            raise EnrollmentError(f"unreadable enrollment: {e}") from e


# ----------------------------------------------------------------------------- calibration
@dataclass(frozen=True)
class Calibration:
    """Result of `calibrate_threshold`. `threshold` is None when it cannot be estimated (then the gate is never eligible)."""

    threshold: float | None
    target_far: float
    achieved_far: float | None  # impostor share ABOVE the threshold on the dev trials
    tar: float | None  # genuine share above the threshold on the dev trials
    n_genuine: int
    n_impostor: int
    reliable: bool  # False when there are fewer than 10 / far impostor trials: the rate cannot be estimated, the number is only indicative
    note: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "target_far": self.target_far, "achieved_far": self.achieved_far, "tar": self.tar, "n_genuine": self.n_genuine,
                "n_impostor": self.n_impostor, "reliable": self.reliable, "note": self.note}


def calibrate_threshold(dev_scores: Sequence[float] | np.ndarray, dev_labels: Sequence[int] | np.ndarray, far: float = 0.001) -> Calibration:
    """Decision threshold on DEV trials (labels: 1 genuine, 0 impostor): the lowest score cut that lets at most `far` of the impostor
    trials through (strictly above the cut). Reuses `speaker.tar_at_far`. Never call this with test-fold scores.
    The guarantee is on the dev trials; with fewer than 10 / far impostor trials it is flagged `reliable=False`."""
    if not 0.0 < far < 1.0:
        raise ValueError("far must be in (0, 1)")
    s = np.asarray(dev_scores, dtype=np.float64).reshape(-1)
    y = np.asarray(dev_labels).reshape(-1).astype(int)
    if s.shape != y.shape:
        raise ValueError("scores and labels differ in length")
    if not np.all(np.isfinite(s)):
        raise ValueError("non-finite scores")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0 (impostor) or 1 (genuine)")
    r = tar_at_far(s, y, far)
    if r.get("tar") is None:
        return Calibration(None, far, None, None, int(r["genuine_trials"]), int(r["impostor_trials"]), False, "not estimable: needs both genuine and impostor trials")
    return Calibration(float(r["threshold"]), far, float(r["achieved_far"]), float(r["tar"]), int(r["genuine_trials"]), int(r["impostor_trials"]), bool(r["reliable"]), r.get("note"))


# ----------------------------------------------------------------------------- the decision
@dataclass(frozen=True)
class CancelDecision:
    cancel_eligible: bool
    reasons: tuple[str, ...]  # empty when eligible
    speaker_score: float | None
    speaker_verified: bool | None  # score >= threshold; None when no score or no threshold
    threshold: float | None
    speech_s: float
    min_speech_s: float
    min_asr_logprob: float
    asr_conf: float | None
    intent: str | None

    def to_record(self) -> dict[str, Any]:
        """Fields merged into the voice-inquiry record. JSON-safe, no audio, no embedding."""
        return {
            "cancel_eligible": self.cancel_eligible, "cancel_reasons": list(self.reasons), "cancel_policy": POLICY,
            "speaker_verified": self.speaker_verified, "speaker_score": None if self.speaker_score is None else round(self.speaker_score, 4),
            "voiceprint_threshold": None if self.threshold is None else round(self.threshold, 4), "speech_s": round(self.speech_s, 2),
        }


def decide(*, enrolled: bool, enrollment_model_ok: bool = True, model_ok: bool = True, embedding_ok: bool = True, threshold: float | None, speech_s: float, min_speech_s: float,
           intent: str | None, asr_conf: float | None, min_asr_logprob: float, score: float | None) -> CancelDecision:
    """The pure rule. `evaluate` computes the inputs; experiment A6 calls this with cached scores so that every operating point is judged by
    the very same logic as the product. Every unmet or unevaluable condition adds a reason; eligible = no reason."""
    reasons: list[str] = []
    if not enrolled:
        reasons.append(NO_ENROLLMENT)
    elif not enrollment_model_ok:
        reasons.append(ENROLLMENT_MODEL_MISMATCH)
    if not model_ok:
        reasons.append(MODEL_MISSING)
    if threshold is None or not math.isfinite(threshold):
        reasons.append(NO_THRESHOLD)
        threshold = None
    if not speech_s >= min_speech_s:  # also catches NaN
        reasons.append(TOO_SHORT)
    if intent is None:
        reasons.append(INTENT_UNAVAILABLE)
    elif intent != "REASSURANCE":
        reasons.append(INTENT_NOT_REASSURANCE)
    elif asr_conf is None or not asr_conf >= min_asr_logprob:
        reasons.append(LOW_ASR_CONFIDENCE)
    verified: bool | None = None
    if not embedding_ok or (score is not None and not math.isfinite(score)):
        reasons.append(EMBEDDING_ERROR)
        score = None
    if score is not None and threshold is not None:
        verified = bool(score >= threshold)
        if not verified:
            reasons.append(SPEAKER_MISMATCH)
    if not reasons and verified is not True:  # defensive: eligibility always needs a score at or above the threshold
        reasons.append(EMBEDDING_ERROR)
    return CancelDecision(not reasons, tuple(dict.fromkeys(reasons)), score, verified, threshold, float(speech_s), float(min_speech_s), float(min_asr_logprob), asr_conf, intent)


class CancelGate:
    """Cancel-eligibility evidence for one voice reply. Evidence only: no method here touches an event."""

    def __init__(self, threshold: float | None = None, min_speech_s: float = DEFAULT_MIN_SPEECH_S, min_asr_logprob: float = DEFAULT_MIN_ASR_LOGPROB,
                 embedder: Embedder | None = None, speaker_model: str = "voxceleb_resnet34_LM.onnx", vad: Any = None) -> None:
        self.threshold = threshold
        self.min_speech_s = float(min_speech_s)
        self.min_asr_logprob = float(min_asr_logprob)
        self._embedder = embedder
        self._speaker_model = speaker_model
        self._vad = vad
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @classmethod
    def from_config(cls, cfg: Any, embedder: Embedder | None = None) -> "CancelGate":
        """`cfg` = AudioCfg (reads cfg.voiceprint)."""
        v = cfg.voiceprint
        return cls(v.threshold, v.min_speech_s, v.min_asr_logprob, embedder, v.speaker_model)

    # ---- model
    def embedder(self) -> Embedder | None:
        """The speaker model, loaded on first use; None (and the reason in `load_error`) when it cannot be loaded. Not retried per call."""
        with self._lock:
            if self._embedder is None and self._load_error is None:
                try:
                    from homeease.ml.audio.speaker import WeSpeakerEmbedder

                    self._embedder = WeSpeakerEmbedder(self._speaker_model)
                except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as e:  # missing file, missing onnxruntime, broken model
                    self._load_error = f"{type(e).__name__}: {str(e)[:160]}"
                    log.warning("speaker model unavailable, cancel evidence stays False (model_missing): %s", self._load_error)
            return self._embedder

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ---- evaluation
    def evaluate(self, audio: np.ndarray, intent_result: Mapping[str, Any] | None, enrollment: VoiceprintEnrollment | None, asr_conf: float | None = None) -> CancelDecision:
        """`audio` = the reply recording (16 kHz mono float; it is only read), `intent_result` = {"intent": ..., "reason": ...} of the intent model,
        `asr_conf` = faster-whisper avg_logprob of the reply. Never raises for model or audio problems: they become reason codes."""
        crop, speech_s = speech_crop(audio, self._vad)
        try:
            intent = None if intent_result is None else str(intent_result.get("intent"))
            enrolled = enrollment is not None
            emb = self.embedder() if enrolled else None
            model_ok = emb is not None
            enr_ok = bool(enrolled and emb is not None and enrollment.model_id == emb.model_id)  # type: ignore[union-attr]
            score: float | None = None
            emb_ok = True
            if enrolled and model_ok and enr_ok and speech_s >= self.min_speech_s:
                try:
                    e = np.asarray(emb.embed(crop), dtype=np.float32)  # type: ignore[union-attr]
                    score = enrollment.score(e)  # type: ignore[union-attr]
                except Exception as ex:  # a model that fails on this audio must never make a reply eligible
                    log.warning("speaker embedding failed: %s", str(ex)[:160])
                    emb_ok = False
            return decide(enrolled=enrolled, enrollment_model_ok=enr_ok or not (enrolled and model_ok), model_ok=model_ok or not enrolled, embedding_ok=emb_ok, threshold=self.threshold,
                          speech_s=speech_s, min_speech_s=self.min_speech_s, intent=intent, asr_conf=asr_conf, min_asr_logprob=self.min_asr_logprob, score=score)
        finally:
            crop[:] = 0.0  # the speech crop is a private copy: gone before we return
