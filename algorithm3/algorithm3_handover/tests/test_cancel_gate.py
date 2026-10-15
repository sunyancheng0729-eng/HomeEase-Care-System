"""Voiceprint enrollment, threshold calibration and the cancel-eligibility gate (ml/audio/cancel_gate.py). Offline: stub embedder, no model, no audio files.

The gate produces EVIDENCE only. These tests pin the reason codes, the "everything else is False" direction of every failure and the privacy rules
(no audio kept, enrollment is an embedding only); tests/test_cancel_policy.py checks that no event ever changes because of it."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from homeease.ml.audio import cancel_gate as cg
from homeease.ml.audio.cancel_gate import CancelGate, EnrollmentError, VoiceprintEnrollment, calibrate_threshold, decide
from homeease.ml.audio.speaker import tar_at_far

SR = 16000
DIM = 16


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def near(base: np.ndarray, seed: int, noise: float = 0.05) -> np.ndarray:
    v = base + np.random.default_rng(seed).normal(size=base.shape) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


class StubEmbedder:
    """Returns a prepared embedding and records exactly what it was given (to check the crop and its zeroing)."""

    model_id = "stub_model"

    def __init__(self, emb: np.ndarray | None = None, fail: bool = False) -> None:
        self.emb, self.fail = emb if emb is not None else unit(1), fail
        self.seen: list[np.ndarray] = []

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        self.seen.append(audio16k)  # a reference, on purpose: the gate must have zeroed it by the time evaluate returns
        if self.fail:
            raise RuntimeError("onnx failed")
        return self.emb


class StubVAD:
    """10 ms frames: speech wherever |x| > 0.01."""

    def frames(self, x: np.ndarray, sr: int = SR) -> np.ndarray:
        n = len(x) // 160
        return (np.abs(x[: n * 160]).reshape(n, 160).max(axis=1) > 0.01).astype(np.int8)


def reply(speech_s: float, total_s: float = 10.0, start_s: float = 2.0) -> np.ndarray:
    """A response window: silence, then `speech_s` seconds of a loud tone."""
    x = np.zeros(int(total_s * SR), dtype=np.float32)
    a = int(start_s * SR)
    x[a : a + int(speech_s * SR)] = 0.3 * np.sin(np.arange(int(speech_s * SR)) / 7).astype(np.float32)
    return x


def make_gate(emb: np.ndarray | None = None, threshold: float | None = 0.6, fail: bool = False, **kw: float) -> tuple[CancelGate, StubEmbedder]:
    e = StubEmbedder(emb, fail)
    return CancelGate(threshold, embedder=e, vad=StubVAD(), **kw), e


def enrollment(base: np.ndarray | None = None, model_id: str = "stub_model") -> VoiceprintEnrollment:
    base = base if base is not None else unit(1)
    return VoiceprintEnrollment.from_embeddings([near(base, s) for s in range(3)], model_id, 9.0)


REASSURE = {"intent": "REASSURANCE", "reason": "reassurance_cue"}


# ------------------------------------------------------------------------------------------------ enrollment
def test_enrollment_is_the_normalised_mean_of_normalised_embeddings() -> None:
    a, b, c = unit(1), unit(2), unit(3)
    e = VoiceprintEnrollment.from_embeddings([a * 3.0, b * 0.5, c], "m", 6.0, min_consistency=None)  # scale of the inputs must not matter
    want = (a + b + c) / np.linalg.norm(a + b + c)
    assert np.allclose(e.embedding, want, atol=1e-6) and abs(float(np.linalg.norm(e.embedding)) - 1.0) < 1e-6
    assert e.n_utterances == 3 and e.model_id == "m" and e.embedding.dtype == np.float32
    assert e.min_consistency is not None and -1.0 <= e.min_consistency <= 1.0


@pytest.mark.parametrize("n", [0, 1, 2])
def test_enrollment_needs_at_least_three_utterances(n: int) -> None:
    with pytest.raises(EnrollmentError, match="at least 3"):
        VoiceprintEnrollment.from_embeddings([unit(i) for i in range(n)], "m")
    with pytest.raises(EnrollmentError):  # also when built directly
        VoiceprintEnrollment(unit(1), "m", n_utterances=n, speech_s=1.0, min_consistency=None)


def test_enrollment_refuses_degenerate_and_inconsistent_embeddings() -> None:
    with pytest.raises(EnrollmentError, match="degenerate"):
        VoiceprintEnrollment.from_embeddings([unit(1), unit(2), np.zeros(DIM)], "m")
    with pytest.raises(EnrollmentError, match="degenerate"):
        VoiceprintEnrollment.from_embeddings([unit(1), unit(2), np.full(DIM, np.nan)], "m")
    with pytest.raises(EnrollmentError, match="dimensions"):
        VoiceprintEnrollment.from_embeddings([unit(1), unit(2), unit(3, DIM + 1)], "m")
    base, other = unit(1), -unit(1)  # two people: one utterance points the other way
    with pytest.raises(EnrollmentError, match="inconsistent"):
        VoiceprintEnrollment.from_embeddings([base, near(base, 5), other, other], "m")
    VoiceprintEnrollment.from_embeddings([base, near(base, 5), other, other], "m", min_consistency=None)  # the guard can be switched off explicitly


def test_enrollment_json_round_trip_keeps_only_an_embedding() -> None:
    e = enrollment()
    text = e.to_json()
    d = json.loads(text)
    assert set(d) == {"schema", "kind", "model_id", "n_utterances", "speech_s", "min_consistency", "created_at", "dim", "embedding"}
    assert not any("audio" in k or "wave" in k or "sample" in k for k in d)  # no audio of any kind
    back = VoiceprintEnrollment.from_json(text)
    assert np.allclose(back.embedding, e.embedding, atol=1e-6) and back.model_id == e.model_id and back.n_utterances == 3 and back.created_at == e.created_at
    assert back.score(e.embedding) == pytest.approx(1.0, abs=1e-5)
    assert "embedding" not in repr(e) and str(e.embedding[0])[:6] not in repr(e)  # the vector never lands in a log line through repr


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(kind="something"), lambda d: d.update(schema=2), lambda d: d.update(dim=DIM + 1), lambda d: d.update(embedding=[0.0] * DIM),
    lambda d: d.update(n_utterances=2), lambda d: d.update(embedding="abc"), lambda d: d.pop("model_id"),
])
def test_enrollment_from_json_rejects_damaged_input(mutate) -> None:  # type: ignore[no-untyped-def]
    d = enrollment().to_dict()
    mutate(d)
    with pytest.raises(EnrollmentError):
        VoiceprintEnrollment.from_json(json.dumps(d))


def test_enrollment_from_json_rejects_garbage() -> None:
    for text in ("", "not json", "[]", "null", '{"kind": "voiceprint_enrollment"}'):
        with pytest.raises(EnrollmentError):
            VoiceprintEnrollment.from_json(text)


def test_enrollment_from_audio_embeds_speech_only_zeroes_the_crop_and_keeps_no_audio() -> None:
    emb = StubEmbedder(unit(4))
    utts = [reply(2.0, total_s=6.0, start_s=1.0) for _ in range(3)]
    before = [u.copy() for u in utts]
    e = VoiceprintEnrollment.from_audio(emb, utts, vad=StubVAD(), min_consistency=None)
    assert e.n_utterances == 3 and e.speech_s == pytest.approx(6.0, abs=0.1) and e.model_id == "stub_model"
    assert all(len(s) < 3 * SR for s in emb.seen)  # the embedder saw the ~2.2 s speech crop, not the 6 s window
    assert all(not np.any(s) for s in emb.seen)  # and the crop was zeroed afterwards
    assert all(np.array_equal(u, b) for u, b in zip(utts, before))  # the caller's audio is untouched
    assert not any(isinstance(v, np.ndarray) and v.size > 100 for v in vars(e).values() if v is not e.embedding)


def test_enrollment_from_audio_refuses_a_short_utterance() -> None:
    emb = StubEmbedder()
    utts = [reply(2.0), reply(0.5), reply(2.0)]
    with pytest.raises(EnrollmentError, match="utterance 2"):
        VoiceprintEnrollment.from_audio(emb, utts, vad=StubVAD())


# ------------------------------------------------------------------------------------------------ calibration
def trials(seed: int, n_gen: int = 200, n_imp: int = 4000, gen_mu: float = 0.7, imp_mu: float = 0.2, sd: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    s = np.concatenate([r.normal(gen_mu, sd, n_gen), r.normal(imp_mu, sd, n_imp)])
    y = np.concatenate([np.ones(n_gen, dtype=int), np.zeros(n_imp, dtype=int)])
    return s, y


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("far", [0.0005, 0.001, 0.01, 0.05, 0.2])
def test_calibrated_threshold_never_exceeds_the_target_far_on_dev(seed: int, far: float) -> None:
    s, y = trials(seed)
    c = calibrate_threshold(s, y, far)
    imp = s[y == 0]
    assert c.threshold is not None and c.n_impostor == len(imp) and c.n_genuine == int(y.sum())
    real_far = float(np.mean(imp > c.threshold))
    assert real_far <= far + 1e-12 and c.achieved_far == pytest.approx(real_far)
    assert c.tar == pytest.approx(float(np.mean(s[y == 1] > c.threshold)))
    assert c.reliable == (len(imp) >= 10 / far)
    # tight: lowering the threshold to the next impostor score would break the target (unless ties or far below one trial)
    k = int(np.floor(far * len(imp)))
    assert k == int(np.sum(imp > c.threshold)) or len(set(imp)) < len(imp)


@pytest.mark.parametrize("seed", range(6))
def test_threshold_and_tar_are_monotone_in_far(seed: int) -> None:
    s, y = trials(seed)
    cs = [calibrate_threshold(s, y, f) for f in (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3)]
    th, tar = [c.threshold for c in cs], [c.tar for c in cs]
    assert all(a >= b for a, b in zip(th, th[1:])) and all(a <= b for a, b in zip(tar, tar[1:]))  # looser FAR: lower threshold, higher TAR


@pytest.mark.parametrize("seed", range(5))
def test_calibration_ignores_trial_order_and_follows_affine_changes_of_the_score(seed: int) -> None:
    s, y = trials(seed, n_imp=2000)
    base = calibrate_threshold(s, y, 0.01)
    p = np.random.default_rng(seed + 100).permutation(len(s))
    assert calibrate_threshold(s[p], y[p], 0.01).threshold == pytest.approx(base.threshold)
    t = calibrate_threshold(s * 2.5 + 3.0, y, 0.01)
    assert t.threshold == pytest.approx(base.threshold * 2.5 + 3.0, abs=1e-6) and t.tar == pytest.approx(base.tar)


def test_calibration_agrees_with_tar_at_far_and_flags_unreliable_far() -> None:
    s, y = trials(3, n_imp=500)
    c = calibrate_threshold(s, y, 0.001)
    r = tar_at_far(s, y, 0.001)
    assert c.threshold == pytest.approx(r["threshold"]) and c.tar == pytest.approx(r["tar"])
    assert not c.reliable and "cannot be estimated reliably" in (c.note or "")  # 500 impostor trials cannot say anything about 0.1%
    assert calibrate_threshold(*trials(3, n_imp=12000), 0.001).reliable


def test_calibration_with_ties_and_degenerate_inputs() -> None:
    s = np.array([0.5] * 50 + [0.9] * 10)
    y = np.array([0] * 50 + [1] * 10)
    c = calibrate_threshold(s, y, 0.01)
    assert c.achieved_far == 0.0 and c.threshold is not None and c.threshold > 0.5  # all-equal impostors: the threshold goes just above them
    assert calibrate_threshold([0.1, 0.2], [0, 0], 0.01).threshold is None  # no genuine trials: nothing to calibrate, the gate then reads no_threshold
    assert calibrate_threshold([0.1, 0.2], [1, 1], 0.01).threshold is None
    assert calibrate_threshold([], [], 0.01).threshold is None
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            calibrate_threshold(s, y, bad)
    with pytest.raises(ValueError):
        calibrate_threshold([0.1, float("nan")], [0, 1], 0.1)
    with pytest.raises(ValueError):
        calibrate_threshold([0.1, 0.2], [0, 2], 0.1)
    with pytest.raises(ValueError):
        calibrate_threshold([0.1, 0.2, 0.3], [0, 1], 0.1)


# ------------------------------------------------------------------------------------------------ decide: the pure rule
def spec_eligible(enrolled: bool, enr_ok: bool, model_ok: bool, thr: float | None, speech: float, min_s: float, intent: str | None, conf: float | None, min_conf: float, score: float | None) -> bool:
    """Independent restatement of the rule in the module docstring."""
    return bool(enrolled and enr_ok and model_ok and thr is not None and speech >= min_s and intent == "REASSURANCE" and conf is not None and conf >= min_conf and score is not None and score >= thr)


@pytest.mark.parametrize("enrolled", [True, False])
@pytest.mark.parametrize("model_ok", [True, False])
@pytest.mark.parametrize("thr", [0.6, None])
@pytest.mark.parametrize("speech", [0.4, 1.0, 3.0])
@pytest.mark.parametrize("intent", ["REASSURANCE", "HELP", "OTHER", "UNCERTAIN", None])
@pytest.mark.parametrize("conf", [-0.3, -1.0, -2.0, None])
@pytest.mark.parametrize("score", [0.2, 0.6, 0.9, None])
def test_decide_is_eligible_exactly_when_every_condition_holds(enrolled, model_ok, thr, speech, intent, conf, score) -> None:  # type: ignore[no-untyped-def]
    d = decide(enrolled=enrolled, model_ok=model_ok, threshold=thr, speech_s=speech, min_speech_s=1.0, intent=intent, asr_conf=conf, min_asr_logprob=-1.0, score=score)
    assert d.cancel_eligible == spec_eligible(enrolled, True, model_ok, thr, speech, 1.0, intent, conf, -1.0, score)
    assert d.cancel_eligible == (len(d.reasons) == 0) and set(d.reasons) <= set(cg.REASON_CODES) and len(set(d.reasons)) == len(d.reasons)
    if d.cancel_eligible:
        assert d.speaker_verified is True and d.speaker_score is not None and d.threshold is not None


def test_decide_names_each_reason() -> None:
    ok = dict(enrolled=True, threshold=0.6, speech_s=2.0, min_speech_s=1.0, intent="REASSURANCE", asr_conf=-0.3, min_asr_logprob=-1.0, score=0.8)
    assert decide(**ok).cancel_eligible
    assert decide(**{**ok, "enrolled": False}).reasons == (cg.NO_ENROLLMENT,)
    assert decide(**{**ok, "enrollment_model_ok": False}).reasons == (cg.ENROLLMENT_MODEL_MISMATCH,)
    assert decide(**{**ok, "model_ok": False, "score": None}).reasons[0] == cg.MODEL_MISSING
    assert decide(**{**ok, "threshold": None}).reasons == (cg.NO_THRESHOLD,)
    assert decide(**{**ok, "threshold": float("nan")}).reasons == (cg.NO_THRESHOLD,)
    assert decide(**{**ok, "speech_s": 0.5}).reasons == (cg.TOO_SHORT,)
    assert decide(**{**ok, "speech_s": float("nan")}).reasons == (cg.TOO_SHORT,)
    assert decide(**{**ok, "intent": "HELP"}).reasons == (cg.INTENT_NOT_REASSURANCE,)
    assert decide(**{**ok, "intent": None}).reasons == (cg.INTENT_UNAVAILABLE,)
    assert decide(**{**ok, "asr_conf": -1.5}).reasons == (cg.LOW_ASR_CONFIDENCE,)
    assert decide(**{**ok, "asr_conf": None}).reasons == (cg.LOW_ASR_CONFIDENCE,)
    assert decide(**{**ok, "score": 0.3}).reasons == (cg.SPEAKER_MISMATCH,) and decide(**{**ok, "score": 0.3}).speaker_verified is False
    assert decide(**{**ok, "embedding_ok": False}).reasons == (cg.EMBEDDING_ERROR,)
    assert decide(**{**ok, "score": float("nan")}).reasons == (cg.EMBEDDING_ERROR,)
    assert decide(**{**ok, "score": None}).reasons == (cg.EMBEDDING_ERROR,)  # everything else fine but no score: still not eligible
    assert decide(**{**ok, "score": 0.6}).cancel_eligible  # exactly at the threshold counts as a match
    multi = decide(**{**ok, "speech_s": 0.2, "intent": "OTHER", "score": 0.1})
    assert set(multi.reasons) == {cg.TOO_SHORT, cg.INTENT_NOT_REASSURANCE, cg.SPEAKER_MISMATCH}  # all unmet conditions are reported, not only the first
    assert not decide(**{**ok, "intent": "HELP", "asr_conf": -3.0}).reasons.count(cg.LOW_ASR_CONFIDENCE)  # ASR confidence only matters for a reassurance


def test_decision_record_is_json_safe_and_marked_evidence_only() -> None:
    d = decide(enrolled=True, threshold=0.61234567, speech_s=2.34567, min_speech_s=1.0, intent="REASSURANCE", asr_conf=-0.2, min_asr_logprob=-1.0, score=0.7777777)
    rec = d.to_record()
    json.dumps(rec)
    assert rec["cancel_policy"] == cg.POLICY == "evidence_only"
    assert rec["cancel_eligible"] is True and rec["cancel_reasons"] == [] and rec["speaker_verified"] is True and rec["speaker_score"] == 0.7778
    assert set(rec) == {"cancel_eligible", "cancel_reasons", "cancel_policy", "speaker_verified", "speaker_score", "voiceprint_threshold", "speech_s"}


# ------------------------------------------------------------------------------------------------ the gate
def test_gate_eligible_when_everything_holds_and_embeds_only_the_speech() -> None:
    gate, emb = make_gate()
    audio = reply(2.5)
    keep = audio.copy()
    d = gate.evaluate(audio, REASSURE, enrollment(), -0.3)
    assert d.cancel_eligible and d.reasons == () and d.speaker_verified is True and d.speaker_score is not None and d.speaker_score >= 0.6
    assert d.speech_s == pytest.approx(2.5, abs=0.05)
    assert len(emb.seen) == 1 and len(emb.seen[0]) < 3.5 * SR and not np.any(emb.seen[0])  # speech crop only, zeroed before evaluate returned
    assert np.array_equal(audio, keep)  # the reply recording itself is not modified (the inquiry service zeroes it)


def test_gate_without_enrollment_is_false_and_never_loads_the_speaker_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("the speaker model must not be loaded without an enrollment")

    monkeypatch.setattr("homeease.ml.audio.speaker.WeSpeakerEmbedder", boom)
    gate = CancelGate(0.6, vad=StubVAD())
    d = gate.evaluate(reply(2.5), REASSURE, None, -0.3)
    assert not d.cancel_eligible and cg.NO_ENROLLMENT in d.reasons and d.speaker_score is None and d.speaker_verified is None


def test_gate_with_a_missing_speaker_model_reports_model_missing_and_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def missing(name: str):  # type: ignore[no-untyped-def]
        calls.append(name)
        raise FileNotFoundError("speaker model missing: x")

    monkeypatch.setattr("homeease.ml.audio.speaker.WeSpeakerEmbedder", missing)
    gate = CancelGate(0.6, vad=StubVAD())
    for _ in range(3):
        d = gate.evaluate(reply(2.5), REASSURE, enrollment(), -0.3)
        assert not d.cancel_eligible and cg.MODEL_MISSING in d.reasons and d.speaker_score is None
    assert len(calls) == 1 and "FileNotFoundError" in (gate.load_error or "")


def test_gate_reports_each_reason_and_stays_false() -> None:
    enr = enrollment()
    gate, _ = make_gate()
    cases = [
        (reply(0.4), REASSURE, -0.3, cg.TOO_SHORT),
        (reply(2.5), {"intent": "HELP", "reason": "help_cue"}, -0.3, cg.INTENT_NOT_REASSURANCE),
        (reply(2.5), {"intent": "OTHER", "reason": "no_cue"}, -0.3, cg.INTENT_NOT_REASSURANCE),
        (reply(2.5), {"intent": "UNCERTAIN", "reason": "low_asr_confidence"}, -0.3, cg.INTENT_NOT_REASSURANCE),
        (reply(2.5), None, -0.3, cg.INTENT_UNAVAILABLE),
        (reply(2.5), REASSURE, -1.4, cg.LOW_ASR_CONFIDENCE),
        (reply(2.5), REASSURE, None, cg.LOW_ASR_CONFIDENCE),
        (np.zeros(10 * SR, dtype=np.float32), REASSURE, -0.3, cg.TOO_SHORT),
        (np.zeros(0, dtype=np.float32), REASSURE, -0.3, cg.TOO_SHORT),
    ]
    for audio, intent, conf, code in cases:
        d = gate.evaluate(audio, intent, enr, conf)
        assert not d.cancel_eligible and code in d.reasons, (code, d)


def test_gate_other_speaker_and_wrong_model_and_missing_threshold() -> None:
    enr = enrollment(unit(1))
    other, _ = make_gate(emb=unit(99))  # a different voice
    d = other.evaluate(reply(2.5), REASSURE, enr, -0.3)
    assert not d.cancel_eligible and d.reasons == (cg.SPEAKER_MISMATCH,) and d.speaker_verified is False and d.speaker_score is not None and d.speaker_score < 0.6
    gate, emb = make_gate()
    d = gate.evaluate(reply(2.5), REASSURE, enrollment(model_id="another_model"), -0.3)
    assert not d.cancel_eligible and cg.ENROLLMENT_MODEL_MISMATCH in d.reasons and not emb.seen  # no comparison across embedding spaces
    nothr, _ = make_gate(threshold=None)
    d = nothr.evaluate(reply(2.5), REASSURE, enr, -0.3)
    assert not d.cancel_eligible and d.reasons == (cg.NO_THRESHOLD,) and d.speaker_score is not None and d.speaker_verified is None


def test_gate_embedding_failure_is_false_not_an_exception() -> None:
    gate, _ = make_gate(fail=True)
    d = gate.evaluate(reply(2.5), REASSURE, enrollment(), -0.3)
    assert not d.cancel_eligible and cg.EMBEDDING_ERROR in d.reasons and d.speaker_score is None
    nan, _ = make_gate(emb=np.full(DIM, np.nan, dtype=np.float32))
    assert not nan.evaluate(reply(2.5), REASSURE, enrollment(), -0.3).cancel_eligible
    wrong_dim, _ = make_gate(emb=unit(1, DIM + 3))
    assert not wrong_dim.evaluate(reply(2.5), REASSURE, enrollment(), -0.3).cancel_eligible  # dimension mismatch is an error, not a match


def test_gate_scores_the_voice_even_when_the_words_do_not_reassure_but_never_makes_it_eligible() -> None:
    gate, _ = make_gate()
    d = gate.evaluate(reply(2.5), {"intent": "HELP", "reason": "help_cue"}, enrollment(), -0.3)
    assert d.speaker_verified is True and d.speaker_score is not None  # the enrolled voice asked for help: useful evidence
    assert not d.cancel_eligible  # but a call for help can never be a cancellation


def test_gate_keeps_no_audio() -> None:
    gate, emb = make_gate()
    audio = reply(2.5)
    gate.evaluate(audio, REASSURE, enrollment(), -0.3)
    held = [v for v in vars(gate).values() if isinstance(v, np.ndarray)]
    assert not held
    assert not np.any(emb.seen[0])


def test_gate_from_config_reads_the_voiceprint_section() -> None:
    from homeease.core.config import AudioCfg

    cfg = AudioCfg()
    g = CancelGate.from_config(cfg, StubEmbedder())
    assert g.threshold is None and g.min_speech_s == 1.0 and g.min_asr_logprob == -1.0  # no calibrated threshold ships: nothing is eligible by default
    cfg.voiceprint.threshold = 0.55
    cfg.voiceprint.min_speech_s = 1.5
    g = CancelGate.from_config(cfg, StubEmbedder())
    assert g.threshold == 0.55 and g.min_speech_s == 1.5


def test_voiceprint_config_bounds_and_the_policy_validators_stay() -> None:
    from pydantic import ValidationError

    from homeease.core.config import AudioCfg, VoiceprintCfg

    for bad in ({"threshold": 1.5}, {"threshold": -2.0}, {"min_speech_s": 0.0}, {"min_asr_logprob": 0.5}, {"unknown": 1}):
        with pytest.raises(ValidationError):
            VoiceprintCfg(**bad)
    with pytest.raises(ValidationError):
        AudioCfg(auto_cancel_emergency=True)  # a voiceprint match does not unlock automatic cancellation
    with pytest.raises(ValidationError):
        AudioCfg(voiceprint={"auto_cancel": True})  # type: ignore[arg-type]


def test_speech_crop_drops_silence_and_counts_speech() -> None:
    crop, sp = cg.speech_crop(reply(1.5, total_s=8.0, start_s=3.0), StubVAD())
    assert sp == pytest.approx(1.5, abs=0.03) and 1.5 * SR <= len(crop) <= 1.8 * SR and np.max(np.abs(crop)) > 0.1
    c0, s0 = cg.speech_crop(np.zeros(SR, dtype=np.float32), StubVAD())
    assert len(c0) == 0 and s0 == 0.0
    c1, s1 = cg.speech_crop(np.zeros(100, dtype=np.float32), StubVAD())  # shorter than one frame
    assert s1 == 0.0 and len(c1) == 0


def test_reason_codes_are_stable_strings() -> None:
    assert set(cg.REASON_CODES) >= {"no_enrollment", "model_missing", "too_short", "intent_not_reassurance", "speaker_mismatch", "low_asr_confidence", "no_threshold", "embedding_error"}
    assert all(isinstance(c, str) and c == c.lower() and " " not in c for c in cg.REASON_CODES)
    assert not math.isnan(cg.DEFAULT_MIN_SPEECH_S)
