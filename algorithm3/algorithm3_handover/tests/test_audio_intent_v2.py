"""Intent v2: script normalisation, negation scope, conservative combination, supervised variant (text only, no audio)."""

from __future__ import annotations

import pytest

from homeease.ml.audio import intent_v2 as I
from homeease.ml.audio.intent import RuleIntent, phrase_corpus

R2 = I.RuleIntentV2()


def test_traditional_to_simplified_table_is_aligned() -> None:
    pairs = I._PAIRS.split()
    assert all(len(w) == 2 and w[0] != w[1] for w in pairs)  # every entry is one real conversion
    assert len({w[0] for w in pairs}) == len(pairs)  # no character mapped twice
    for trad, simp in (("\u6c92\u4e8b", "\u6ca1\u4e8b"), ("\u6551\u8b77\u8eca", "\u6551\u62a4\u8f66"), ("\u64d4\u5fc3", "\u62c5\u5fc3"), ("\u6211\u9084\u597d", "\u6211\u8fd8\u597d"), ("\u6c92\u6709\u53d7\u50b7", "\u6ca1\u6709\u53d7\u4f24"), ("\u8b1d\u8b1d", "\u8c22\u8c22"), ("\u982d\u6688", "\u5934\u6655"), ("\u5e6b\u5e6b\u6211", "\u5e2e\u5e2e\u6211"), ("\u8d77\u4e0d\u4f86", "\u8d77\u4e0d\u6765")):
        assert I.to_simplified(trad) == simp
    assert I.to_simplified("\u6454\u5012\u4e86 hello 123") == "\u6454\u5012\u4e86 hello 123"  # untouched characters stay


def test_normalise_v2() -> None:
    assert I.normalise_v2("I’m All Right!") == "i'm alright"
    assert I.normalise_v2("\u6211 \u6c92\u4e8b \u5566") == "\u6211\u6ca1\u4e8b\u5566"
    assert I.normalise_v2("OK, can not move") == "ok cannot move"
    assert I.normalise_v2(None) == ""  # type: ignore[arg-type]


HELP = [
    "Help", "help me please", "I fell down", "I have fallen", "I can't get up", "I cannot stand up", "I'm stuck on the floor", "Call an ambulance", "call my son",
    "I'm not okay", "I am not feeling fine", "I don't feel well", "it hurts", "my knee hurts a lot", "I'm in pain", "Ouch", "Ow!", "Aiyo my leg", "I slipped in the kitchen",
    "I'm fine but I can't get up", "I'm okay, but I fell and it hurts", "No problem but please call an ambulance", "I need a doctor", "something is wrong",
    "\u6551\u547d", "\u6551\u6551\u6211", "\u6211\u6454\u5012\u4e86", "\u6211\u8dcc\u5012\u4e86", "\u6211\u8d77\u4e0d\u6765\u4e86", "\u6211\u52a8\u4e0d\u4e86", "\u597d\u75db\u554a", "\u53eb\u6551\u62a4\u8f66", "\u6211\u4e0d\u592a\u597d", "\u6211\u5934\u6655", "\u6211\u6ca1\u4e8b\uff0c\u4f46\u662f\u6211\u8d77\u4e0d\u6765", "\u6c92\u95dc\u4fc2\uff0c\u4e0d\u904e\u6211\u597d\u50cf\u6454\u50b7\u4e86", "\u6211\u611f\u89c9\u4e0d\u5bf9\u52b2",
    "\u6551\u8b77\u8eca", "\u6211\u7ad9\u4e0d\u8d77\u4f86", "\u5feb\u4f86\u5e6b\u5e6b\u6211",
    "\u6211\u8d77\u4e0d\u6765\u5e2e\u5e2e\u6211", "\u6211\u4e0d\u884c\u4e86\u6551\u547d", "\u6211\u52a8\u4e0d\u4e86\u5e2e\u6211", "\u522b\u8d70\u6551\u547d", "\u4e0d\u7528\u53eb\u533b\u751f\uff0c\u6551\u547d", "\u6211\u4e0d\u8981\u7d27\uff0c\u5e2e\u5e2e\u6211", "don't leave me, help", "no need to worry, call an ambulance",  # a negator next to a real call
]
REASSURANCE = [
    "I'm fine", "I am okay", "I'm okay lah", "I'm alright", "No problem", "Don't worry about me", "no worries", "All good here", "I'm doing fine", "I'm not hurt", "Nothing hurts, I'm fine",
    "I didn't fall", "I'm unhurt", "I don't need any help", "It's nothing", "She's fine, just resting", "Everything is under control", "Fine", "Okay", "Yes fine",
    "\u6211\u6ca1\u4e8b", "\u6c92\u4e8b", "\u6ca1\u4e8b\u5566", "\u6211\u5f88\u597d", "\u6211\u9084\u597d", "\u4e0d\u7528\u62c5\u5fc3", "\u5225\u64d4\u5fc3", "\u6ca1\u5173\u7cfb", "\u6ca1\u95ee\u9898", "\u6211\u6ca1\u6709\u53d7\u4f24", "\u6211\u4e00\u70b9\u90fd\u4e0d\u75db", "\u4e00\u70b9\u513f\u4e5f\u4e0d\u75bc", "\u4ed6\u6ca1\u6709\u6454\u5012", "\u6ca1\u6709\u4eba\u53d7\u4f24",
    "\u6211\u4e0d\u9700\u8981\u5e2e\u52a9", "\u4e0d\u7528\u5e2e\u5fd9", "\u4e0d\u7528\u53eb\u6551\u62a4\u8f66", "\u522b\u53eb\u533b\u751f", "\u4e0d\u7528\u6551\u6211", "don't call an ambulance", "no need to call the doctor", "I'm fine, no need to call an ambulance",  # a refusal of help
]
OTHER = [
    "Um", "Hmm let me think", "Yes", "No", "Nope", "yeah", "What time is it", "Who is at the door", "The weather is nice today", "I can't hear you", "Where are my glasses",
    "My doctor visit is tomorrow", "Temperatures will fall overnight", "I'm watching television", "Please turn on the light", "Dinner is ready",
    "\u55ef", "\u662f", "\u4e0d\u662f", "\u51e0\u70b9\u4e86", "\u4eca\u5929\u5929\u6c14\u4e0d\u9519", "\u6211\u5728\u770b\u7535\u89c6", "\u597d\u7684", "\u4f60\u8bf4\u4ec0\u4e48", "\u6ca1\u4e8b\u5427", "\u6211\u7684\u773c\u955c\u5728\u54ea\u91cc",
]


@pytest.mark.parametrize("text", HELP)
def test_help_cases(text: str) -> None:
    assert R2.predict(text)["intent"] == "HELP", R2.cues(text)


@pytest.mark.parametrize("text", REASSURANCE)
def test_reassurance_cases(text: str) -> None:
    assert R2.predict(text)["intent"] == "REASSURANCE", R2.cues(text)


@pytest.mark.parametrize("text", OTHER)
def test_other_cases(text: str) -> None:
    assert R2.predict(text)["intent"] == "OTHER", R2.cues(text)


def test_mixed_sentences_any_help_cue_wins() -> None:
    for ok in REASSURANCE[:20]:
        for tail in (" but I can't get up", ", but I fell", ", but call an ambulance", " but it hurts"):
            assert R2.predict(ok + tail)["intent"] == "HELP", ok + tail
    r = R2.predict("I'm fine but I can't get up")
    assert r["reason"] == "help_cue_overrides_reassurance"
    for ok in ("\u6211\u6ca1\u4e8b", "\u6ca1\u5173\u7cfb", "\u4e0d\u7528\u62c5\u5fc3"):
        assert R2.predict(ok + "\uff0c\u4f46\u662f\u6211\u8d77\u4e0d\u6765")["intent"] == "HELP"


def test_nothing_unintelligible_is_ever_reassurance_or_help() -> None:
    for junk in ("", "   ", "...", "asdf qwer", "uh", "\u570b\u65d7\u5df2\u4e0d\u4f86\u4e86", "\u5b50", "Alan", "\u6c34\u5728\u9580\u53e3", "a b c d e f g"):
        assert R2.predict(junk)["intent"] in ("UNCERTAIN", "OTHER"), junk
    assert R2.predict("")["intent"] == "UNCERTAIN"


def test_low_asr_confidence_without_cue_is_uncertain_but_cues_still_count() -> None:
    assert R2.predict("blah blah", -2.0)["intent"] == "UNCERTAIN"
    assert R2.predict("help", -2.0)["intent"] == "HELP"  # a garbled-but-help transcript still escalates
    assert R2.predict("I'm fine", -2.0)["intent"] == "REASSURANCE"


def test_v1_failures_that_v2_fixes() -> None:
    v1 = RuleIntent()
    assert v1.predict("I'm not hurt")["intent"] == "HELP" and R2.predict("I'm not hurt")["intent"] == "REASSURANCE"
    assert v1.predict("\u6c92\u4e8b")["intent"] == "OTHER" and R2.predict("\u6c92\u4e8b")["intent"] == "REASSURANCE"  # Traditional script
    assert v1.predict("\u6211\u8dcc\u5012\u4e86")["intent"] == "OTHER" and R2.predict("\u6211\u8dcc\u5012\u4e86")["intent"] == "HELP"
    assert v1.predict("I need my glasses")["intent"] == "HELP" and R2.predict("I need my glasses")["intent"] == "OTHER"


def test_v1_ablation_on_simplified_script() -> None:
    a = I.RuleIntentV1Simplified()
    assert a.predict("\u6c92\u4e8b")["intent"] == "REASSURANCE" and a.predict("\u6551\u547d")["intent"] == "HELP"


def test_supervised_v2_reads_both_scripts() -> None:
    m = I.TextIntentLR().fit(phrase_corpus())
    assert m.predict("\u6211\u6c92\u4e8b")["intent"] == m.predict("\u6211\u6ca1\u4e8b")["intent"] == "REASSURANCE"
    assert m.predict("")["intent"] == "UNCERTAIN"


class _Fixed:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, str]:
        return {"intent": self.intent, "reason": "fixed"}


@pytest.mark.parametrize("a,b,out", [
    ("HELP", "OTHER", "HELP"), ("REASSURANCE", "HELP", "HELP"), ("HELP", "UNCERTAIN", "HELP"), ("HELP", "HELP", "HELP"),
    ("REASSURANCE", "REASSURANCE", "REASSURANCE"), ("REASSURANCE", "OTHER", "UNCERTAIN"), ("OTHER", "REASSURANCE", "UNCERTAIN"),
    ("REASSURANCE", "UNCERTAIN", "UNCERTAIN"), ("OTHER", "OTHER", "OTHER"), ("OTHER", "UNCERTAIN", "UNCERTAIN"), ("UNCERTAIN", "UNCERTAIN", "UNCERTAIN"),
])
def test_combined_is_conservative(a: str, b: str, out: str) -> None:
    assert I.CombinedIntent(_Fixed(a), _Fixed(b)).predict("x")["intent"] == out


def test_reassurance_never_comes_from_a_disagreement_or_failure() -> None:
    for a in ("HELP", "REASSURANCE", "OTHER", "UNCERTAIN"):
        for b in ("HELP", "REASSURANCE", "OTHER", "UNCERTAIN"):
            if (a, b) != ("REASSURANCE", "REASSURANCE"):
                assert I.CombinedIntent(_Fixed(a), _Fixed(b)).predict("x")["intent"] != "REASSURANCE"


def test_hotword_and_prompt_vocabularies_are_in_the_matching_script() -> None:
    assert "\u6551\u547d" in I.HOTWORDS["zh"] and "\u6551\u62a4\u8f66" in I.HOTWORDS["zh"] and "help" in I.HOTWORDS["en"]  # Simplified: biases the output script too
    assert I.HOTWORDS["auto"].startswith(I.HOTWORDS["en"]) and I.HOTWORDS["zh"] in I.HOTWORDS["auto"]
    assert set(I.PROMPTS) == {"en", "zh", "auto"}
