"""Response intent: HELP / REASSURANCE / OTHER / UNCERTAIN (English and Chinese).

NO_RESPONSE is not an intent: it is a workflow state set when the inquiry window ends
without usable speech. Conservative rules: any help cue wins over reassurance
("I'm fine but I can't get up" -> HELP); negated reassurance counts as HELP.
An intent is evidence only - it never closes or downgrades an event.
"""

from __future__ import annotations

import re
from typing import Any

INTENTS = ["HELP", "REASSURANCE", "OTHER", "UNCERTAIN"]

HELP_EN = [r"\bhelp\b", r"\bi fell\b", r"\bfallen\b", r"can'?t get up", r"cannot get up", r"\bhurts?\b", r"\bpain\b", r"\bambulance\b", r"\bcall (someone|my|a)\b", r"\bnot (ok|okay|fine|alright|all right)\b", r"\bi need\b"]
HELP_ZH = ["救命", "帮帮我", "帮我", "需要帮助", "摔倒", "起不来", "疼", "痛", "救护车", "不舒服", "不好"]
OK_EN = [r"\bi'?m (fine|ok|okay|alright|all right|good)\b", r"\bi am (fine|ok|okay|alright|all right|good)\b", r"\bno problem\b", r"\bdon'?t worry\b", r"\bnot hurt\b", r"^\s*(fine|ok|okay)\s*$", r"\bi'?m not hurt\b"]
OK_ZH = ["我没事", "没事", "我很好", "不用担心", "没关系", "我还好", "没问题", "挺好"]


def normalise(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^\w\s'一-鿿]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class RuleIntent:
    name = "rule"
    version = "1.0"

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        t = normalise(text or "")
        if not t:
            return {"intent": "UNCERTAIN", "reason": "empty_transcript"}
        help_hit = any(re.search(p, t) for p in HELP_EN) or any(k in t for k in HELP_ZH)
        ok_hit = any(re.search(p, t) for p in OK_EN) or any(k in t for k in OK_ZH)
        if "不好" in t or re.search(r"\bnot (ok|okay|fine|alright)\b", t):
            ok_hit = False
            help_hit = True
        if asr_conf is not None and asr_conf < -1.2 and not (help_hit or ok_hit):
            return {"intent": "UNCERTAIN", "reason": "low_asr_confidence"}
        if help_hit:
            return {"intent": "HELP", "reason": "help_cue" + ("_overrides_reassurance" if ok_hit else "")}
        if ok_hit:
            return {"intent": "REASSURANCE", "reason": "reassurance_cue"}
        return {"intent": "OTHER", "reason": "no_cue"}


def phrase_corpus() -> list[tuple[str, str]]:
    """Synthetic training phrases (text only) for the supervised baseline."""
    help_ = ["help", "help me", "please help", "i need help", "i fell down", "i can't get up", "my hip hurts", "it hurts", "call an ambulance", "call my daughter",
             "救命", "帮帮我", "快来帮我", "我需要帮助", "我摔倒了", "我起不来", "好疼", "腿很痛", "叫救护车", "我不舒服"]
    ok = ["i'm fine", "i am fine", "i'm okay", "i'm ok", "i'm alright", "i'm all right", "no problem", "don't worry", "i'm not hurt", "i'm good thanks",
          "我没事", "没事的", "我很好", "不用担心", "没关系", "我还好", "没问题", "我挺好的", "放心吧我没事", "不用管我"]
    other = ["what time is it", "the weather is nice today", "turn on the television", "where are my glasses", "dinner is ready", "who is at the door",
             "现在几点了", "今天天气不错", "把电视打开", "我的眼镜呢", "吃饭了", "谁在门口", "hello", "你好", "yes", "是的", "no", "不是"]
    return [(p, "HELP") for p in help_] + [(p, "REASSURANCE") for p in ok] + [(p, "OTHER") for p in other]


class SupervisedIntent:
    name = "char_ngram_lr"
    version = "1.0"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)), LogisticRegression(max_iter=2000, class_weight="balanced"))

    def fit(self, pairs: list[tuple[str, str]]) -> "SupervisedIntent":
        self.pipe.fit([normalise(t) for t, _ in pairs], [y for _, y in pairs])
        return self

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        t = normalise(text or "")
        if not t:
            return {"intent": "UNCERTAIN", "reason": "empty_transcript"}
        probs = self.pipe.predict_proba([t])[0]
        k = int(probs.argmax())
        if probs[k] < 0.45:
            return {"intent": "UNCERTAIN", "reason": "low_classifier_confidence", "p": float(probs[k])}
        return {"intent": str(self.pipe.classes_[k]), "reason": "classifier", "p": float(probs[k])}
