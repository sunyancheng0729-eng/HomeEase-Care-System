"""Reply intent v2 (HELP / REASSURANCE / OTHER / UNCERTAIN): script normalisation, a broader rule with negation scope,
a supervised variant and a conservative combination. `intent.py` (v1, published numbers) is left untouched.

Why a second module:
* Whisper writes Mandarin in Traditional characters ("\u6c92\u4e8b", "\u6551\u8b77\u8eca") while the v1 lexicon is Simplified, so most
  Mandarin cues never matched. `to_simplified` maps the transcript to Simplified before any matching.
* v1 flagged "I'm not hurt" and "nothing hurts" as HELP (the word "hurt") and missed everyday help words
  ("stuck", "slipped", "ouch", "\u8dcc\u5012", "\u52a8\u4e0d\u4e86"). v2 looks at negation next to harm words and has more cues.

Safety semantics are unchanged: an intent is EVIDENCE only. Whenever a help cue and a reassurance cue are both
present the help cue wins ("I'm fine but I can't get up" -> HELP). Failures to understand give OTHER / UNCERTAIN, never
REASSURANCE. REASSURANCE here means "the words reassure", not "this is the monitored person" (voiceprint, later step).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from homeease.ml.audio.intent import SupervisedIntent

# ---------------------------------------------------------------- Traditional -> Simplified (characters seen in everyday replies)
# pairs 'TraditionalSimplified' separated by blanks; characters of everyday speech (body, family, weather, shop), not a general converter
_PAIRS = (
    "\u6c92\u6ca1 \u95dc\u5173 \u4fc2\u7cfb \u554f\u95ee \u984c\u9898 \u64d4\u62c5 \u9084\u8fd8 \u50b7\u4f24 \u8b77\u62a4 \u8eca\u8f66 \u8a71\u8bdd \u96fb\u7535 \u5e6b\u5e2e \u96e3\u96be \u91ab\u533b \u9ebc\u4e48 \u8b1d\u8c22 \u9ede\u70b9 \u9019\u8fd9 \u4f86\u6765 \u7dca\u7d27 \u5f35\u5f20 \u70ba\u4e3a \u5c0d\u5bf9 \u8aaa\u8bf4 \u898b\u89c1 \u807d\u542c \u6642\u65f6 \u71c8\u706f \u93e1\u955c \u88e1\u91cc \u6c23\u6c14 \u805e\u95fb "
    "\u71b1\u70ed \u958b\u5f00 \u9580\u95e8 \u8ab0\u8c01 \u55ce\u5417 \u5011\u4eec \u500b\u4e2a \u904e\u8fc7 \u8b93\u8ba9 \u8acb\u8bf7 \u88dd\u88c5 \u8166\u8111 \u982d\u5934 \u8173\u811a \u52d5\u52a8 \u6688\u6655 \u60b6\u95f7 \u559a\u5524 \u885b\u536b \u5e2b\u5e08 \u96aa\u9669 \u56b4\u4e25 \u5be6\u5b9e \u969b\u9645 \u7169\u70e6 \u6b61\u6b22 \u9858\u613f \u61c9\u5e94 \u8a72\u8be5 \u7576\u5f53 \u7e3d\u603b \u7d93\u7ecf \u6b72\u5c81 "
    "\u5920\u591f \u5e7e\u51e0 \u6975\u6781 \u6a23\u6837 \u908a\u8fb9 \u88cf\u91cc \u5f8c\u540e \u570b\u56fd \u5b78\u5b66 \u6703\u4f1a \u9577\u957f \u6771\u4e1c \u8cb7\u4e70 \u8ce3\u5356 \u9322\u94b1 \u98ef\u996d \u9eb5\u9762 \u9b5a\u9c7c \u96de\u9e21 \u9ce5\u9e1f \u99ac\u9a6c \u66f8\u4e66 \u5831\u62a5 \u7d19\u7eb8 \u7b46\u7b14 \u8072\u58f0 \u97ff\u54cd \u6a02\u4e50 \u8996\u89c6 \u7bc0\u8282 \u6a5f\u673a \u986f\u663e \u89c0\u89c2 "
    "\u89ba\u89c9 \u89aa\u4eb2 \u611b\u7231 \u5152\u513f \u5abd\u5988 \u723a\u7237 \u5b6b\u5b59 \u5a66\u5987 \u7d05\u7ea2 \u7da0\u7eff \u85cd\u84dd \u9ec3\u9ec4 \u7121\u65e0 \u73fe\u73b0 \u767c\u53d1 \u689d\u6761 \u7522\u4ea7 \u696d\u4e1a \u52d9\u52a1 \u5834\u573a \u7fd2\u4e60 \u5099\u5907 \u5e36\u5e26 \u81fa\u53f0 \u7063\u6e7e \u5340\u533a \u5ee3\u5e7f \u5c08\u4e13 \u7d50\u7ed3 \u69cb\u6784 \u8a2d\u8bbe \u8a08\u8ba1 \u8ad6\u8bba "
    "\u8a66\u8bd5 \u9a57\u9a8c \u6e96\u51c6 \u96e2\u79bb \u9032\u8fdb \u904b\u8fd0 \u9060\u8fdc \u9023\u8fde \u7e8c\u7eed \u9054\u8fbe \u9031\u5468 \u865f\u53f7 \u7a31\u79f0 \u8a31\u8bb8 \u8a8d\u8ba4 \u8b58\u8bc6 \u8a18\u8bb0 \u61b6\u5fc6 \u932f\u9519 \u8cb4\u8d35 \u50f9\u4ef7 \u8cec\u8d26 \u55ae\u5355 \u6578\u6570 \u64da\u636e \u8207\u4e0e \u65bc\u4e8e \u5f9e\u4ece \u7d66\u7ed9 \u5c07\u5c06 \u4e26\u5e76 \u5e25\u5e05 \u6368\u820d "
    "\u98db\u98de \u806f\u8054 \u6de8\u51c0 \u975c\u9759 \u5716\u56fe \u9928\u9986 \u5712\u56ed \u85e5\u836f \u91dd\u9488 \u8a3a\u8bca \u7642\u7597 \u6aa2\u68c0 \u8655\u5904 \u65b7\u65ad \u6b78\u5f52 \u5beb\u5199 \u9418\u949f \u967d\u9633 \u96f2\u4e91 \u98a8\u98ce \u9670\u9634 \u9810\u9884 \u6eab\u6e29 \u8f03\u8f83 \u5287\u5267 \u5ef3\u5385 \u5eda\u53a8 \u9593\u95f4 \u8c93\u732b \u7210\u7089 \u934b\u9505 \u9e7d\u76d0 \u91ac\u9171 "
    "\u7dda\u7ebf \u5225\u522b \u58de\u574f \u5687\u5413 \u6182\u5fe7 \u616e\u8651 \u60f1\u607c \u5f48\u5f39 \u55b2\u54df \u81c9\u8138 \u6a13\u697c \u5ec1\u5395 \u81e5\u5367 \u9ad4\u4f53 \u52c1\u52b2 \u96a8\u968f \u7d42\u7ec8 \u8f15\u8f7b \u9b06\u677e \u8d95\u8d76 \u9072\u8fdf \u6dbc\u51c9 \u6fd5\u6e7f \u71df\u8425 \u512a\u4f18 \u5fa9\u590d \u8ab2\u8bfe \u89bd\u89c8 \u8b70\u8bae \u76e4\u76d8 \u6f32\u6da8 \u6cc1\u51b5"
)
_T2S_TABLE = str.maketrans({w[0]: w[1] for w in _PAIRS.split()})


def to_simplified(text: str) -> str:
    """Traditional -> Simplified for the characters in `_PAIRS` (others unchanged). Not a general converter."""
    return text.translate(_T2S_TABLE)


def normalise_v2(text: str) -> str:
    """Lower case, Simplified script, no punctuation, no blanks between Chinese characters, 'all right' -> 'alright'."""
    t = to_simplified(text or "").lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^\w\s'\u4e00-\u9fff]", " ", t)
    t = re.sub(r"\bcan not\b", "cannot", t)
    t = re.sub(r"\ball right\b", "alright", t)
    t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- lexicon
def _w(p: str) -> str:
    """Word boundaries that also work next to Chinese characters (\\b does not: CJK counts as a word character)."""
    return rf"(?<![a-z']){p}(?![a-z'])"


# Harm / event words. A negation just before them turns the cue into reassurance ("not hurt", "nothing hurts", "didn't fall").
_HARM_EN = re.compile(_w(
    r"(?:hurts?|hurting|hurt|pain|painful|aching|aches?|fell|fallen|fall(?:s|ing)? (?:down|over|off)|(?:a|bad|nasty|big) fall|slipped|tripped over|collapsed|fainted|dizzy|bleeding|injured|unwell)"))
_NEG_EN = re.compile(_w(r"(?:not|no|nobody|never|nothing|without|didn't|don't|doesn't|isn't|wasn't|haven't|hasn't|hadn't|aren't)"))
_HARM_ZH = re.compile(r"\u53d7\u4f24|\u6454\u4f24|\u6454\u5012|\u8dcc\u5012|\u6ed1\u5012|\u6454\u4e86|\u6454\u4e0b|\u6454\u4e00\u8de4|\u8dcc\u4e86|\u5012\u4e0b|\u75bc|\u75db|\u96be\u53d7|\u4e0d\u8212\u670d|\u5934\u6655|\u6655\u5012|\u51fa\u8840|\u6d41\u8840|\u9aa8\u6298|\u626d\u4f24")
_NEG_ZH_BEFORE = re.compile(r"(?:\u6ca1\u6709?\u4eba?|\u4e0d\u4f1a?|\u522b|\u65e0\u4eba?|\u672a|\u5e76\u6ca1\u6709?|\u4e00\u70b9[\u513f\u90fd\u4e5f]*)$")

# Help cues that need no negation check (a negated 'help' request is removed first, see _NO_HELP).
_HELP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [(n, re.compile(p)) for n, p in [
    ("help", _w(r"help")),
    ("need_help", _w(r"i need (?:help|a doctor|an ambulance|someone|somebody|medical|assistance)")),
    ("ambulance", _w(r"(?:ambulance|paramedics?|emergency)")),
    ("get_doctor", _w(r"(?:(?:call|get|fetch|find|need|send|bring)\s+(?:\w+\s+){0,2}(?:doctor|nurse|hospital)|(?:go to|take me to|taken to|rush me to|bring me to)\s+(?:the |a )?(?:hospital|clinic))")),
    ("call_someone", _w(r"call (?:for )?(?:someone|somebody|anyone|anybody|my|an?|the|your|him|her|them)")),
    ("cant_do", _w(r"(?:can't|cannot|unable to|couldn't) (?:\w+ )?(?:get up|get out|get back|stand|move|walk|breathe|feel|reach|sit up)")),
    ("stuck", _w(r"(?:stuck|trapped)")),
    ("on_floor", _w(r"(?:on|lying on|found .{0,12} on) the (?:floor|ground)")),
    ("not_well", _w(r"(?:not|don't feel|doesn't feel|isn't|wasn't|aren't) (?:feeling |doing |too |very |so |that |really )?(?:ok|okay|fine|alright|well|good|safe)")),
    ("something_wrong", _w(r"something(?:'s| is| was)? (?:wrong|not right)")),
    ("cry", _w(r"(?:ow+|ouch|owie|aiyo+h?|aiyah?|aduh|alamak|argh+|oh no)")),
    ("zh_help", r"\u6551\u547d|\u6551\u6551|\u6c42\u6551|\u6551\u6211|\u5e2e\u5e2e|\u5e2e\u5fd9|\u5e2e\u52a9|\u5e2e\u6211|\u6765\u4eba|\u6276\u6211|\u6276\u4ed6|\u6276\u5979|\u6276\u4e00\u4e0b|\u6276\u8d77"),
    ("zh_cant", r"\u8d77\u4e0d\u6765|\u7ad9\u4e0d\u8d77|\u722c\u4e0d\u8d77|\u7ad9\u4e0d\u4f4f|\u52a8\u4e0d\u4e86|\u8d70\u4e0d\u4e86|\u8d70\u4e0d\u52a8|\u4e0d\u80fd\u52a8|\u7ad9\u4e0d\u7a33|\u8d77\u4e0d\u4e86|\u5598\u4e0d\u8fc7\u6c14|\u5598\u4e0d\u4e0a\u6c14"),
    ("zh_ambulance", r"\u6551\u62a4|\u6025\u6551|\u6551\u63f4|[\u53bb\u9001\u5230]\u533b\u9662|(?:\u770b|\u627e|\u53eb|\u8bf7|\u8981|\u9700\u8981|\u5feb\u8bf7|\u5feb\u627e|\u5feb\u53eb)\u533b\u751f|\u533b\u751f\u6765"),
    ("zh_unwell", r"\u4e0d\u592a\u597d|\u4e0d\u597d(?!\u610f\u601d)|\u4e0d\u884c\u4e86|\u4e0d\u5bf9\u52b2|\u80f8\u95f7|\u80f8\u53e3[\u75db\u75bc\u95f7]"),
    ("zh_down", r"\u5012\u5728\u5730|\u8eba\u5728\u5730"),
    ("zh_cry", r"\u54ce\u54df|\u54ce\u5537|\u54ce\u5466|\u5509\u54df|\u554a\u54df|\u54ce\u5440"),
]]
# A refusal of help is reassurance and never a help cue. English: "I don't need help", "no need to call an ambulance". Chinese: the
# equivalents of "I do not need help", "no need to help", "no need to call an ambulance" and "do not call the doctor" (a negator that is
# directly followed by the help word, an optional person and an optional call).
_NO_HELP_EN = (
    r"(?:(?:don't|do not|doesn't|no|not|never|didn't) (?:need|want|require|ask for)(?: any| a| your| some)? (?:help|assistance)|no (?:help|assistance) (?:is )?(?:needed|necessary|required)"
    r"|(?:don't|do not|doesn't|no need to|no need for|need not|never|not|please don't|won't|stop) (?:(?:you|him|her|them|anyone|anybody|someone|somebody) )?"
    r"(?:call|calling|get|getting|fetch|send|sending|bring|phone|dial|ask for|asking for) (?:(?:for|an?|the|any|some|my|your) )*(?:ambulance|doctor|nurse|paramedics?|hospital|help|assistance|emergency))"
)
_NO_HELP_ZH = (
    r"(?:\u4e0d\u7528\u518d?|\u4e0d\u5fc5|\u4e0d\u9700\u8981|\u4e0d\u9700|\u4e0d\u8981\u518d?|\u4e0d\u60f3\u8981?|\u65e0\u9700|\u65e0\u987b|\u6ca1\u5fc5\u8981|\u6ca1\u6709\u5fc5\u8981|\u7528\u4e0d\u7740|\u522b\u518d?|\u83ab\u8981?|\u52ff|\u4e0d)(?:\u8981|\u60f3|\u80fd|\u53ef\u4ee5|\u5fc5)?(?:(?:\u53bb|\u6765)?(?:\u53eb|\u8bf7|\u627e|\u558a|\u6253|\u62e8|\u901a\u77e5|\u9001)|\u9ebb\u70e6|\u52b3\u70e6)?"
    r"(?:\u4f60\u4eec?|\u60a8|\u5927\u5bb6|\u4efb\u4f55\u4eba?|\u4ec0\u4e48|\u522b\u4eba|\u4eba\u5bb6|\u4ed6\u4eec?|\u5bb6\u4eba)?(?:\u4e2a|\u4e00\u4e2a|\u4e00\u4e0b|\u4e0b)?(?:\u5e2e\u52a9|\u5e2e\u5fd9|\u5e2e\u6211|\u5e2e\u5e2e?|\u6551\u6211|\u6551\u63f4|\u6551\u62a4\u8f66?|\u6025\u6551|\u533b\u751f|\u5927\u592b|\u6551)"
    r"(?:(?:\u53bb|\u6765)?(?:\u53eb|\u8bf7|\u627e|\u558a|\u6253|\u62e8)(?:\u4e2a|\u4e00\u4e2a)?(?:\u6551\u62a4\u8f66?|\u533b\u751f|\u5927\u592b|\u6551\u63f4|\u6025\u6551))?"
)
_NO_HELP = re.compile(_w(_NO_HELP_EN) + "|" + _NO_HELP_ZH)

_OK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [(n, re.compile(p)) for n, p in [
    ("is_fine", _w(r"(?:i'?m|i am|im|i feel|feeling|she'?s|he'?s|they'?re|it'?s|that'?s|we'?re|we are|all|everything(?:'?s| is| was)?|everyone(?:'?s| is)?|everybody(?:'?s| is)?|is|are|was|were|doing|being|still)\s+(?:all |really |very |quite |pretty |just |perfectly |totally |completely |now )?(?:fine|ok|okay|alright|good|well|safe|unharmed|unhurt|uninjured|unscathed)")),
    ("all_good", _w(r"all (?:good|fine|ok|okay|well)")),
    ("no_problem", _w(r"(?:no (?:problem|worries|worry)|not a problem|no need to worry|no need for (?:you to )?worr(?:y|ies)|nothing to worry|nothing (?:happened|serious|wrong)|no harm)")),
    ("dont_worry", _w(r"(?:don't|do not|doesn't|please don't|no need to) (?:you )?(?:worry|fuss)")),
    ("no_fall", _w(r"(?:didn't|did not|never|haven't|hasn't|hadn't|not) (?:\w+ )?(?:fall|fell|fallen|slip|slipped|trip|tripped)")),
    ("under_control", _w(r"(?:under control|safe and (?:well|sound)|just a scratch|it was nothing|it's nothing|its nothing)")),
    ("standalone_ok", r"^(?:(?:yes|yeah|yep|ya|ok|okay|alright|sure|fine|good|really|thanks|thank you|lah|la|ah|then|no)(?:\s+|$))+$"),
    ("zh_ok", r"\u6ca1\u4e8b|\u6ca1\u5173\u7cfb|\u6ca1\u95ee\u9898|\u4e0d\u8981\u7d27|\u4e0d\u788d\u4e8b|\u6ca1\u4ec0\u4e48(?:\u4e8b|\u5927\u4e0d\u4e86|\u95ee\u9898)?|\u4e0d\u7528(?:\u62c5\u5fc3|\u64cd\u5fc3|\u7740\u6025|\u7d27\u5f20|\u7ba1)|\u522b(?:\u62c5\u5fc3|\u64cd\u5fc3|\u7d27\u5f20|\u7740\u6025)|\u4e0d\u5fc5\u62c5\u5fc3|\u4e0d\u8981\u62c5\u5fc3|\u653e\u5fc3|\u6ca1\u6709\u4ec0\u4e48\u597d\u62c5\u5fc3|\u5b89\u7136\u65e0\u6059|\u5e73\u5b89|\u65e0\u4eba\u53d7\u4f24|\u5f97\u5230\u63a7\u5236|\u63a7\u5236\u4f4f\u4e86|(?:\u6837\u6837|\u90fd|\u4e00\u5207\u90fd?)\u4e0d\u9519|\u4e00\u5207(?:\u90fd)?(?:\u6b63\u5e38|\u5f88\u597d|\u90fd\u597d|\u6ca1\u4e8b|\u597d)|\u597d\u597d\u7684|(?:\u6211|\u5979|\u4ed6|\u90fd|\u4e00\u5207|\u8fd8|\u633a|\u5f88|\u86ee|\u771f|\u6700|\u4e5f)(?:\u90fd)?(?:\u5f88|\u633a|\u8fd8|\u86ee|\u771f|\u6700)?\u597d(?:\u7684|\u5566|\u554a|\u5462|\u54e6|$|\uff0c)"),
]]
_STANDALONE_NEEDS = re.compile(_w(r"(?:fine|ok|okay|alright|good)"))
_ZH_QUESTION_END = re.compile(r"[\u5417\u5427\u4e48\u5462]$")


_CLAUSE_BREAK = {"but", "however", "though", "although", "and", "so", "then", "because", "yet"}


def _negated_en(t: str, start: int, window: int = 3) -> bool:
    """A negator within the last `window` words of the same clause (punctuation is already gone, so conjunctions end the scope)."""
    words = t[:start].split()[-window:]
    for i in range(len(words) - 1, -1, -1):
        if words[i] in _CLAUSE_BREAK:
            words = words[i + 1 :]
            break
    return any(_NEG_EN.fullmatch(w) for w in words)


def _negated_zh(t: str, start: int) -> bool:
    before = t[max(0, start - 4) : start]
    return bool(_NEG_ZH_BEFORE.search(before)) or bool(re.search(r"[\u6ca1\u4e0d\u522b\u65e0\u672a]", before[-2:]))


class RuleIntentV2:
    """Lexicon rule with negation scope and 'any help cue wins' (see module docstring)."""

    name = "rule_v2"
    version = "2.0"

    def cues(self, text: str) -> tuple[list[str], list[str]]:
        """(help cues, reassurance cues) in the normalised text. A negated harm word is a reassurance cue."""
        t = normalise_v2(text)
        help_c: list[str] = []
        ok_c: list[str] = []
        if _NO_HELP.search(t):
            ok_c.append("no_help_needed")
        t_help = _NO_HELP.sub(" ", t)  # 'I don't need help' must not fire the plain 'help' cue
        help_c += [n for n, pat in _HELP_PATTERNS if pat.search(t_help)]
        for m in _HARM_EN.finditer(t):
            neg = _negated_en(t, m.start())
            (ok_c if neg else help_c).append(("not_" if neg else "harm:") + m.group(0))
        for m in _HARM_ZH.finditer(t):
            neg = _negated_zh(t, m.start())
            (ok_c if neg else help_c).append(("not_" if neg else "harm:") + m.group(0))
        for name, pat in _OK_PATTERNS:
            if not pat.search(t):
                continue
            if name == "standalone_ok" and not _STANDALONE_NEEDS.search(t):
                continue
            if name == "zh_ok" and _ZH_QUESTION_END.search(t):
                continue  # the Chinese "is everything all right?" asks, it does not answer
            ok_c.append(name)
        return sorted(set(help_c)), sorted(set(ok_c))

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        if not normalise_v2(text or ""):
            return {"intent": "UNCERTAIN", "reason": "empty_transcript"}
        help_c, ok_c = self.cues(text)
        if asr_conf is not None and asr_conf < -1.2 and not (help_c or ok_c):
            return {"intent": "UNCERTAIN", "reason": "low_asr_confidence"}
        if help_c:
            return {"intent": "HELP", "reason": "help_cue" + ("_overrides_reassurance" if ok_c else ""), "cues": help_c}
        if ok_c:
            return {"intent": "REASSURANCE", "reason": "reassurance_cue", "cues": ok_c}
        return {"intent": "OTHER", "reason": "no_cue"}


# ---------------------------------------------------------------- the answer to the spoken prompt
# The inquiry asks "Do you need help? Please say yes or no." The natural answer "yes" is not a help cue in a context-free
# reading (the phrase bank labels bare yes / no OTHER on purpose: A3b scores words, not answers). Read as the answer to
# THIS question, a bare "yes" means "I need help". The step below is applied by the product after the intent model and
# is not part of A3b / A6. It only ever raises OTHER to HELP (the escalation direction); a bare "no" stays OTHER because
# it never counts as reassurance evidence without a reassurance cue.
_YES_EN = re.compile(r"(?:yes|yeah|yep|yea|yup|ya|please|i do|i did)(?: (?:yes|yeah|yep|yea|yup|ya|please|i do|i did))*")
_YES_ZH = re.compile(r"(?:\u662f\u7684?|\u662f\u554a|\u9700\u8981\u7684?|\u6211\u9700\u8981\u7684?|\u8981\u7684?|\u5bf9\u7684?|\u6709\u7684?)+")


def read_as_answer_to_help_prompt(text: str, result: dict[str, Any]) -> dict[str, Any]:
    """Context reading of a reply to "Do you need help?": a bare affirmative (yes / yeah / \u662f\u7684 / \u9700\u8981) is HELP.
    Anything the intent model already decided (HELP, REASSURANCE, UNCERTAIN) is kept as it is."""
    if result.get("intent") != "OTHER":
        return result
    t = normalise_v2(text or "")
    if t and (_YES_EN.fullmatch(t) or _YES_ZH.fullmatch(t)):
        return {"intent": "HELP", "reason": "yes_to_help_prompt", "cues": ["yes_to_prompt"]}
    return result


class RuleIntentV1Simplified:
    """Ablation: the v1 rule applied to Simplified-script text (isolates the effect of the script fix)."""

    name = "rule_v1_t2s"
    version = "1.0+t2s"

    def __init__(self) -> None:
        from homeease.ml.audio.intent import RuleIntent

        self._rule = RuleIntent()

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        return self._rule.predict(to_simplified(text or ""), asr_conf)


class TextIntentLR(SupervisedIntent):
    """The v1 character n-gram classifier, but on `normalise_v2` text (Simplified script, no blanks inside Chinese)."""

    name = "char_ngram_lr_v2"
    version = "2.0"

    def __init__(self, min_p: float = 0.45) -> None:
        super().__init__()
        self.min_p = min_p

    def fit(self, pairs: Sequence[tuple[str, str]]) -> "TextIntentLR":
        self.pipe.fit([normalise_v2(t) for t, _ in pairs], [y for _, y in pairs])
        return self

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        t = normalise_v2(text or "")
        if not t:
            return {"intent": "UNCERTAIN", "reason": "empty_transcript"}
        probs = self.pipe.predict_proba([t])[0]
        k = int(probs.argmax())
        if probs[k] < self.min_p:
            return {"intent": "UNCERTAIN", "reason": "low_classifier_confidence", "p": float(probs[k])}
        return {"intent": str(self.pipe.classes_[k]), "reason": "classifier", "p": float(probs[k])}


class CombinedIntent:
    """Conservative combination of the lexicon rule and a supervised model.

    * either says HELP -> HELP (any help cue wins; the escalation direction is the safe one)
    * both say REASSURANCE -> REASSURANCE (reassurance needs two independent votes)
    * both say OTHER -> OTHER
    * anything else (disagreement, UNCERTAIN) -> UNCERTAIN
    """

    name = "combined"
    version = "1.0"

    def __init__(self, rule: Any, supervised: Any) -> None:
        self.rule, self.sup = rule, supervised

    def predict(self, text: str, asr_conf: float | None = None) -> dict[str, Any]:
        a = self.rule.predict(text, asr_conf)
        b = self.sup.predict(text, asr_conf)
        ia, ib = a["intent"], b["intent"]
        if "HELP" in (ia, ib):
            return {"intent": "HELP", "reason": f"combined_help({ia}/{ib})"}
        if ia == ib == "REASSURANCE":
            return {"intent": "REASSURANCE", "reason": "combined_agree"}
        if ia == ib == "OTHER":
            return {"intent": "OTHER", "reason": "combined_agree"}
        return {"intent": "UNCERTAIN", "reason": f"combined_disagree({ia}/{ib})"}


# ---------------------------------------------------------------- decoding bias vocabulary (hotwords) per language
HOTWORDS = {
    "en": "help, I fell, I can't get up, ambulance, it hurts, I'm okay, I'm fine, don't worry, no problem",
    "zh": "\u6551\u547d, \u5e2e\u5e2e\u6211, \u6211\u6454\u5012\u4e86, \u6211\u8d77\u4e0d\u6765\u4e86, \u6551\u62a4\u8f66, \u597d\u75db, \u6211\u6ca1\u4e8b, \u4e0d\u7528\u62c5\u5fc3, \u6ca1\u5173\u7cfb",
}
HOTWORDS["auto"] = HOTWORDS["en"] + ", " + HOTWORDS["zh"]
PROMPTS = {
    "en": "Safety check. The speaker may say: help, I fell, I can't get up, call an ambulance, I'm okay, I'm fine, don't worry.",
    "zh": "\u4ee5\u4e0b\u662f\u666e\u901a\u8bdd\u7684\u53e5\u5b50\uff1a\u6551\u547d\uff0c\u5e2e\u5e2e\u6211\uff0c\u6211\u6454\u5012\u4e86\uff0c\u6211\u8d77\u4e0d\u6765\u4e86\uff0c\u6211\u6ca1\u4e8b\uff0c\u4e0d\u7528\u62c5\u5fc3\u3002",
}
PROMPTS["auto"] = PROMPTS["en"] + " " + PROMPTS["zh"]
