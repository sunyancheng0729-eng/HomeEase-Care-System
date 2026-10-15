"""Algorithm 3 (voice interaction and audio monitoring): stand-alone facade over the HomeEase audio modules.

    from algorithm3 import Algorithm3
    a3 = Algorithm3()
    a3.label_sounds(samples, sr)          # daily sound labels (HELP / IMPACT / ABNORMAL / DAILY / UNKNOWN)
    a3.classify_reply(audio=..., text=...)  # reply to "what's wrong? are you OK?"
    a3.detect_spoken_help(samples, sr)    # spoken call for help (Chinese / English)
    from algorithm3.policy import inquiry_outcome, alarm_tag
"""

from algorithm3.api import Algorithm3  # noqa: F401
from algorithm3.policy import alarm_tag, inquiry_outcome, voice_cancel_verdict  # noqa: F401

__all__ = ["Algorithm3", "alarm_tag", "inquiry_outcome", "voice_cancel_verdict"]
