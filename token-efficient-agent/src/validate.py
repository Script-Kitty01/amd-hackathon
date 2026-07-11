"""Answer validation before shipping (Requirements 2, 5).

Form-level checks used to decide accept-vs-escalate and as a final guard. A
failing answer is escalated (or replaced by the fallback only as a last resort).
Checks are lenient: they reject clearly-bad output (empty, sentinel, wrong shape)
without second-guessing correct content.
"""

from __future__ import annotations

import json
import re

from .categories import Category

_FALLBACK = "Unable to produce an answer."
_SENTIMENT_LABEL = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)
_NER_LABEL = re.compile(r"\b(person|organi[sz]ation|org|location|date)\b", re.I)


def _looks_like_ner(text: str) -> bool:
    # Accept compact JSON with entity-type keys, or explicit type labels.
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        if isinstance(obj, dict) and obj:
            return True
    except (ValueError, json.JSONDecodeError):
        pass
    return bool(_NER_LABEL.search(text))


def is_valid(category: Category, answer: str) -> bool:
    """True if the answer is well-formed enough to ship for its category."""
    text = (answer or "").strip()
    if not text or text == _FALLBACK:
        return False

    if category == Category.NER:
        return _looks_like_ner(text)
    if category == Category.SENTIMENT:
        return bool(_SENTIMENT_LABEL.search(text))

    # Other categories: any non-empty, non-sentinel answer is structurally valid.
    return True
