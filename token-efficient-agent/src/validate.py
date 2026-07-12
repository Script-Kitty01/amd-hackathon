"""Answer validation before shipping (Requirements 2, 5).

Two levels of validation:
  - is_valid(): structural check — rejects clearly-bad output (empty, sentinel,
    wrong shape). Used as accept/reject gate.
  - needs_escalation(): quality heuristic — flags answers that are technically
    valid but likely too weak to pass the judge. Triggers escalation to a
    stronger model.
"""

from __future__ import annotations

import json
import re

from .categories import Category

_FALLBACK = "Unable to produce an answer."
_SENTIMENT_LABEL = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.I)
_NER_LABEL = re.compile(r"\b(person|organi[sz]ation|org|location|date)\b", re.I)
_HAS_CODE = re.compile(r"```|def\s+\w+|function\s+\w+|class\s+\w+", re.I)
_HAS_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")


def _looks_like_ner(text: str) -> bool:
    """Accept compact JSON with entity-type keys, or explicit type labels."""
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
    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        # Must contain something code-like
        return bool(_HAS_CODE.search(text)) or "return" in text.lower()
    if category == Category.MATH:
        # Must contain at least one number
        return bool(_HAS_NUMBER.search(text))

    # Factual, Summarization, Logic: any non-empty answer is valid
    return True


def needs_escalation(category: Category, answer: str) -> bool:
    """True if the answer is structurally valid but likely too weak for the judge.
    
    This triggers trying a stronger model. Conservative: only escalate when we're
    fairly confident the answer will fail.
    """
    text = (answer or "").strip()
    if not text:
        return True

    # Math: if the answer is just "I cannot" or similar cop-out
    if category == Category.MATH:
        lower = text.lower()
        if any(w in lower for w in ("cannot", "can't", "unable", "insufficient", "not enough")):
            return True
        # Very short math answers without a number might be wrong
        if len(text) < 3 and not _HAS_NUMBER.search(text):
            return True

    # Logic: same cop-out detection
    if category == Category.LOGIC:
        lower = text.lower()
        if any(w in lower for w in ("cannot", "can't", "unable", "insufficient")):
            return True

    # Code: if no actual code/function definition present
    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        if not _HAS_CODE.search(text) and "return" not in text.lower():
            return True

    # Sentiment: if label is missing (the judge wants a clear label)
    if category == Category.SENTIMENT:
        if not _SENTIMENT_LABEL.search(text):
            return True

    return False
