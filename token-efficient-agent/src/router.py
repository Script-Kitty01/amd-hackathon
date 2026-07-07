"""Local, zero-token category classifier.

Runs entirely on the local machine, so it costs nothing toward the token score.
Starts as fast keyword/regex rules. If launch-day eval shows misclassification,
this is the single place to upgrade to a small trained classifier without
touching the rest of the pipeline.
"""

from __future__ import annotations

import re

from .categories import Category

_CODE_FENCE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|=>|;\s*$", re.MULTILINE)
_DEBUG_HINT = re.compile(r"\b(bug|fix|error|wrong|broken|debug|fails?|exception)\b", re.I)


def classify(prompt: str) -> Category:
    """Return the most likely category for a task prompt."""
    p = prompt.lower()

    # Code-related first: strong, unambiguous signals.
    if _CODE_FENCE.search(prompt) or "function" in p or "code" in p:
        if _DEBUG_HINT.search(prompt):
            return Category.CODE_DEBUG
        return Category.CODE_GEN

    if any(w in p for w in ("summarise", "summarize", "summary", "one sentence", "tl;dr")):
        return Category.SUMMARIZATION

    if any(w in p for w in ("sentiment", "positive or negative", "tone", "emotion")):
        return Category.SENTIMENT

    if any(w in p for w in ("named entit", "extract", "entities", "person, org", "recognition")):
        return Category.NER

    if any(w in p for w in ("calculate", "percent", "%", "how many", "how much",
                            "total", "average", "sum of", "projection")):
        return Category.MATH

    if any(w in p for w in ("puzzle", "if and only", "given that", "deduce",
                            "who is", "arrange", "order them", "constraint")):
        return Category.LOGIC

    return Category.FACTUAL
