"""Local, zero-token category classifier.

Runs entirely on the local machine, so it costs nothing toward the token score.

Scoring model: every category accumulates a score from keyword/regex signals in
the prompt; the highest score wins. This replaces the old first-match-wins order
so conflicting signals resolve by strength, not by check order. The winning
share of total signal gives a confidence, and a lightweight heuristic estimates
task complexity (easy/complex) for downstream tiered model escalation.

Public API:
  - classify(prompt) -> Category                (stable; used across the codebase)
  - route(prompt)    -> RouteResult             (category + confidence + complexity)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .categories import Category

# --- signal patterns -------------------------------------------------------

_CODE_FENCE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|=>|;\s*$", re.MULTILINE)
_DEBUG_HINT = re.compile(r"\b(bug|fix|error|wrong|broken|debug|fails?|exception)\b", re.I)

# Ranking / ordering / deductive-constraint puzzles. Catches phrasings the old
# keyword list missed (e.g. "in some order", "who came first", "Carol beat Bob").
_LOGIC_HINT = re.compile(
    r"\b(puzzle|deduce|arrange|rank(ing|ed)?|in some order|order them|"
    r"who (came|finished|won|sits?|stands?|is (?:in|at|the))|beat|taller|shorter|"
    r"older|younger|faster|slower|(?:to the\s+)?(?:left|right)\s+of|"
    r"in the middle|in a row|next to|seated|sits?\s+(?:in|at|between)|"
    r"if and only|given that|constraint|neither|exactly one|each of)\b",
    re.I,
)

# Simple keyword signals per category. Each hit adds 1.0 to that category.
_KEYWORDS: dict[Category, tuple[str, ...]] = {
    Category.SUMMARIZATION: (
        "summarise", "summarize", "summary", "one sentence", "tl;dr",
        "condense", "in a sentence", "in brief",
    ),
    Category.SENTIMENT: (
        "sentiment", "positive or negative", "tone", "emotion", "how do they feel",
    ),
    Category.NER: (
        "named entit", "extract", "entities", "person, org", "recognition",
    ),
    Category.MATH: (
        # Note: dropped bare "average"/"total" — too common in prose (e.g.
        # "average O(1)"). "average of"/"mean of" (with numbers) are handled by
        # the _MATH_SIGNAL regex below, which avoids that false positive.
        "calculate", "percent", "%", "how many", "how much",
        "sum of", "projection", "compute", "product of", "divided by",
        # Word-problem cues (speed/rate/distance/finance) that carry numbers but
        # no explicit operator keyword.
        "how fast", "how far", "how long", "speed", "per hour", "km/h", "mph",
        "interest", "rate of", "what is its", "on average",
    ),
    Category.FACTUAL: (
        "explain", "what is", "what are", "who ", "how does", "how do",
        "why does", "why do", "define", "definition",
    ),
}

# Explicit numeric-arithmetic signals: aggregates of a list, percent-of, and
# two-operand word arithmetic. Requires "of"/digits so it won't fire on prose
# like "average O(1)".
_MATH_SIGNAL = re.compile(
    r"\b(?:average|mean|sum|product)\s+of\s+[-\d]|"
    r"\bwhat\s+percent(?:age)?\s+of\b|\bsquare\s+root\s+of\b|"
    r"-?\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied by|divided by)\s+-?\d",
    re.I,
)
_MATH_SIGNAL_WEIGHT = 2.0

# Weight for logic matches (each regex hit is a fairly strong signal).
_LOGIC_WEIGHT = 2.0
# Weight for the strong, high-precision code signal.
_CODE_WEIGHT = 3.0

# Tie-break priority: more specific categories win ties over general ones.
_PRIORITY = {
    c: i
    for i, c in enumerate(
        [
            Category.CODE_DEBUG,
            Category.CODE_GEN,
            Category.LOGIC,
            Category.MATH,
            Category.NER,
            Category.SENTIMENT,
            Category.SUMMARIZATION,
            Category.FACTUAL,
        ]
    )
}

# Categories that inherently lean toward multi-step reasoning.
_COMPLEX_CATS = {
    Category.MATH,
    Category.LOGIC,
    Category.CODE_DEBUG,
    Category.CODE_GEN,
}

_COMPLEX_CUES = re.compile(
    r"\b(step by step|explain your reasoning|prove|derive|show your work|"
    r"for each|all of the following|multiple|constraints?)\b",
    re.I,
)
# Clause separators, used as a rough proxy for constraint density.
_CLAUSE = re.compile(r"[.;,]|\band\b", re.I)

_LONG_PROMPT_CHARS = 400
_AMBIGUOUS_CONFIDENCE = 0.5


@dataclass(frozen=True)
class RouteResult:
    category: Category
    confidence: float  # 0..1: winning category's share of total signal
    complexity: str  # "easy" | "complex"
    ambiguous: bool  # True when signal is absent or split across categories


def _score(prompt: str) -> dict[Category, float]:
    """Accumulate a signal score for every category."""
    p = prompt.lower()
    scores: dict[Category, float] = {c: 0.0 for c in Category}

    # Code is a strong, high-precision signal; debug hint splits fix vs. write.
    if _CODE_FENCE.search(prompt) or "function" in p or "code" in p:
        if _DEBUG_HINT.search(prompt):
            scores[Category.CODE_DEBUG] += _CODE_WEIGHT
        else:
            scores[Category.CODE_GEN] += _CODE_WEIGHT

    # Explicit numeric arithmetic (averages, sums, products, percent-of, word ops).
    if _MATH_SIGNAL.search(prompt):
        scores[Category.MATH] += _MATH_SIGNAL_WEIGHT

    # Logic / deductive puzzles.
    scores[Category.LOGIC] += _LOGIC_WEIGHT * len(_LOGIC_HINT.findall(prompt))

    # Keyword-driven categories.
    for cat, words in _KEYWORDS.items():
        scores[cat] += sum(1.0 for w in words if w in p)

    return scores


def _complexity(prompt: str, category: Category) -> str:
    """Rough easy/complex estimate feeding tiered model escalation (T5)."""
    score = 0
    if category in _COMPLEX_CATS:
        score += 1
    if len(prompt) > _LONG_PROMPT_CHARS:
        score += 1
    if _COMPLEX_CUES.search(prompt):
        score += 1
    if len(_CLAUSE.findall(prompt)) >= 3:
        score += 1
    return "complex" if score >= 2 else "easy"


def route(prompt: str) -> RouteResult:
    """Classify a prompt and report confidence + complexity."""
    scores = _score(prompt)
    total = sum(scores.values())

    if total <= 0:
        # No signal at all -> safe default, flagged ambiguous.
        return RouteResult(
            category=Category.FACTUAL,
            confidence=0.0,
            complexity=_complexity(prompt, Category.FACTUAL),
            ambiguous=True,
        )

    top = max(scores, key=lambda c: (scores[c], -_PRIORITY[c]))
    confidence = scores[top] / total
    return RouteResult(
        category=top,
        confidence=confidence,
        complexity=_complexity(prompt, top),
        ambiguous=confidence < _AMBIGUOUS_CONFIDENCE,
    )


def classify(prompt: str) -> Category:
    """Return the most likely category for a task prompt (stable API)."""
    return route(prompt).category
