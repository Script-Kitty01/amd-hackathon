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
import json
import urllib.request
from dataclasses import dataclass

from .categories import Category

# --- signal patterns -------------------------------------------------------

_CODE_FENCE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|=>|;\s*$", re.MULTILINE)
_DEBUG_HINT = re.compile(r"\b(bug|fix|error|wrong|broken|debug|fails?|exception)\b", re.I)

# Ranking / ordering / deductive-constraint puzzles. Catches phrasings the old
# keyword list missed (e.g. "in some order", "who came first", "Carol beat Bob").
_LOGIC_HINT = re.compile(
    r"\b(puzzle|deduce|arrange|rank(ing|ed)?|in some order|order them|"
    r"who (came|finished|won|sits?|stands?)|beat|taller|shorter|older|younger|"
    r"faster|slower|to the (left|right) of|if and only|given that|"
    r"constraint|neither|exactly one|each of)\b",
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


def _classify_via_local_llm(prompt: str) -> Category | None:
    """Tier 2 Fallback: Run local Qwen to classify ambiguous text for 0 cloud tokens."""
    url = "http://localhost:11434/api/generate"
    
    system_instruction = (
        "Classify this user prompt into exactly one category: "
        "CODE_DEBUG, CODE_GEN, SUMMARIZATION, SENTIMENT, NER, MATH, LOGIC, FACTUAL. "
        "Output ONLY the single word corresponding to the category."
    )
    
    payload = {
        "model": "qwen3.5:4b",
        "prompt": f"{system_instruction}\n\nPrompt to classify: {prompt}\n\nCategory:",
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        # Bumping timeout to 15 seconds to allow the local model to wake up on cold start
        with urllib.request.urlopen(req, timeout=1000) as response:
            res = json.loads(response.read().decode("utf-8"))
            ans = res.get("response", "").strip().upper()
            
            # DEBUG TRACKER: Let's see what text the local model is throwing back
            print(f"   [Debug] Local LLM raw response: '{ans}'")
            
            for cat in Category:
                if cat.name in ans:
                    return cat
    except Exception as e:
        # DEBUG TRACKER: Let's catch if it's a connection drop or a formatting error
        print(f"   [Debug] Local LLM execution failed: {e}")
        pass
        
    return None


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
        # ==========================================
        # TIER 2: Local LLM Verification Shield (Disabled for now)
        # ==========================================
        # If keywords miss, let local Qwen figure it out before giving up
        # local_choice = _classify_via_local_llm(prompt)
        # if local_choice is not None:
        #     return RouteResult(
        #         category=local_choice,
        #         confidence=1.0,
        #         complexity=_complexity(prompt, local_choice),
        #         ambiguous=False,
        #     )
        
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


if __name__ == "__main__":
    # Sample prompts to test our hybrid routing shield
    test_prompts = [
        "What is the derivative of 3x^2 + 5x - 9?", 
        "I absolutely love how smoothly this local setup works!", 
        "Can you rewrite this loop to use list comprehension?", 
        "If box A is inside box B, and box B is inside box C, is box A inside box C?",
        "Extract the names of the people and organizations mentioned in this paragraph."
    ]
    
    print("🚀 Running Router Standalone Test...\n" + "="*40)
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: \"{prompt}\"")
        detected_category = classify(prompt)
        print(f"👉 Result: {detected_category}\n" + "-"*40)
