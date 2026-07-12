"""Local answer finalization: normalize output on the free local side.

Post-processing costs zero tokens (runs after the API call).

What it does:
  1. Strip tagged reasoning blocks (<think>, <mm:think>, <|think|>)
  2. Strip UNTAGGED prose reasoning spill (Gemma-style "Let me think..." leakage)
     using a score-based detector: strip when confident it's reasoning, not answer
  3. Extract clean Answer: line for math/logic
  4. Compact NER JSON
  5. Strip Answer:/Final answer: prefixes, outer quotes, trailing periods on
     short answers, normalize whitespace
"""

from __future__ import annotations

import json
import re

from .categories import Category

# ── 1. Tagged reasoning blocks ──────────────────────────────────────────────
# Covers: <think>, <thinking>, <reasoning>, <thought>, <scratchpad> (standard)
#         <mm:think> (MiniMax M3)
#         <|think|> (Gemma 4)
_THINK_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|thought|scratchpad)\s*>.*?<\s*/\s*\1\s*>|"
    r"<mm:think>.*?</mm:think>|"
    r"<\|think\|>.*?<\|/think\|>",
    re.I | re.S,
)
_CLOSE_TAG = re.compile(
    r"<\s*/\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>|"
    r"</mm:think>|<\|/think\|>",
    re.I,
)
_OPEN_TAG = re.compile(
    r"<\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>|"
    r"<mm:think>|<\|think\|>",
    re.I,
)
_DANGLING_CLOSE = re.compile(
    r".*(?:<\s*/\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>|</mm:think>|<\|/think\|>)",
    re.I | re.S,
)
_DANGLING_OPEN = re.compile(
    r"(?:<\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>|<mm:think>|<\|think\|>).*$",
    re.I | re.S,
)


def strip_reasoning(text: str) -> str:
    """Remove tagged <think>-style reasoning traces."""
    text = _THINK_BLOCK.sub("", text)
    if _CLOSE_TAG.search(text):
        text = _DANGLING_CLOSE.sub("", text, count=1)
    if _OPEN_TAG.search(text):
        text = _DANGLING_OPEN.sub("", text, count=1)
    return text.strip()


# ── 2. Untagged prose reasoning spill (Gemma-style) ────────────────────────
# Score-based: require ≥3 to strip, not just any single cue. This avoids
# falsely stripping valid answers like "Let me explain..." or "First, ...".

_SPILL_CUES = [
    # Score-2 cues: very strong signal this is meta-reasoning, not the answer
    (2, re.compile(r"^the user (?:wants|is asking|needs)", re.I)),
    (2, re.compile(r"^we need (?:to|answer)", re.I)),
    (2, re.compile(r"^let me (?:think|analyze|consider|work)", re.I)),
    (2, re.compile(r"^thought\s*\n", re.I)),
    # Score-1 cues: weaker signal, need combination
    (1, re.compile(r"^(?:let's|let us) (?:solve|work|figure|analyze|think)", re.I)),
    (1, re.compile(r"^to solve this", re.I)),
    (1, re.compile(r"^first,?\s+(?:i need|we need|let's|let me)", re.I)),
    (1, re.compile(r"^(?:looking at|analyzing) (?:this|the)", re.I)),
]


def _prose_spill_score(text: str) -> int:
    """Return a spill score for the first meaningful line of text."""
    first_line = text.lstrip().split("\n")[0].strip()
    score = 0
    for weight, pattern in _SPILL_CUES:
        if pattern.match(first_line):
            score += weight
    return score


def strip_prose_spill(text: str, category: Category) -> str:
    """Strip untagged reasoning spill from the beginning of the answer.

    Only strips when score >= 3. After stripping, returns what remains — if
    nothing useful remains, returns the original (don't make it worse).
    For reasoning categories (math/logic) we WANT the working, so skip.
    """
    if category in (Category.MATH, Category.LOGIC):
        return text  # reasoning categories want visible working
    if _prose_spill_score(text) < 3:
        return text
    # Find the first line that looks like real answer content
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Re-score from this line — if it still looks like spill, keep going
        remaining = "\n".join(lines[i:])
        if _prose_spill_score(remaining) < 3:
            result = remaining.strip()
            return result if result else text
    return text


# ── 3. Math/Logic answer extraction ────────────────────────────────────────
_ANSWER_LINE = re.compile(
    r"^\s*(?:\*{0,2})(?:final\s+)?answer\s*(?:\*{0,2})\s*[:=]\s*(.+)",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_math_answer(text: str) -> str:
    """Extract the last Answer: <value> line if present."""
    matches = _ANSWER_LINE.findall(text)
    if matches:
        answer = matches[-1].strip().strip("*`").rstrip(".")
        # Remove trailing sentence if it snuck past (e.g. "1672 units")
        # Keep currency/units that are part of the answer
        return answer
    # No explicit Answer: line — return last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


# ── 4. NER JSON compaction ──────────────────────────────────────────────────
_KEY_MAP = {
    "org": "organization", "orgs": "organization",
    "organizations": "organization", "organisation": "organization",
    "organisations": "organization",
    "persons": "person", "people": "person",
    "locations": "location", "loc": "location",
    "place": "location", "places": "location", "gpe": "location",
    "dates": "date", "time": "date",
}


def _compact_ner(text: str) -> str:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        if isinstance(obj, dict):
            normalized: dict[str, list] = {}
            for k, v in obj.items():
                key = _KEY_MAP.get(k.lower().strip(), k.lower().strip())
                normalized[key] = v if isinstance(v, list) else [str(v)]
            return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, json.JSONDecodeError):
        pass
    return text


# ── 5. General answer cleaning ──────────────────────────────────────────────
_ANSWER_PREFIX = re.compile(
    r"^\s*(?:answer|final\s+answer|final|output|result|response)\s*[:=]\s*",
    re.IGNORECASE,
)
_SHORT_TRAILING_PERIOD = re.compile(r"\.\s*$")


def _clean_general(text: str) -> str:
    """Strip common wrapper noise that adds no information."""
    # Strip "Answer: " prefix the model emits before its answer
    text = _ANSWER_PREFIX.sub("", text, count=1).strip()
    # Strip outer quotes on short single-line answers
    if "\n" not in text and len(text) < 120:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "`"):
            text = text[1:-1].strip()
    # Strip trailing period on short answers (≤8 words) — "Paris." → "Paris"
    word_count = len(text.split())
    if word_count <= 8 and text.endswith("."):
        text = text[:-1].strip()
    # Normalize unicode spaces and excess whitespace
    text = text.replace("\u00a0", " ").replace("\u2009", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


# ── Public API ──────────────────────────────────────────────────────────────

def finalize(category: Category, answer: str) -> str:
    """Full finalization pipeline. Free — runs after the API call."""
    text = strip_reasoning((answer or "").strip())
    if not text:
        return text

    text = strip_prose_spill(text, category)
    if not text:
        return answer.strip()  # fallback to original if we stripped too much

    if category in (Category.MATH, Category.LOGIC):
        return _extract_math_answer(text)

    if category == Category.NER:
        return _compact_ner(text)

    # Code: preserve as-is (fenced blocks must stay intact)
    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        return text

    # Prose categories: general cleaning
    return _clean_general(text)
