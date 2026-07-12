"""Local answer finalization (T29): normalize/trim answers on the free local side.

Cheap, deterministic post-processing applied to whatever tier answered, so we
don't rely on the model to emit perfectly-shaped output:
  - strip reasoning traces (<think>...</think> and friends) that "thinking"
    models (e.g. minimax) emit — otherwise the judge sees a wall of reasoning
    instead of the answer;
  - strip common lead-in preamble ("Sure, here is ...");
  - MATH/LOGIC: pull just the value from an `Answer: <value>` line if present,
    else the last meaningful line;
  - NER: compact any embedded JSON object;
  - everything: strip surrounding whitespace/quotes.

Conservative: if a transform can't be applied cleanly, the cleaned text is
returned unchanged.
"""

from __future__ import annotations

import json
import re

from .categories import Category

_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+?)\s*$", re.I | re.M)

# Reasoning/thinking blocks emitted by "thinking" models. Matched case-insensitively.
_THINK_BLOCK = re.compile(
    r"<\s*(think|thinking|reasoning|thought|scratchpad)\s*>.*?<\s*/\s*\1\s*>",
    re.I | re.S,
)
# A dangling close tag (opener lost to truncation): keep only what follows it.
_DANGLING_CLOSE = re.compile(r".*<\s*/\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>",
                             re.I | re.S)
# A dangling open tag (answer never emitted after it): drop from the tag onward.
_DANGLING_OPEN = re.compile(r"<\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>.*$",
                            re.I | re.S)

def strip_reasoning(text: str) -> str:
    """Remove <think>-style reasoning traces, leaving the actual answer."""
    text = _THINK_BLOCK.sub("", text)
    # Handle truncated/asymmetric tags left over after removing balanced blocks.
    if re.search(r"<\s*/\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>", text, re.I):
        text = _DANGLING_CLOSE.sub("", text, count=1)
    if re.search(r"<\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>", text, re.I):
        text = _DANGLING_OPEN.sub("", text, count=1)
    return text.strip()


def finalize(category: Category, answer: str) -> str:
    text = strip_reasoning((answer or "").strip())
    if not text:
        return text

    if category in (Category.MATH, Category.LOGIC):
        matches = _ANSWER_LINE.findall(text)
        if matches:
            return matches[-1].strip().strip("*`").rstrip(".")
        # No explicit Answer: line — fall back to the last non-empty line.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1].rstrip(".") if lines else text

    if category == Category.NER:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return text
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    # Code categories: keep fenced code intact and untouched.
    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        return text

    # Prose categories: return the cleaned text as-is (an intent-based judge
    # tolerates minor preamble; over-stripping risks removing real content).
    return text
