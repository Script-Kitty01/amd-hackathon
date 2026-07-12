"""Local answer finalization: normalize output on the free local side.

IMPORTANT: local post-processing does NOT reduce the token score — the judging
proxy counts tokens on the Fireworks API call, so anything we trim afterward is
free. finalize() therefore optimizes for CORRECTNESS/readability only.

What it does:
  - strip reasoning traces (<think>...</think> and friends)
  - Math/Logic: extract the "Answer: <value>" line so the judge sees a clean answer
  - NER: compact any embedded JSON object into the expected shape
  - Code: preserve code blocks intact
  - Sentiment: ensure the label is clearly visible
  - Everything else: return cleaned text intact
"""

from __future__ import annotations

import json
import re

from .categories import Category

# Reasoning/thinking blocks emitted by "thinking" models.
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

# Pattern for "Answer: <value>" lines (math/logic)
_ANSWER_LINE = re.compile(
    r"^\s*(?:answer|final\s+answer)\s*[:=]\s*(.+)",
    re.MULTILINE | re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Remove <think>-style reasoning traces, leaving the actual answer."""
    text = _THINK_BLOCK.sub("", text)
    if _CLOSE_TAG.search(text):
        text = _DANGLING_CLOSE.sub("", text, count=1)
    if _OPEN_TAG.search(text):
        text = _DANGLING_OPEN.sub("", text, count=1)
    return text.strip()


def _extract_math_answer(text: str) -> str:
    """For math/logic: extract the Answer: line if present, else return full text.
    
    The judge evaluates the FINAL answer, so we extract it cleanly. If no
    explicit answer line, return the full working (judge can handle it).
    """
    # Look for the LAST "Answer:" line (models sometimes have intermediate answers)
    matches = _ANSWER_LINE.findall(text)
    if matches:
        answer = matches[-1].strip()
        # Clean up common suffixes models add after the answer
        answer = re.sub(r"\s*[\.\,]?\s*$", "", answer)
        return answer
    return text


def _compact_ner(text: str) -> str:
    """Extract and compact NER JSON output."""
    # Try to find a JSON object in the text
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        if isinstance(obj, dict):
            # Normalize keys to lowercase
            normalized = {}
            for k, v in obj.items():
                key = k.lower().strip()
                # Map common variants
                if key in ("org", "orgs", "organizations", "organisation", "organisations"):
                    key = "organization"
                elif key in ("persons", "people"):
                    key = "person"
                elif key in ("locations", "loc", "place", "places", "gpe"):
                    key = "location"
                elif key in ("dates", "time"):
                    key = "date"
                normalized[key] = v if isinstance(v, list) else [v]
            return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    except (ValueError, json.JSONDecodeError):
        pass
    return text


def _clean_code(text: str) -> str:
    """Preserve code blocks, strip any surrounding prose."""
    # If there's a fenced code block, extract it
    fence_match = re.search(r"```(?:\w*)\n?(.*?)```", text, re.S)
    if fence_match:
        code = fence_match.group(1).strip()
        # Also include any "Bug:" line after the fence
        bug_match = re.search(r"```\s*\n*(Bug:.*?)$", text, re.MULTILINE)
        if bug_match:
            return f"```python\n{code}\n```\n{bug_match.group(1)}"
        return f"```python\n{code}\n```"
    return text


def finalize_answer(category: Category, answer: str) -> str:
    """Finalize an answer for submission: strip reasoning, extract clean output."""
    text = strip_reasoning((answer or "").strip())
    if not text:
        return text

    if category in (Category.MATH, Category.LOGIC):
        return _extract_math_answer(text)

    if category == Category.NER:
        return _compact_ner(text)

    if category in (Category.CODE_DEBUG, Category.CODE_GEN):
        return _clean_code(text)

    # Sentiment, Factual, Summarization: return cleaned text intact
    return text
