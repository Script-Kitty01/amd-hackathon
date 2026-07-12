"""Local answer finalization: normalize output on the free local side.

IMPORTANT: local post-processing does NOT reduce the token score — the judging
proxy counts tokens on the Fireworks API call, so anything we trim afterward is
free. finalize() therefore optimizes for CORRECTNESS/readability only, and must
never risk dropping part of a correct answer (e.g. a two-part math result).

What it does:
  - strip reasoning traces (<think>...</think> and friends) that "thinking"
    models (e.g. minimax) emit — otherwise the judge sees a wall of reasoning
    instead of the answer;
  - NER: compact any embedded JSON object into the expected shape;
  - everything else: return the reasoning-stripped text intact (keep all parts,
    working, and justification the judge looks for).

Conservative: if a transform can't be applied cleanly, the cleaned text is
returned unchanged.
"""

from __future__ import annotations

import json
import re

from .categories import Category

# Reasoning/thinking blocks emitted by "thinking" models. Matched case-insensitively.
# Covers: <think>, <thinking>, <reasoning>, <thought>, <scratchpad> (standard),
#         <mm:think> (MiniMax M3), <|think|> (Gemma 4).
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
# A dangling close tag (opener lost to truncation): keep only what follows it.
_DANGLING_CLOSE = re.compile(
    r".*(?:<\s*/\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>|</mm:think>|<\|/think\|>)",
    re.I | re.S,
)
# A dangling open tag (answer never emitted after it): drop from the tag onward.
_DANGLING_OPEN = re.compile(
    r"(?:<\s*(?:think|thinking|reasoning|thought|scratchpad)\s*>|<mm:think>|<\|think\|>).*$",
    re.I | re.S,
)


def strip_reasoning(text: str) -> str:
    """Remove <think>-style reasoning traces, leaving the actual answer."""
    text = _THINK_BLOCK.sub("", text)
    # Handle truncated/asymmetric tags left over after removing balanced blocks.
    if _CLOSE_TAG.search(text):
        text = _DANGLING_CLOSE.sub("", text, count=1)
    if _OPEN_TAG.search(text):
        text = _DANGLING_OPEN.sub("", text, count=1)
    return text.strip()


def finalize(category: Category, answer: str) -> str:
    text = strip_reasoning((answer or "").strip())
    if not text:
        return text

    if category == Category.NER:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return text
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    # All other categories: keep the cleaned answer intact. Trimming saves no
    # tokens and risks dropping part of a correct multi-part answer.
    return text
