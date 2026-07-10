"""Local answer finalization (T29): normalize/trim answers on the free local side.

Cheap, deterministic post-processing applied to whatever tier answered, so we
don't rely on the model to emit perfectly-shaped output:
  - MATH: pull just the value from an `Answer: <value>` line if present.
  - NER: compact any embedded JSON object.
  - everything: strip surrounding whitespace/fences.

Conservative: if a transform can't be applied cleanly, the original text is
returned unchanged.
"""

from __future__ import annotations

import json
import re

from .categories import Category

_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*([^\n]+)", re.I)


def finalize(category: Category, answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    if category == Category.MATH:
        matches = _ANSWER_LINE.findall(text)
        if matches:
            return matches[-1].strip().rstrip(".")
        return text

    if category == Category.NER:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return text
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    return text
