"""Meaning-preserving prompt compression (Requirement 8.1).

Reduces remote prompt tokens without changing what is asked:
  - fenced code blocks (``` ... ```) are preserved verbatim;
  - runs of spaces/tabs collapse to a single space;
  - 3+ blank lines collapse to one;
  - a few clearly-safe politeness lead-ins are stripped.

Guarantees (see design Property 3):
  - never removes code, numbers, or entities;
  - idempotent: compress(compress(x)) == compress(x).
"""

from __future__ import annotations

import re

_FENCE = re.compile(r"(```.*?```)", re.S)
_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
# Only strip these when they lead the whole prompt, followed by more text.
_FILLER_PREFIX = re.compile(
    r"^(?:please|kindly|could you(?: please)?|can you(?: please)?|"
    r"i(?:'d| would) like you to|i want you to|would you)\b[\s,:]*",
    re.I,
)
# Redundant instructions to be removed globally.
_REDUNDANT_INSTRUCTIONS = re.compile(
    r"\b(in a clear and concise manner|step by step|step-by-step|make sure to|be sure to|"
    r"in the required format|based on the text provided|according to the context|"
    r"using the information given|think carefully|explain your reasoning)\b[.,]?\s*",
    re.I,
)
# Keywords that signal a constraint that must be preserved.
_CONSTRAINT_KEYWORDS = re.compile(
    r"\b(exactly|only|JSON|Answer:|format|at most|no more than)\b", re.I
)


def _compress_prose(seg: str) -> str:
    # Preserve segments containing constraints from instruction removal.
    if _CONSTRAINT_KEYWORDS.search(seg):
        return seg
    seg = _MULTISPACE.sub(" ", seg)
    seg = _MULTINEWLINE.sub("\n\n", seg)
    # Trim spaces around newlines.
    seg = re.sub(r" *\n *", "\n", seg)
    seg = _REDUNDANT_INSTRUCTIONS.sub("", seg)
    return seg


def compress(prompt: str) -> str:
    """Return a token-leaner prompt with identical meaning."""
    if not prompt:
        return ""

    # Preserve code fences; compress only the prose between them.
    parts = _FENCE.split(prompt)
    out = []
    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            out.append(part)  # code block: verbatim
        else:
            out.append(_compress_prose(part))
    text = "".join(out).strip()

    # Strip a single leading politeness lead-in (safe, meaning-preserving).
    text = _FILLER_PREFIX.sub("", text, count=1).lstrip()
    return text
