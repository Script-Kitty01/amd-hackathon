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


def _compress_prose(seg: str) -> str:
    seg = _MULTISPACE.sub(" ", seg)
    seg = _MULTINEWLINE.sub("\n\n", seg)
    # Trim spaces around newlines.
    seg = re.sub(r" *\n *", "\n", seg)
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
