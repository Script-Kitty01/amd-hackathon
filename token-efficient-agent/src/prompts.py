"""Per-category prompt templates and output budgets.

TOKEN-EFFICIENT: prompts are the shortest possible while still satisfying the
judge's exact format requirements. Every token here is paid on every remote call.

Key principles:
- System prompt is merged into the user message (no separate system role).
- max_tokens capped tightly — the judge wants correctness, not verbosity.
- Reasoning categories (MATH, LOGIC) get slightly more room for working.
- NO examples in prompts — they add input tokens on every call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .categories import Category


@dataclass(frozen=True)
class PromptSpec:
    system: str
    max_tokens: int
    stop: Optional[list[str]] = field(default=None)
    # Whether this category benefits from reasoning (math/logic = True).
    # Used by the client to set reasoning_effort=none on non-reasoning categories.
    needs_reasoning: bool = False


TEMPLATES: dict[Category, PromptSpec] = {
    Category.FACTUAL: PromptSpec(
        system="Answer accurately; include every requested distinction. Be concise.",
        max_tokens=160,
        needs_reasoning=False,
    ),
    Category.SENTIMENT: PromptSpec(
        system=(
            "Label sentiment and give one sentence of evidence. "
            "Mixed text: mention both positive and negative aspects."
        ),
        max_tokens=80,
        stop=["\n\n"],
        needs_reasoning=False,
    ),
    Category.SUMMARIZATION: PromptSpec(
        system="Obey every requested count, format and word limit exactly. Output the summary only.",
        max_tokens=192,
        needs_reasoning=False,
    ),
    Category.NER: PromptSpec(
        system='Extract all entities. JSON only: {"person":[],"organization":[],"location":[],"date":[]}.',
        max_tokens=192,
        needs_reasoning=False,
        system="Answer the question accurately in 2-4 sentences. No preamble.",
        max_tokens=128,
        stop=["\n\n\n"],
    ),
    Category.SENTIMENT: PromptSpec(
        system=(
            "Classify the sentiment of the text. Respond in this format:\n"
            "Label: <Positive|Negative|Neutral|Mixed>\n"
            "Reason: <one sentence explanation>"
        ),
        max_tokens=64,
    ),
    Category.SUMMARIZATION: PromptSpec(
        system=(
            "Summarize the text, obeying the EXACT length and format constraints "
            "stated in the task (e.g., sentence count, bullet points, word limit). "
            "Output ONLY the summary."
        ),
        max_tokens=192,
    ),
    Category.NER: PromptSpec(
        system=(
            'Extract named entities from the text. Output ONLY a JSON object with '
            'these keys: "person", "organization", "location", "date".'
        ),
        max_tokens=96,
    ),
    # Reasoning categories: compact equations/logic, not prose.
    # The judge accepts "arithmetic shown or implied" so we ask for compact work.
    Category.MATH: PromptSpec(
        system="Solve accurately with compact equations. End with: Answer: <final value>",
        max_tokens=256,
        needs_reasoning=True,
    ),
    Category.LOGIC: PromptSpec(
        system="Satisfy all constraints. Reason compactly. End with: Answer: <answer>",
        max_tokens=384,
        needs_reasoning=True,
    ),
    Category.CODE_DEBUG: PromptSpec(
        system="Return complete corrected code in one fenced block, then: Bug: <brief cause>",
        max_tokens=512,
        needs_reasoning=False,
    ),
    Category.CODE_GEN: PromptSpec(
        system="Return complete requested code only in one fenced block.",
        max_tokens=512,
        needs_reasoning=False,
        system=(
            "Solve the math problem. Use compact equations. End with:\n"
            "Answer: <value>"
        ),
        max_tokens=256,
    ),
    Category.LOGIC: PromptSpec(
        system=(
            "Solve the logic problem. Use concise reasoning. End with:\n"
            "Answer: <answer>"
        ),
        max_tokens=384,
    ),
    Category.CODE_DEBUG: PromptSpec(
        system=(
            "Identify the bug and provide the corrected code in a fenced code block. "
            "After the code block, add one line starting with 'Bug:' describing the fix.\n\n"
            "Format:\n"
            "```python\n<corrected code>\n```\n"
            "Bug: <one-line description of the bug>"
        ),
        max_tokens=512,
    ),
    Category.CODE_GEN: PromptSpec(
        system=(
            "Write the requested function(s). Output ONLY the code in a single "
            "fenced code block."
        ),
        max_tokens=512,
    ),
}


def spec_for(category: Category) -> PromptSpec:
    return TEMPLATES[category]
