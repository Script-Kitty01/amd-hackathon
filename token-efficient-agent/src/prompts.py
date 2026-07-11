"""Per-category prompt templates and output budgets.

This is the primary tuning surface. Each category maps to a terse system prompt
and a hard max_tokens cap. Keep prompts minimal: every token here is paid on
every call. Only add few-shot examples if a category provably fails without one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .categories import Category


@dataclass(frozen=True)
class PromptSpec:
    system: str
    max_tokens: int


# ACCURACY-FIRST budgets. Caps are generous so answers are never truncated
# before they're complete (a truncated answer fails the judge). max_tokens is a
# ceiling, not a target — well-formed answers stop early, so lean answers still
# cost few tokens. Tighten these only after the accuracy gate is comfortably
# cleared. Prompts request exactly what each category's judge looks for
# (e.g. sentiment must JUSTIFY the label; summaries must obey the constraint).
TEMPLATES: dict[Category, PromptSpec] = {
    Category.FACTUAL: PromptSpec(
        system=("Answer the question accurately and completely, explaining the "
                "concept clearly in 2-4 sentences. No preamble."),
        max_tokens=320,
    ),
    Category.SENTIMENT: PromptSpec(
        system=("Classify the sentiment as Positive, Negative, Neutral, or Mixed. "
                "State the label first, then one sentence of justification. If the "
                "text contains BOTH positive and negative aspects, use Mixed (or "
                "Neutral) and your justification MUST mention both the positive and "
                "the negative aspects."),
        max_tokens=150,
    ),
    Category.SUMMARIZATION: PromptSpec(
        system=("Summarise the text, obeying the EXACT length/format constraint "
                "stated in the task — e.g. 'exactly two sentences', or 'exactly "
                "three bullet points, each under 15 words'. Match the requested "
                "count precisely. Output only the summary — no preamble."),
        max_tokens=220,
    ),
    Category.NER: PromptSpec(
        system=("Extract every named entity and label its type as PERSON, "
                "ORGANIZATION, LOCATION, or DATE. Output ONLY compact JSON with "
                'keys "person","organization","location","date", each a list of '
                "the exact entity strings from the text (empty list if none)."),
        max_tokens=320,
    ),
    # Reasoning categories: thinking models reason before answering, so the cap
    # must fit the full chain PLUS the answer. finalize() extracts the final
    # 'Answer:' line for math/logic, so verbose working doesn't hurt.
    # Reasoning categories: thinking models emit a long chain BEFORE the answer,
    # so the cap must fit the full chain plus the final line or the answer gets
    # truncated (observed on hard puzzles). Generous caps; finalize() extracts the
    # 'Answer:' line for math/logic so verbose working doesn't hurt. The prompt
    # also asks the model to state the answer promptly to reduce runaway chains.
    Category.MATH: PromptSpec(
        system=("Solve step by step but concisely, then end with 'Answer: "
                "<value>' on its own line. State the final numeric value clearly."),
        max_tokens=1536,
    ),
    Category.LOGIC: PromptSpec(
        system=("Solve the puzzle so every stated constraint is satisfied. Reason "
                "concisely through the constraints and, as soon as the solution is "
                "determined, output it on its own line as 'Answer: <value>'."),
        max_tokens=2048,
    ),
    Category.CODE_DEBUG: PromptSpec(
        system=("Identify the bug, then provide the full corrected implementation "
                "in a single code block. Keep any explanation to one short line."),
        max_tokens=1280,
    ),
    Category.CODE_GEN: PromptSpec(
        system=("Write the requested function(s), correct and complete, in a "
                "single code block. Include only what the spec asks for."),
        max_tokens=1280,
    ),
}


def spec_for(category: Category) -> PromptSpec:
    return TEMPLATES[category]
