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


TEMPLATES: dict[Category, PromptSpec] = {
    # Non-reasoning categories: push for the shortest correct output.
    Category.FACTUAL: PromptSpec(
        system=("Answer in at most 3 short sentences. No preamble, no filler, "
                "do not restate the question."),
        max_tokens=130,
    ),
    Category.SENTIMENT: PromptSpec(
        system=("Classify the sentiment as Positive, Negative, or Neutral. Output "
                "the label, then a reason of at most 6 words."),
        max_tokens=36,
    ),
    Category.SUMMARIZATION: PromptSpec(
        system=("Summarise, obeying any length/format constraint in the task. "
                "Output only the summary — no preamble, no lead-in."),
        max_tokens=90,
    ),
    Category.NER: PromptSpec(
        system=('Output ONLY compact JSON (no whitespace, no prose) with keys '
                '"person","org","location","date", each a list of strings from '
                "the text."),
        max_tokens=120,
    ),
    # Reasoning categories: just enough working for correctness, answer anchored.
    Category.MATH: PromptSpec(
        system=("Solve with brief working, then end with 'Answer: <value>' on "
                "its own line."),
        max_tokens=160,
    ),
    Category.LOGIC: PromptSpec(
        system=("Solve the puzzle so every stated constraint holds. Give at most "
                "two brief reasoning steps, then output the final answer on its "
                "own line as 'Answer: <value>'."),
        max_tokens=150,
    ),
    # Code: keep generous headroom — a truncated answer fails the gate.
    Category.CODE_DEBUG: PromptSpec(
        system=("State the bug in one short line, then output the corrected code "
                "only, in a single code block. No other explanation."),
        max_tokens=300,
    ),
    Category.CODE_GEN: PromptSpec(
        system=("Output only the requested function(s) in one code block — "
                "correct and complete, no explanation, no usage examples."),
        max_tokens=300,
    ),
}


def spec_for(category: Category) -> PromptSpec:
    return TEMPLATES[category]
