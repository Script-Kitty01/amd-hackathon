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
    Category.FACTUAL: PromptSpec(
        system="Provide a direct, factual answer in 1-2 sentences. No intro, no filler.",
        max_tokens=60,
    ),
    Category.MATH: PromptSpec(
        system=("Solve the problem. Show minimal working, then output the final "
                "result on a new line as 'Answer: <value>'."),
        max_tokens=220,
    ),
    Category.SENTIMENT: PromptSpec(
        system=("Classify the sentiment as Positive, Negative, or Neutral. "
                "Reply with the label followed by a brief (max 8 words) reason."),
        max_tokens=40,
    ),
    Category.SUMMARIZATION: PromptSpec(
        system=("Summarise the text obeying any length/format constraint stated "
                "in the task. Output only the summary, nothing else."),
        max_tokens=100,
    ),
    Category.NER: PromptSpec(
        system=('Extract named entities. Output ONLY compact JSON with keys '
                '"person", "org", "location", "date"; each a list of strings.'),
        max_tokens=140,
    ),
    Category.CODE_DEBUG: PromptSpec(
        system=("Identify the bug in one short line, then output the corrected "
                "code only, in a single code block. No extra explanation."),
        max_tokens=320,
    ),
    Category.LOGIC: PromptSpec(
        system=("Reason step by step but concisely, satisfying every stated "
                "constraint. End with the final answer on its own line."),
        max_tokens=260,
    ),
    Category.CODE_GEN: PromptSpec(
        system=("Write only the requested function(s), correct and complete, in "
                "a single code block. No explanation, no usage examples."),
        max_tokens=320,
    ),
}


def spec_for(category: Category) -> PromptSpec:
    return TEMPLATES[category]
