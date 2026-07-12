"""Per-category prompt templates and output budgets.

ACCURACY-FIRST: prompts are tuned to maximize judge pass rate. Each category
has a system prompt that tells the model exactly what the judge expects, and a
generous-enough max_tokens so answers never get truncated. Token savings come
from model choice (Gemma = non-reasoning, dense tokenizer) not from starving
the model of output space.

Stop sequences are used to cut generation at the right point without truncation.
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


# ACCURACY-FIRST budgets. These are generous so answers are never truncated.
# The judge evaluates correctness + completeness, so we'd rather have a verbose
# correct answer than a truncated one. Token savings come from Gemma-first
# (non-reasoning, dense tokenizer) not from tight caps.
TEMPLATES: dict[Category, PromptSpec] = {
    Category.FACTUAL: PromptSpec(
        system=(
            "You are a knowledgeable assistant. Answer the question accurately "
            "and completely in 2-4 sentences. Explain the concept clearly. "
            "No preamble, no filler, just the answer."
        ),
        max_tokens=256,
        stop=["\n\n\n"],
    ),
    Category.SENTIMENT: PromptSpec(
        system=(
            "Classify the sentiment of the given text. Your response MUST follow "
            "this exact format:\n"
            "Label: <Positive|Negative|Neutral|Mixed>\n"
            "Reason: <one sentence explaining why, mentioning specific evidence>\n\n"
            "IMPORTANT RULES:\n"
            "- If the text contains BOTH positive AND negative aspects, use 'Mixed' "
            "and your reason MUST mention BOTH the positive and the negative aspects.\n"
            "- If the text is purely positive, use 'Positive'.\n"
            "- If the text is purely negative, use 'Negative'.\n"
            "- If the text is neither, use 'Neutral'.\n"
            "- Always cite specific words/phrases from the text as evidence."
        ),
        max_tokens=120,
    ),
    Category.SUMMARIZATION: PromptSpec(
        system=(
            "Summarize the text, obeying the EXACT length and format constraint "
            "stated in the task. For example:\n"
            "- 'in one sentence' = exactly 1 sentence\n"
            "- 'in exactly two sentences' = exactly 2 sentences\n"
            "- 'exactly three bullet points' = exactly 3 bullet points\n"
            "- If a word limit is given, stay under it\n\n"
            "Match the requested count PRECISELY. Output ONLY the summary. "
            "No preamble, no 'Here is the summary:', just the summary itself."
        ),
        max_tokens=256,
    ),
    Category.NER: PromptSpec(
        system=(
            "Extract ALL named entities from the text and categorize each as "
            "PERSON, ORGANIZATION, LOCATION, or DATE.\n\n"
            "Output ONLY a JSON object with these exact keys: "
            '"person", "organization", "location", "date" — each mapping to a '
            "list of entity strings found in the text. Use empty lists [] for "
            "categories with no entities.\n\n"
            "Example output format:\n"
            '{"person":["John Smith"],"organization":["Google"],'
            '"location":["New York"],"date":["March 2023"]}\n\n'
            "Be thorough — extract EVERY entity. Include full names and dates "
            "exactly as they appear in the text."
        ),
        max_tokens=256,
    ),
    Category.MATH: PromptSpec(
        system=(
            "Solve the math problem step by step. Show your work clearly but "
            "concisely. After your working, you MUST end with a final line in "
            "exactly this format:\n\n"
            "Answer: <final numeric value>\n\n"
            "The Answer line must contain ONLY the final number (with units or "
            "currency symbol if appropriate). No extra text after it."
        ),
        max_tokens=768,
    ),
    Category.LOGIC: PromptSpec(
        system=(
            "Solve the logic puzzle or reasoning problem. Think through it "
            "carefully, checking that every stated constraint is satisfied. "
            "Show your reasoning concisely, then state your final answer on "
            "its own line in exactly this format:\n\n"
            "Answer: <value>\n\n"
            "Make sure your answer satisfies ALL constraints in the problem."
        ),
        max_tokens=1024,
    ),
    Category.CODE_DEBUG: PromptSpec(
        system=(
            "You are a code debugging expert. Identify the bug in the code, "
            "then provide the COMPLETE corrected implementation in a single "
            "fenced code block. After the code block, add exactly one line "
            "starting with 'Bug:' that names what was wrong.\n\n"
            "Format:\n"
            "```python\n<corrected code>\n```\n"
            "Bug: <one-line description of the bug>"
        ),
        max_tokens=640,
    ),
    Category.CODE_GEN: PromptSpec(
        system=(
            "Write the requested function(s). Output ONLY the code in a single "
            "fenced code block. The code must be correct, complete, and handle "
            "edge cases. No prose, no examples, no explanation — just the code.\n\n"
            "Format:\n```python\n<your code>\n```"
        ),
        max_tokens=640,
    ),
}


def spec_for(category: Category) -> PromptSpec:
    return TEMPLATES[category]
