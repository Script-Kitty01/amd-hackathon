from enum import Enum
from dataclasses import dataclass
from .categories import Category

@dataclass
class PromptSpec:
    system: str
    max_tokens: int
    stop: list[str] | None = None

TEMPLATES: dict[Category, PromptSpec] = {
    Category.FACTUAL: PromptSpec(
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
    Category.MATH: PromptSpec(
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