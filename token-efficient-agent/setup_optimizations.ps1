$profilerContent = @'
"""Lightweight token profiler for evaluation runs."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CallStats:
    task_id: str
    category: str
    model: str
    prompt: int
    completion: int
    total: int


class TokenProfiler:
    def __init__(self) -> None:
        self.is_active = os.environ.get("DEBUG_TOKEN_STATS", "").lower() == "true"
        self.calls: list[CallStats] = []
        self.retries = 0

    def record(self, **kwargs) -> None:
        if not self.is_active:
            return
        self.calls.append(CallStats(**kwargs))

    def record_retry(self) -> None:
        if not self.is_active:
            return
        self.retries += 1

    def print_report(self) -> None:
        if not self.is_active or not self.calls:
            return

        cat_totals: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        model_totals: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        
        total_prompt = 0
        total_completion = 0

        for call in self.calls:
            cat_totals[call.category]["prompt"] += call.prompt
            cat_totals[call.category]["completion"] += call.completion
            cat_totals[call.category]["total"] += call.total
            cat_totals[call.category]["count"] += 1

            model_totals[call.model]["prompt"] += call.prompt
            model_totals[call.model]["completion"] += call.completion
            model_totals[call.model]["total"] += call.total
            model_totals[call.model]["count"] += 1
            
            total_prompt += call.prompt
            total_completion += call.completion

        print("\n--- Token Profiler Report ---")

        print("\nCategory Totals:")
        for cat, totals in sorted(cat_totals.items()):
            print(
                f"  {cat:<16} "
                f"prompt={totals['prompt']:<6} "
                f"completion={totals['completion']:<6} "
                f"total={totals['total']:<6}"
            )

        print("\nModel Totals:")
        for model, totals in sorted(model_totals.items()):
            print(
                f"  {model:<48} "
                f"prompt={totals['prompt']:<6} "
                f"completion={totals['completion']:<6} "
                f"total={totals['total']:<6}"
            )

        print(f"\nRetry count: {self.retries}")

        print("\nAverage Tokens per Category:")
        for cat, totals in sorted(cat_totals.items()):
            count = totals['count']
            avg_prompt = totals['prompt'] / count
            avg_completion = totals['completion'] / count
            print(
                f"  {cat:<16} "
                f"prompt={avg_prompt:<6.1f} "
                f"completion={avg_completion:<6.1f}"
            )
        
        if self.calls:
            print(f"\nAverage prompt tokens: {total_prompt / len(self.calls):.1f}")
            print(f"Average completion tokens: {total_completion / len(self.calls):.1f}")
        
        print("---------------------------\n")
'@

# Save to disk using standard UTF-8 encoding
[System.IO.File]::WriteAllText("eval/profiler.py", $profilerContent, [System.Text.Encoding]::UTF8)
Write-Host "Deployed: eval/profiler.py" -ForegroundColor Green


# 2. Write src/prompts.py
$promptsContent = @'
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
'@

# Save to disk using standard UTF-8 encoding
[System.IO.File]::WriteAllText("src/prompts.py", $promptsContent, [System.Text.Encoding]::UTF8)
Write-Host "Deployed: src/prompts.py" -ForegroundColor Green
write-host "All files successfully written. You can now execute: git add . ; git commit -m 'feat: complete token performance optimization pipeline'" -ForegroundColor Cyan