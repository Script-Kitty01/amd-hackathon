"""Accuracy checks and token accounting for local evaluation.

The real grader uses an LLM judge we cannot replicate exactly. Here we support
two lightweight signals for tuning:
  - exact/keyword match against an optional `expected` field
  - token totals per task and per category (from the API usage field)
Use these to find the accuracy-vs-token curve before submitting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class EvalRecord:
    task_id: str
    category: str
    answer: str
    total_tokens: int
    passed: bool | None = None  # None when no reference is available


@dataclass
class EvalReport:
    records: list[EvalRecord] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def tokens_by_category(self) -> dict[str, int]:
        agg: dict[str, int] = defaultdict(int)
        for r in self.records:
            agg[r.category] += r.total_tokens
        return dict(agg)

    @property
    def accuracy(self) -> float | None:
        judged = [r for r in self.records if r.passed is not None]
        if not judged:
            return None
        return sum(1 for r in judged if r.passed) / len(judged)

    def summary(self) -> str:
        lines = [
            f"tasks:        {len(self.records)}",
            f"total tokens: {self.total_tokens}",
        ]
        acc = self.accuracy
        if acc is not None:
            lines.append(f"accuracy:     {acc:.1%}")
        lines.append("tokens by category:")
        for cat, tok in sorted(self.tokens_by_category.items()):
            lines.append(f"  {cat:<16} {tok}")
        return "\n".join(lines)


def check_match(answer: str, expected: str | None) -> bool | None:
    """Naive reference check for local tuning only."""
    if expected is None:
        return None
    return expected.strip().lower() in answer.strip().lower()
