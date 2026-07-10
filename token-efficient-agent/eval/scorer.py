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
    tier: str | None = None  # which cascade tier answered


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

    @property
    def tier_counts(self) -> dict[str, int]:
        agg: dict[str, int] = defaultdict(int)
        for r in self.records:
            if r.tier:
                agg[r.tier] += 1
        return dict(agg)

    @property
    def local_answer_rate(self) -> float | None:
        """Fraction of tasks answered locally (zero Fireworks tokens)."""
        tiered = [r for r in self.records if r.tier]
        if not tiered:
            return None
        local = sum(1 for r in tiered if r.tier not in ("fireworks", "cloud"))
        return local / len(tiered)

    def summary(self) -> str:
        lines = [
            f"tasks:        {len(self.records)}",
            f"total tokens: {self.total_tokens}",
        ]
        acc = self.accuracy
        if acc is not None:
            lines.append(f"accuracy:     {acc:.1%}")
        rate = self.local_answer_rate
        if rate is not None:
            lines.append(f"local-answer: {rate:.1%}  (0-token tasks)")
        if self.tier_counts:
            lines.append("tier breakdown:")
            for tier, n in sorted(self.tier_counts.items()):
                lines.append(f"  {tier:<14} {n}")
        lines.append("tokens by category:")
        for cat, tok in sorted(self.tokens_by_category.items()):
            lines.append(f"  {cat:<16} {tok}")
        return "\n".join(lines)


def check_match(answer: str, expected: str | None) -> bool | None:
    """Naive reference check for local tuning only."""
    if expected is None:
        return None
    return expected.strip().lower() in answer.strip().lower()
