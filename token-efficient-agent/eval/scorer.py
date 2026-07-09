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

    @property
    def accuracy_by_category(self) -> dict[str, float | None]:
        acc_dict: dict[str, list[bool]] = defaultdict(list)
        for r in self.records:
            if r.passed is not None:
                acc_dict[r.category].append(r.passed)
        
        result = {}
        for cat, results in acc_dict.items():
            if results:
                result[cat] = sum(1 for p in results if p) / len(results)
            else:
                result[cat] = None
        return result

    def summary(self) -> str:
        lines = [
            f"tasks:        {len(self.records)}",
            f"total tokens: {self.total_tokens}",
        ]
        acc = self.accuracy
        if acc is not None:
            lines.append(f"accuracy:     {acc:.1%}")
            
        lines.append("\naccuracy by category:")
        acc_by_cat = self.accuracy_by_category
        # Get all unique categories from records to ensure we list them all, even with 0% or None
        all_categories = sorted(list(set(r.category for r in self.records)))
        for cat in all_categories:
            cat_acc = acc_by_cat.get(cat)
            acc_str = f"{cat_acc:.1%}" if cat_acc is not None else "N/A"
            lines.append(f"  {cat:<16} {acc_str}")

        lines.append("\ntokens by category:")
        for cat, tok in sorted(self.tokens_by_category.items()):
            lines.append(f"  {cat:<16} {tok}")
        return "\n".join(lines)


def check_match(answer: str, expected: str | None, category: str) -> bool | None:
    """Naive reference check for local tuning only. Enhanced with category specific heuristics."""
    if expected is None:
        return None
        
    ans_lower = answer.strip().lower()
    exp_lower = expected.strip().lower()
    
    # Basic substring match
    passed = exp_lower in ans_lower
    
    # Category specific shape checks (simulating LLM Judge intent checking)
    if category == "ner":
        # NER must output JSON. Let's do a basic JSON parse check.
        import json
        try:
            parsed = json.loads(answer)
            # Basic key check
            if not all(k in parsed for k in ["person", "org", "location", "date"]):
                 return False # Failed format constraint
        except json.JSONDecodeError:
            return False # Not valid JSON
            
    elif category == "sentiment":
         # Must start with Positive, Negative, or Neutral
         if not any(ans_lower.startswith(s) for s in ["positive", "negative", "neutral"]):
             return False
             
    elif category in ["code_gen", "code_debug"]:
        # Should ideally be just a code block. We check for basic python keywords.
        if "def " not in answer and "class " not in answer and "import " not in answer:
             # Very weak check, but better than nothing for a mock scorer
             pass
             
    return passed
