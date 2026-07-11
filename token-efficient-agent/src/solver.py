"""Per-task orchestration: route -> pick tier -> call -> escalate -> fallback.

Tiered model selection (T6), grounded in the scoring rule that token cost is a
raw count and NOT weighted by model size. The only token-relevant reasons to
prefer a smaller model are one-shot success (a failed call still costs tokens,
and a retry doubles them) and output concision. So the policy is:

  - Predict the tier from route() up front. Easy, high-confidence tasks start on
    the cheap preferred model; complex or ambiguous tasks go straight to the
    strong model to avoid paying for a cheap call we expect to fail.
  - Escalate to the strong model only when the primary call errors or returns
    an empty answer.

Guarantees a non-empty answer for every task so results.json is always valid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .categories import Category, escalation_model, select_model
from .config import Config
from .prompts import spec_for
from .router import route

if TYPE_CHECKING:
    from .fireworks_client import FireworksClient


@dataclass
class SolveOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int


_FALLBACK = "Unable to produce an answer."


class Solver:
    def __init__(self, cfg: Config, client: GoogleClient) -> None:
        self._cfg = cfg
        self._client = client

    def _plan_attempts(self, category: Category, complexity: str, ambiguous: bool) -> list[str]:
        """Ordered, de-duplicated list of models to try for one task."""
        models = self._cfg.models
        strong = escalation_model(models)

        if complexity == "complex" or ambiguous:
            primary = strong  # skip a cheap call we expect to fail
        else:
            primary = select_model(category, models)

        attempts = [primary]
        if strong != primary:
            attempts.append(strong)  # escalate on failure
        return attempts

    def solve(self, task_id: str, prompt: str) -> SolveOutcome:
        r = route(prompt)
        spec = spec_for(r.category)
        attempts = self._plan_attempts(r.category, r.complexity, r.ambiguous)

        total_tokens = 0
        for model in attempts:
            try:
                result = self._client.complete(
                    model=model,
                    system=spec.system,
                    user=prompt,
                    max_tokens=spec.max_tokens,
                )
                total_tokens += result.total_tokens
                if result.text:
                    return SolveOutcome(task_id, result.text, r.category, total_tokens)
            except Exception:
                continue  # try the next tier

        return SolveOutcome(task_id, _FALLBACK, r.category, total_tokens)
