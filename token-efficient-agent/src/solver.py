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
    def __init__(self, cfg: Config, client: FireworksClient) -> None:
        self._cfg = cfg
        self._client = client

    def _plan_attempts(self, category: Category, complexity: str, ambiguous: bool) -> list[str]:
        """Model(s) to try for one task.

        Accuracy-first: always send the task to its category-appropriate model
        (Gemma for language, MiniMax for reasoning, Kimi for code). This is a
        SINGLE call in the success case — we do not make a speculative cheap
        call first. A different fallback model is appended only so a first call
        that errors or returns empty still yields an answer (never score zero);
        it costs extra tokens only on failure, not in the normal path.
        """
        models = self._cfg.models
        primary = select_model(category, models)

        attempts = [primary]
        fallback = escalation_model(models)
        if fallback != primary:
            attempts.append(fallback)
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
