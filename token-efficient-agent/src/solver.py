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
from .compress import compress
from .config import Config
from .finalize import strip_reasoning
from .prompts import spec_for
from .router import route
from .validate import is_valid

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
        """Ordered models to try for one task.

        Accuracy-first: the category-appropriate model goes first (Gemma for
        language, MiniMax for reasoning, Kimi for code) — a SINGLE call in the
        success case. The remaining allowed models follow as backstops so a task
        never ends up empty just because one or two models errored, timed out,
        or returned malformed output. Backstops cost extra tokens ONLY on the
        failure path (a valid primary answer returns immediately); on the
        accuracy gate that trade is always worth it. The strongest general
        fallback is ordered ahead of near-duplicates of the primary.
        """
        models = self._cfg.models
        primary = select_model(category, models)
        strong = escalation_model(models)

        # Primary + ONE strong fallback. Two attempts is the accuracy-vs-latency
        # sweet spot: it recovers from a single failed/empty call without turning
        # a slow task into 5 sequential remote calls, which on the 2-vCPU / 10-min
        # grading box would risk timing out and leaving later tasks empty.
        attempts = [primary]
        if strong != primary:
            attempts.append(strong)
        return attempts

    def solve(self, task_id: str, prompt: str) -> SolveOutcome:
        r = route(prompt)
        spec = spec_for(r.category)
        attempts = self._plan_attempts(r.category, r.complexity, r.ambiguous)
        # Compress the prompt for the remote call only (routing/caching used the
        # original). Meaning-preserving; saves input tokens on every call.
        user = compress(prompt)

        total_tokens = 0
        last_text = ""  # best non-empty answer seen, used if all attempts fail validation
        for model in attempts:
            try:
                result = self._client.complete(
                    model=model,
                    system=spec.system,
                    user=user,
                    max_tokens=spec.max_tokens,
                )
            except Exception:
                continue  # try the next tier
            total_tokens += result.total_tokens
            # Strip reasoning traces before validating/shipping so a thinking
            # model's <think> block never reaches the judge.
            text = strip_reasoning(result.text)
            # Ship the first valid, non-empty answer. We do NOT force escalation
            # on finish_reason=="length": with a thinking model the reasoning is
            # what filled the budget, and after stripping it the visible answer is
            # usually complete. A second slow call rarely helps and risks the
            # wall-clock cap. Escalate only when there's no usable answer at all.
            if text and is_valid(r.category, text):
                return SolveOutcome(task_id, text, r.category, total_tokens)
            if text:
                last_text = text  # keep as fallback; escalate for a cleaner one

        return SolveOutcome(task_id, last_text or _FALLBACK, r.category, total_tokens)
