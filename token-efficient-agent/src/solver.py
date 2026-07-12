"""Per-task orchestration: route -> pick tier -> call -> escalate -> fallback.

ACCURACY-FIRST strategy: use Gemma-31b (non-reasoning, dense tokenizer) as the
primary model for ALL categories. Escalate to minimax-m3 (reasoning) or
kimi-k2p7-code only when gemma fails or returns empty/invalid output.

Escalation is more aggressive than before: if the primary answer fails
validation or looks suspiciously short for reasoning tasks, we try the
escalation model. A few extra tokens on the failure path is always worth it
vs. failing the accuracy gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .categories import Category, escalation_model, select_model
from .compress import compress
from .config import Config
from .finalize import finalize_answer
from .prompts import spec_for
from .router import route
from .validate import is_valid, needs_escalation

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
    def __init__(self, cfg: Config, client: "FireworksClient") -> None:
        self._cfg = cfg
        self._client = client

    def _plan_attempts(self, category: Category, complexity: str, ambiguous: bool) -> list[str]:
        """Ordered models to try for one task.

        Primary: Gemma-31b (non-reasoning, terse, dense tokenizer).
        Escalation: minimax-m3 (reasoning) or kimi (code) when primary fails.
        Third backstop: try the other strong model if both above fail.
        """
        models = self._cfg.models
        primary = select_model(category, models)
        strong = escalation_model(models)

        attempts = [primary]
        if strong != primary:
            attempts.append(strong)
        # Add a third backstop for hard categories: try kimi for code, minimax for reasoning
        if category in (Category.CODE_DEBUG, Category.CODE_GEN):
            for m in models:
                if "kimi" in m.lower() and m not in attempts:
                    attempts.append(m)
                    break
        elif category in (Category.MATH, Category.LOGIC):
            for m in models:
                if "minimax" in m.lower() and m not in attempts:
                    attempts.append(m)
                    break
        return attempts

    def solve(self, task_id: str, prompt: str) -> SolveOutcome:
        r = route(prompt)
        spec = spec_for(r.category)
        attempts = self._plan_attempts(r.category, r.complexity, r.ambiguous)
        # Compress the prompt for the remote call only (routing used the original).
        user = compress(prompt)

        total_tokens = 0
        last_text = ""  # best non-empty answer seen
        for model in attempts:
            try:
                result = self._client.complete(
                    model=model,
                    system=spec.system,
                    user=user,
                    max_tokens=spec.max_tokens,
                    stop=spec.stop,
                )
            except Exception:
                continue  # try the next model
            total_tokens += result.total_tokens
            # Finalize: strip reasoning traces, extract answer lines, compact NER
            text = finalize_answer(r.category, result.text)

            # Accept the first valid answer that doesn't need escalation
            if text and is_valid(r.category, text):
                if not needs_escalation(r.category, text):
                    return SolveOutcome(task_id, text, r.category, total_tokens)
                # Answer is structurally valid but weak — keep it, try escalation
                if not last_text or len(text) > len(last_text):
                    last_text = text
            elif text:
                last_text = text  # keep as fallback

        return SolveOutcome(task_id, last_text or _FALLBACK, r.category, total_tokens)
