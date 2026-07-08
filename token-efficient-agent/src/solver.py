"""Per-task orchestration: route -> select model -> prompt -> call -> fallback.

Guarantees a non-empty answer string for every task so results.json is always
valid and complete, even if an individual API call fails.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .categories import Category, select_model
from .config import Config
from .google_client import GoogleClient
from .prompts import spec_for
from .router import classify


@dataclass
class SolveOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int


class Solver:
    def __init__(self, cfg: Config, client: GoogleClient) -> None:
        self._cfg = cfg
        self._client = client

    def solve(self, task_id: str, prompt: str) -> SolveOutcome:
        category = classify(prompt)
        spec = spec_for(category)
        model = select_model(category, self._cfg.models)

        try:
            print(f"Solver: Attempting to solve task {task_id} with category {category} and model {model}")
            time.sleep(12.5)  # Throttle for Google Free Tier (5 requests/minute limit)
            result = self._client.complete(
                model=model,
                system=spec.system,
                user=prompt,
                max_tokens=spec.max_tokens,
            )
            answer = result.text or "Unable to produce an answer."
            print(f"Solver: Task {task_id} solved successfully.")
            return SolveOutcome(task_id, answer, category, result.total_tokens)
        except Exception as e:
            print(f"Solver: Initial attempt for task {task_id} failed: {e}. Retrying with default model.")
            # One retry with the default model before falling back.
            try:
                result = self._client.complete(
                    model=self._cfg.default_model,
                    system=spec.system,
                    user=prompt,
                    max_tokens=spec.max_tokens,
                )
                print(f"Solver: Task {task_id} solved successfully with default model.")
                return SolveOutcome(task_id, result.text or "", category, result.total_tokens)
            except Exception as e_retry:
                print(f"Solver: Retry attempt for task {task_id} failed: {e_retry}. Returning fallback answer.")
                return SolveOutcome(task_id, "Unable to produce an answer.", category, 0)
