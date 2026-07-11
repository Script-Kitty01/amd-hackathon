"""Cascade orchestrator (T26): the local-first confidence cascade.

Order per task:
  1. router.route()            -> category + confidence + complexity (0 tokens)
  2. deterministic local solver -> accept if confidence >= threshold (0 tokens)
  3. local LLM (gemma)          -> accept if confidence >= threshold (0 tokens)
  4. Fireworks fallback         -> paid tokens, tiered cheap->strong

Guarantees a non-empty answer for every task. Records which tier answered and
how many (Fireworks) tokens it cost, for eval/observability.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional

from .categories import Category
from .finalize import finalize
from .local_llm import LocalLLM
from .local_solvers import Solution, solvers_for
from .router import route
from .thresholds import Thresholds, load_thresholds

_FALLBACK = "Unable to produce an answer."


@dataclass
class CascadeOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int
    tier: str  # local_solver | local_llm | fireworks | best_effort


class Cascade:
    def __init__(
        self,
        thresholds: Optional[Thresholds] = None,
        fireworks_solver=None,  # object with .solve(task_id, prompt) -> SolveOutcome
        local_llm: Optional[LocalLLM] = None,
    ) -> None:
        self._thr = thresholds or load_thresholds()
        self._fireworks = fireworks_solver
        self._local_llm = local_llm
        # In-run dedup cache (T13): identical prompts computed once per run.
        self._cache: dict[str, CascadeOutcome] = {}
        self._cache_guard = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}

    def solve(self, task_id: str, prompt: str) -> CascadeOutcome:
        """Cached entry point: equivalent prompts reuse the first result (0 extra tokens)."""
        key = _normalize_key(prompt)

        with self._cache_guard:
            hit = self._cache.get(key)
            if hit is not None:
                return _reuse(hit, task_id)
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # Serialize identical prompts so we don't pay for the same work twice.
        with key_lock:
            with self._cache_guard:
                hit = self._cache.get(key)
                if hit is not None:
                    return _reuse(hit, task_id)

            outcome = self._solve_uncached(task_id, prompt)

            with self._cache_guard:
                self._cache[key] = outcome
            return outcome

    def _solve_uncached(self, task_id: str, prompt: str) -> CascadeOutcome:
        r = route(prompt)
        cat = r.category
        best: Optional[Solution] = None  # highest-confidence local answer seen

        # Tier 1: deterministic local solvers.
        for solver in solvers_for(cat):
            sol = solver.try_solve(prompt)
            if sol is None:
                continue
            best = _better(best, sol)
            if sol.confidence >= self._thr.local_solver(cat):
                return CascadeOutcome(task_id, finalize(cat, sol.answer), cat, 0, "local_solver")

        # Tier 2: local LLM.
        if self._local_llm is not None:
            sol = self._local_llm.try_solve(cat, prompt)
            if sol is not None:
                best = _better(best, sol)
                if sol.confidence >= self._thr.local_llm(cat):
                    return CascadeOutcome(task_id, finalize(cat, sol.answer), cat, 0, "local_llm")

        # Tier 3: Fireworks fallback (paid).
        if self._fireworks is not None:
            out = self._fireworks.solve(task_id, prompt)
            answer = out.answer or (best.answer if best else _FALLBACK)
            return CascadeOutcome(
                task_id, finalize(out.category, answer), out.category, out.total_tokens, "fireworks"
            )

        # No Fireworks configured: return the best local answer we have.
        if best is not None:
            return CascadeOutcome(task_id, best.answer, cat, 0, "best_effort")
        return CascadeOutcome(task_id, _FALLBACK, cat, 0, "best_effort")


def _better(current: Optional[Solution], candidate: Solution) -> Solution:
    if current is None or candidate.confidence > current.confidence:
        return candidate
    return current


def _reuse(hit: CascadeOutcome, task_id: str) -> CascadeOutcome:
    """Reuse a cached answer for a duplicate prompt — no additional tokens."""
    return CascadeOutcome(task_id, hit.answer, hit.category, 0, "cache")


_WS = re.compile(r"\s+")


def _normalize_key(prompt: str) -> str:
    """Cache key: whitespace-collapsed, lowercased — so trivially-equivalent
    prompts (spacing/case) share one computation. In-run only; nothing persisted."""
    return _WS.sub(" ", prompt.strip().lower())
