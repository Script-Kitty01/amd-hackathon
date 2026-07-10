"""Confidence thresholds for the cascade (T27 calibration, T28 escalation policy).

A local answer is accepted only if its confidence meets the threshold for its
tier and category; otherwise the cascade escalates. Higher threshold = more
conservative = escalate more often.

Defaults encode the escalation policy (T28):
  - Trust deterministic local solvers where they're reliable (math).
  - Distrust the local LLM on reasoning/verification-heavy categories (math,
    logic) so they escalate to Fireworks — protects the accuracy gate.

Calibration (T27) overlays measured cutoffs from `config/confidence.json`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .categories import Category

# Min confidence to accept a DETERMINISTIC local-solver answer.
_DEFAULT_LOCAL_SOLVER: dict[Category, float] = {
    Category.MATH: 0.85,       # deterministic parse -> trust it
    Category.SENTIMENT: 0.70,  # lexicon with clear margin
    Category.NER: 0.90,        # heuristic -> strict, usually escalate
}

# Min confidence to accept a LOCAL LLM answer. 0.99 => effectively always escalate.
_DEFAULT_LOCAL_LLM: dict[Category, float] = {
    Category.FACTUAL: 0.65,
    Category.SUMMARIZATION: 0.65,
    Category.SENTIMENT: 0.65,
    Category.NER: 0.60,
    Category.CODE_GEN: 0.70,
    Category.CODE_DEBUG: 0.90,
    Category.MATH: 0.99,   # don't trust LLM arithmetic -> escalate
    Category.LOGIC: 0.99,  # conservative on deductive reasoning -> escalate
}

_FALLBACK_SOLVER = 0.80
_FALLBACK_LLM = 0.70

_THRESH_ENV = "CONFIDENCE_PATH"
_DEFAULT_THRESH_PATH = "config/confidence.json"


@dataclass
class Thresholds:
    local_solver_thr: dict[Category, float]
    local_llm_thr: dict[Category, float]

    def local_solver(self, category: Category) -> float:
        return self.local_solver_thr.get(category, _FALLBACK_SOLVER)

    def local_llm(self, category: Category) -> float:
        return self.local_llm_thr.get(category, _FALLBACK_LLM)


def _overlay(base: dict[Category, float], raw: dict) -> None:
    for name, val in raw.items():
        try:
            base[Category(name)] = float(val)
        except (ValueError, TypeError):
            continue


def load_thresholds(path: str | None = None) -> Thresholds:
    """Defaults, overlaid by `config/confidence.json` if present."""
    solver = dict(_DEFAULT_LOCAL_SOLVER)
    llm = dict(_DEFAULT_LOCAL_LLM)

    path = path or os.environ.get(_THRESH_ENV, _DEFAULT_THRESH_PATH)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return Thresholds(solver, llm)

    if isinstance(raw, dict):
        if isinstance(raw.get("local_solver"), dict):
            _overlay(solver, raw["local_solver"])
        if isinstance(raw.get("local_llm"), dict):
            _overlay(llm, raw["local_llm"])
    return Thresholds(solver, llm)
