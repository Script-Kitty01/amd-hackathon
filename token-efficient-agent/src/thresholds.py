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
# Only the MATH solver runs locally now, and it already abstains unless the
# pattern is a clean, unambiguous match — so trust its answers (it returns
# 0.88-0.95). Sentiment/NER solvers are disabled (see local_solvers registry);
# their thresholds are moot but kept for when they're re-enabled + verified.
_DEFAULT_LOCAL_SOLVER: dict[Category, float] = {
    Category.MATH: 0.85,       # accept the deterministic math solver's outputs
    Category.SENTIMENT: 0.85,
    Category.NER: 0.90,
}

# Min confidence to accept a LOCAL LLM answer. 0.99 => effectively always escalate.
# ACCURACY FIRST: Very high thresholds to force escalation to Fireworks models
_DEFAULT_LOCAL_LLM: dict[Category, float] = {
    Category.FACTUAL: 0.85,     # escalate most factual questions
    Category.SUMMARIZATION: 0.80,  # escalate most summarization
    Category.SENTIMENT: 0.85,   # escalate most sentiment
    Category.NER: 0.85,         # escalate most NER
    Category.CODE_GEN: 0.95,    # escalate almost all code generation
    Category.CODE_DEBUG: 0.99,  # escalate all debugging
    Category.MATH: 0.99,        # escalate all math
    Category.LOGIC: 0.99,       # escalate all logic
}

_FALLBACK_SOLVER = 0.90  # More conservative fallback
_FALLBACK_LLM = 0.85     # More conservative fallback

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
