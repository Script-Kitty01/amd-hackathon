"""Canonical task categories and per-category model policy.

Model selection is data-driven: `MODEL_PREFERENCE` maps each category to an
index into `config.models` (the ALLOWED_MODELS list). Index 0 is the safe
default. The launch-day sweep (eval.run_eval --sweep) emits a recommended
mapping to `config/model_preference.json`, which `load_model_preference()`
overlays at startup — so tuning needs no code changes.

Convention: arrange ALLOWED_MODELS cheapest -> most capable. Then index 0 is the
cheap default and the last entry is the escalation (strong) fallback.
"""

from __future__ import annotations

import json
import os
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    LOGIC = "logic"
    CODE_GEN = "code_gen"


# Calibrated per-category model index into config.models. Populated by the
# launch-day sweep (via load_model_preference); empty = not yet calibrated.
MODEL_PREFERENCE: dict[Category, int] = {}

# Fallback preference by model-name substring, matched against ALLOWED_MODELS at
# runtime. COMPLIANT: we only ever return a model that IS in ALLOWED_MODELS — the
# hints just express which allowed model suits a category (e.g. a code-specialised
# model for code, a reasoning model for logic/math). Used only when the sweep
# hasn't set a calibrated preference for the category.
# Three model groups: Gemma for language tasks, MiniMax for reasoning, Kimi for
# code. Hints are substring-matched against ALLOWED_MODELS at runtime, so we only
# ever return a model that IS permitted; if no hint matches, select_model falls
# back to the first allowed model (always compliant).
MODEL_HINTS: dict[Category, tuple[str, ...]] = {
    # Gemma-first doctrine: Gemma is non-reasoning by default + densest tokenizer
    # (262k vocab) = cheapest on every axis. Escalation to reasoning models only
    # happens via the escalation_model fallback path.
    #
    # Code -> gemma-4-31b-it first (LiveCodeBench 80%, non-reasoning, terse).
    # Escalation to kimi only if gemma fails.
    Category.CODE_DEBUG: ("gemma", "31b"),
    Category.CODE_GEN: ("gemma", "31b"),
    # Math/Logic -> gemma-4-31b-it first (AIME 89.2%, non-reasoning).
    # Escalation to minimax-m3 (thinking-on) only if gemma fails.
    Category.LOGIC: ("gemma", "31b"),
    Category.MATH: ("gemma", "31b"),
    # Language / knowledge tasks -> Gemma 31b for quality (not the tiny 26b).
    Category.FACTUAL: ("gemma", "31b"),
    Category.SENTIMENT: ("gemma", "31b"),
    Category.SUMMARIZATION: ("gemma", "31b"),
    Category.NER: ("gemma", "31b"),
}

_PREF_ENV = "MODEL_PREFERENCE_PATH"
_DEFAULT_PREF_PATH = "config/model_preference.json"


def load_model_preference(path: str | None = None) -> None:
    """Overlay MODEL_PREFERENCE from a JSON {category: index} file if present.

    Produced by the launch-day model sweep. A missing or malformed file leaves
    the safe all-zero defaults untouched.
    """
    path = path or os.environ.get(_PREF_ENV, _DEFAULT_PREF_PATH)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(raw, dict):
        return
    for name, idx in raw.items():
        try:
            MODEL_PREFERENCE[Category(name)] = int(idx)
        except (ValueError, TypeError):
            continue


def select_model(category: Category, models: list[str]) -> str:
    """Pick the preferred allowed model for a category.

    Precedence: calibrated sweep index > name-hint match within ALLOWED_MODELS >
    first allowed model. Only ever returns a model present in `models`.
    """
    if category in MODEL_PREFERENCE:
        idx = max(0, min(MODEL_PREFERENCE[category], len(models) - 1))
        return models[idx]

    for hint in MODEL_HINTS.get(category, ()):
        for m in models:
            if hint in m.lower():
                return m

    return models[0]


def escalation_model(models: list[str]) -> str:
    """Strongest available model, used when a primary attempt fails.

    Prefers minimax-m3 (strong reasoning) as the escalation target, then kimi
    for code. Falls back to the last model in the list if neither is found.
    """
    for hint in ("minimax", "kimi"):
        for m in models:
            if hint in m.lower():
                return m
    return models[-1]
