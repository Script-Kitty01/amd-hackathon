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


# Preferred model index into config.models, per category.
# Default everything to 0 (cheapest) until launch-day eval says otherwise.
MODEL_PREFERENCE: dict[Category, int] = {c: 0 for c in Category}

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
    """Pick the preferred allowed model for a category, clamped to range."""
    idx = MODEL_PREFERENCE.get(category, 0)
    idx = max(0, min(idx, len(models) - 1))
    return models[idx]


def escalation_model(models: list[str]) -> str:
    """Strongest available model, used when a primary attempt fails.

    Assumes ALLOWED_MODELS is ordered cheapest -> most capable (see module docs).
    """
    return models[-1]
