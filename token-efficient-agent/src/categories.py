"""Canonical task categories and per-category runtime configuration.

This is one of the two files you will tune (the other is prompts.py).
`MODEL_PREFERENCE` is filled in on launch day once ALLOWED_MODELS is known:
map each category to the *index* of the preferred model within ALLOWED_MODELS.
Index 0 (the first allowed model) is the safe default for every category.
"""

from __future__ import annotations

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
# Default everything to 0 until launch-day eval says otherwise.
MODEL_PREFERENCE: dict[Category, int] = {c: 0 for c in Category}


def select_model(category: Category, models: list[str]) -> str:
    """Pick the allowed model for a category, clamped to available range."""
    idx = MODEL_PREFERENCE.get(category, 0)
    idx = min(idx, len(models) - 1)
    return models[idx]
