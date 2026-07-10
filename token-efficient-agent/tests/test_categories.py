"""Tests for model selection over the real ALLOWED_MODELS set."""

import src.categories as categories
from src.categories import Category, escalation_model, select_model

# The actual Track 1 allowed models.
ALLOWED = [
    "minimax-m3",
    "kimi-k2p7-code",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
]


def setup_function(_):
    categories.MODEL_PREFERENCE.clear()  # no calibrated prefs -> hints apply


def test_code_tasks_prefer_code_model():
    assert select_model(Category.CODE_GEN, ALLOWED) == "kimi-k2p7-code"
    assert select_model(Category.CODE_DEBUG, ALLOWED) == "kimi-k2p7-code"


def test_reasoning_tasks_prefer_reasoning_model():
    assert select_model(Category.LOGIC, ALLOWED) == "minimax-m3"
    assert select_model(Category.MATH, ALLOWED) == "minimax-m3"


def test_uncategorized_defaults_to_first_allowed():
    assert select_model(Category.FACTUAL, ALLOWED) == "minimax-m3"  # models[0]


def test_calibrated_preference_overrides_hint():
    categories.MODEL_PREFERENCE[Category.CODE_GEN] = 2  # sweep says index 2
    assert select_model(Category.CODE_GEN, ALLOWED) == "gemma-4-31b-it"


def test_hint_falls_back_when_no_match():
    # No code/kimi model present -> fall back to first.
    models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    assert select_model(Category.CODE_GEN, models) == "gemma-4-31b-it"


def test_escalation_is_last_model():
    assert escalation_model(ALLOWED) == "gemma-4-31b-it-nvfp4"
