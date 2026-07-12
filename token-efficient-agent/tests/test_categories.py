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


def test_code_tasks_prefer_gemma_31b():
    # Gemma-first doctrine: code tasks use gemma-4-31b-it (non-reasoning, dense tokenizer)
    assert select_model(Category.CODE_GEN, ALLOWED) == "gemma-4-31b-it"
    assert select_model(Category.CODE_DEBUG, ALLOWED) == "gemma-4-31b-it"


def test_reasoning_tasks_prefer_gemma_31b():
    # Gemma-first: math/logic also start with gemma (AIME 89.2%, non-reasoning)
    assert select_model(Category.LOGIC, ALLOWED) == "gemma-4-31b-it"
    assert select_model(Category.MATH, ALLOWED) == "gemma-4-31b-it"


def test_language_tasks_prefer_gemma():
    # factual/sentiment/summarization/NER route to the Gemma group.
    assert select_model(Category.FACTUAL, ALLOWED) == "gemma-4-31b-it"
    assert select_model(Category.SENTIMENT, ALLOWED) == "gemma-4-31b-it"
    assert select_model(Category.SUMMARIZATION, ALLOWED) == "gemma-4-31b-it"
    assert select_model(Category.NER, ALLOWED) == "gemma-4-31b-it"


def test_calibrated_preference_overrides_hint():
    categories.MODEL_PREFERENCE[Category.CODE_GEN] = 2  # sweep says index 2
    assert select_model(Category.CODE_GEN, ALLOWED) == "gemma-4-31b-it"


def test_hint_falls_back_when_no_match():
    # No code/kimi model present -> fall back to first.
    models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    assert select_model(Category.CODE_GEN, models) == "gemma-4-31b-it"


def test_escalation_prefers_reasoning_model():
    # Escalation picks minimax-m3 (strong reasoning) over just the last model
    assert escalation_model(ALLOWED) == "minimax-m3"


def test_escalation_fallback_when_no_reasoning():
    # Without minimax/kimi, falls back to the last model
    models = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
    assert escalation_model(models) == "gemma-4-26b-a4b-it"
