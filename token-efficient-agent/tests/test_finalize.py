"""Tests for local answer finalization (T29)."""

from src.categories import Category
from src.finalize import finalize


def test_math_extracts_answer_value():
    assert finalize(Category.MATH, "Working: 40 * 0.75 = 30\nAnswer: 30") == "30"


def test_math_without_answer_line_is_unchanged():
    assert finalize(Category.MATH, "30") == "30"


def test_ner_compacts_json():
    messy = 'Here you go: {\n  "person": ["Ada"],\n  "org": [],\n  "location": [],\n  "date": []\n}'
    out = finalize(Category.NER, messy)
    assert out == '{"person":["Ada"],"org":[],"location":[],"date":[]}'


def test_ner_non_json_unchanged():
    assert finalize(Category.NER, "no json here") == "no json here"


def test_generic_strips_whitespace():
    assert finalize(Category.FACTUAL, "  Paris  ") == "Paris"


def test_empty_stays_empty():
    assert finalize(Category.MATH, "") == ""
