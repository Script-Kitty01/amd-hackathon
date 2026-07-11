"""Tests for answer validation (Requirements 2, 5)."""

from src.categories import Category
from src.validate import is_valid


def test_empty_and_fallback_invalid():
    assert not is_valid(Category.FACTUAL, "")
    assert not is_valid(Category.FACTUAL, "   ")
    assert not is_valid(Category.FACTUAL, "Unable to produce an answer.")


def test_factual_any_nonempty_valid():
    assert is_valid(Category.FACTUAL, "Paris is the capital of France.")


def test_sentiment_requires_label():
    assert is_valid(Category.SENTIMENT, "Positive. The reviewer loved it.")
    assert is_valid(Category.SENTIMENT, "This is Mixed: good screen but bad battery.")
    assert not is_valid(Category.SENTIMENT, "The reviewer had a lot to say.")


def test_ner_accepts_json():
    assert is_valid(Category.NER, '{"person":["Ada"],"organization":[],"location":[],"date":[]}')


def test_ner_accepts_labeled_text():
    assert is_valid(Category.NER, "Sundar Pichai - PERSON; Google - ORGANIZATION")


def test_ner_rejects_prose_without_entities():
    assert not is_valid(Category.NER, "there are some names in the text")
