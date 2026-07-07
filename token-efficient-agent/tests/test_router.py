"""Tests for the local category classifier."""

from src.categories import Category
from src.router import classify


def test_summarization():
    assert classify("Summarise the following text in one sentence: ...") == Category.SUMMARIZATION


def test_sentiment():
    assert classify("Classify the sentiment of this review: great product") == Category.SENTIMENT


def test_ner():
    assert classify("Extract named entities from this paragraph") == Category.NER


def test_math():
    assert classify("Calculate 25% of 400") == Category.MATH


def test_code_debug():
    prompt = "Find and fix the bug:\n```python\ndef f(): return\n```"
    assert classify(prompt) == Category.CODE_DEBUG


def test_code_gen():
    assert classify("Write a function that reverses a string") == Category.CODE_GEN


def test_factual_default():
    assert classify("Who painted the Mona Lisa?") in (Category.FACTUAL, Category.LOGIC)
