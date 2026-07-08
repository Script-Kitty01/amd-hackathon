"""Tests for the local category classifier."""

from src.categories import Category
from src.router import classify, route


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


def test_logic_ranking_puzzle():
    prompt = ("Alice, Bob, and Carol finished 1st, 2nd, 3rd in some order. "
              "Alice was not last. Carol beat Bob. Who came first?")
    assert classify(prompt) == Category.LOGIC


# --- scoring / confidence / complexity (T4, T5) ---

def test_route_confident_on_clear_signal():
    r = route("Extract named entities from this paragraph")
    assert r.category == Category.NER
    assert r.confidence == 1.0
    assert r.ambiguous is False


def test_route_ambiguous_on_no_signal():
    r = route("The weather today.")
    assert r.category == Category.FACTUAL  # safe default
    assert r.ambiguous is True
    assert r.confidence == 0.0


def test_route_complexity_easy_vs_complex():
    easy = route("Calculate 25% of 400")
    assert easy.complexity == "easy"

    complex_puzzle = route(
        "Alice, Bob, and Carol finished 1st, 2nd, 3rd in some order. "
        "Alice was not last. Carol beat Bob. Who came first?"
    )
    assert complex_puzzle.complexity == "complex"
