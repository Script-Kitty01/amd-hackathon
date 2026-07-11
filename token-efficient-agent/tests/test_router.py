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


def test_average_in_prose_does_not_route_to_math():
    # "average O(1)" is not a math task — must not misroute to MATH.
    prompt = "Explain how a hash table achieves average O(1) lookups and when it degrades to O(n)."
    assert classify(prompt) == Category.FACTUAL


def test_numeric_arithmetic_routes_to_math():
    assert classify("What is the average of 4, 8, and 12?") == Category.MATH
    assert classify("What is 15 plus 27?") == Category.MATH
    assert classify("What percent of 200 is 50?") == Category.MATH
    assert classify("Find the sum of 10, 20 and 30.") == Category.MATH


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
