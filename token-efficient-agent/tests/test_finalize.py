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


def test_strips_think_block_math():
    raw = "<think>Let me compute 40*0.75 = 30, checking...</think>\nAnswer: 30"
    assert finalize(Category.MATH, raw) == "30"


def test_strips_think_block_prose():
    raw = "<think>The user wants the capital.</think>Paris is the capital of France."
    assert finalize(Category.FACTUAL, raw) == "Paris is the capital of France."


def test_strips_dangling_open_think():
    # Truncated reasoning with no answer after the opener -> nothing usable.
    raw = "Answer: 42\n<think>now let me double check by re-deriving"
    assert finalize(Category.MATH, raw) == "42"


def test_strips_dangling_close_think():
    # Opener lost to truncation; keep only what follows the close tag.
    raw = "reasoning about the problem...</think>\nAnswer: London"
    assert finalize(Category.LOGIC, raw) == "London"


def test_logic_falls_back_to_last_line():
    raw = "First, Alice isn't last. Carol beat Bob.\nAlice came first."
    assert finalize(Category.LOGIC, raw) == "Alice came first"
