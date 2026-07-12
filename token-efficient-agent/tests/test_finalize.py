"""Tests for local answer finalization."""

from src.categories import Category
from src.finalize import finalize, strip_reasoning


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


def test_math_keeps_full_multipart_answer():
    # Must NOT reduce to the last line — both values are needed by the judge.
    raw = "30 cookies need 1.875 cups.\nCost = 1.875 x $2.40 = $4.50."
    out = finalize(Category.MATH, raw)
    assert "1.875" in out and "$4.50" in out


def test_strips_think_block_math():
    raw = "<think>Let me compute 40*0.75 = 30, checking...</think>\nAnswer: 30"
    assert finalize(Category.MATH, raw) == "Answer: 30"


def test_strips_think_block_prose():
    raw = "<think>The user wants the capital.</think>Paris is the capital of France."
    assert finalize(Category.FACTUAL, raw) == "Paris is the capital of France."


def test_strips_dangling_open_think():
    # Truncated reasoning after a complete answer -> drop the dangling opener.
    raw = "Answer: 42\n<think>now let me double check by re-deriving"
    assert finalize(Category.MATH, raw) == "Answer: 42"


def test_strips_dangling_close_think():
    # Opener lost to truncation; keep only what follows the close tag.
    raw = "reasoning about the problem...</think>\nLondon came first."
    assert finalize(Category.LOGIC, raw) == "London came first."


def test_strip_reasoning_idempotent():
    cleaned = strip_reasoning("<think>x</think>hello")
    assert strip_reasoning(cleaned) == cleaned == "hello"
