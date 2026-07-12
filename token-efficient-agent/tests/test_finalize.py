"""Tests for local answer finalization."""

from src.categories import Category
from src.finalize import finalize, strip_reasoning


def test_ner_compacts_json():
    messy = 'Here you go: {\n  "person": ["Ada"],\n  "org": [],\n  "location": [],\n  "date": []\n}'
    out = finalize(Category.NER, messy)
    assert '"person":["Ada"]' in out


def test_ner_non_json_unchanged():
    assert finalize(Category.NER, "no json here") == "no json here"


def test_generic_strips_whitespace():
    assert finalize(Category.FACTUAL, "  Paris  ") == "Paris"


def test_empty_stays_empty():
    assert finalize(Category.MATH, "") == ""


def test_math_extracts_answer_line():
    raw = "Step 1: 30/12 = 2.5\nStep 2: 0.75 * 2.5 = 1.875 cups\nAnswer: 1.875 cups"
    out = finalize(Category.MATH, raw)
    assert "1.875" in out


def test_math_keeps_full_multipart_when_no_answer_line():
    raw = "30 cookies need 1.875 cups.\nCost = 1.875 x $2.40 = $4.50."
    out = finalize(Category.MATH, raw)
    # No Answer: line → keep last line so both values are visible to judge
    assert "$4.50" in out


def test_strips_think_block_math():
    raw = "<think>Let me compute 40*0.75 = 30</think>\nAnswer: 30"
    assert finalize(Category.MATH, raw) == "30"


def test_strips_mm_think_block():
    raw = "<mm:think>2400 * 0.37 = 888...</mm:think>\nThe answer is 1672."
    out = finalize(Category.MATH, raw)
    assert "1672" in out


def test_strips_gemma_think_block():
    raw = "<|think|>Reasoning...<|/think|>\nParis is the capital of France."
    out = finalize(Category.FACTUAL, raw)
    assert "Paris is the capital of France" in out


def test_strips_think_block_prose():
    raw = "<think>The user wants the capital.</think>Paris is the capital of France."
    out = finalize(Category.FACTUAL, raw)
    assert "Paris is the capital of France" in out


def test_strips_dangling_open_think():
    raw = "Answer: 42\n<think>now let me double check"
    out = finalize(Category.MATH, raw)
    assert "42" in out


def test_strips_dangling_close_think():
    raw = "reasoning about the problem...</think>\nAnswer: London"
    out = finalize(Category.LOGIC, raw)
    assert "London" in out


def test_strip_reasoning_idempotent():
    cleaned = strip_reasoning("<think>x</think>hello")
    assert strip_reasoning(cleaned) == cleaned == "hello"


def test_strips_answer_prefix():
    assert finalize(Category.FACTUAL, "Answer: Paris") == "Paris"


def test_strips_outer_quotes_short():
    assert finalize(Category.FACTUAL, '"Paris"') == "Paris"


def test_strips_trailing_period_short():
    assert finalize(Category.FACTUAL, "Paris.") == "Paris"


def test_does_not_strip_period_long():
    # Long answers should keep their trailing period
    long = "Machine learning is a subset of AI that learns patterns from data."
    assert finalize(Category.FACTUAL, long).endswith(".")


def test_prose_spill_stripped_for_non_reasoning():
    # "We need to" is score-2, should be stripped from factual
    raw = "We need to analyze this. The answer is Paris."
    out = finalize(Category.FACTUAL, raw)
    assert "Paris" in out


def test_prose_spill_not_stripped_for_math():
    # Math categories keep their working even if it looks like spill
    raw = "We need to find 30% of 2400. 2400 * 0.3 = 720. Answer: 720"
    out = finalize(Category.MATH, raw)
    assert "720" in out
