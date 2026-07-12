"""Tests for local answer finalization."""

from src.categories import Category
from src.finalize import finalize_answer, strip_reasoning


def test_ner_compacts_json():
    messy = 'Here you go: {\n  "person": ["Ada"],\n  "org": [],\n  "location": [],\n  "date": []\n}'
    out = finalize_answer(Category.NER, messy)
    assert '"person":["Ada"]' in out


def test_ner_non_json_unchanged():
    assert finalize_answer(Category.NER, "no json here") == "no json here"


def test_generic_strips_whitespace():
    assert finalize_answer(Category.FACTUAL, "  Paris  ") == "Paris"


def test_empty_stays_empty():
    assert finalize_answer(Category.MATH, "") == ""


def test_math_extracts_answer_line():
    raw = "Step 1: 30 cookies / 12 = 2.5 ratio\nStep 2: 0.75 * 2.5 = 1.875 cups\nAnswer: 1.875 cups"
    out = finalize_answer(Category.MATH, raw)
    assert "1.875" in out


def test_math_keeps_full_when_no_answer_line():
    # If no explicit "Answer:" line, keep the full working for the judge
    raw = "30 cookies need 1.875 cups.\nCost = 1.875 x $2.40 = $4.50."
    out = finalize_answer(Category.MATH, raw)
    assert "1.875" in out and "$4.50" in out


def test_strips_think_block_math():
    raw = "<think>Let me compute 40*0.75 = 30, checking...</think>\nAnswer: 30"
    out = finalize_answer(Category.MATH, raw)
    assert out == "30"


def test_strips_mm_think_block():
    # MiniMax M3 uses <mm:think> tags
    raw = "<mm:think>Let me solve step by step. 2400 * 0.37 = 888...</mm:think>\nThe answer is 1672."
    out = finalize_answer(Category.MATH, raw)
    assert "1672" in out


def test_strips_gemma_think_block():
    # Gemma 4 uses <|think|> tags
    raw = "<|think|>Reasoning about the problem...<|/think|>\nParis is the capital of France."
    assert finalize_answer(Category.FACTUAL, raw) == "Paris is the capital of France."


def test_strips_think_block_prose():
    raw = "<think>The user wants the capital.</think>Paris is the capital of France."
    assert finalize_answer(Category.FACTUAL, raw) == "Paris is the capital of France."


def test_strips_dangling_open_think():
    # Truncated reasoning after a complete answer -> drop the dangling opener.
    raw = "Answer: 42\n<think>now let me double check by re-deriving"
    out = finalize_answer(Category.MATH, raw)
    assert "42" in out


def test_strips_dangling_close_think():
    # Opener lost to truncation; keep only what follows the close tag.
    raw = "reasoning about the problem...</think>\nAnswer: London"
    out = finalize_answer(Category.LOGIC, raw)
    assert "London" in out


def test_strip_reasoning_idempotent():
    cleaned = strip_reasoning("<think>x</think>hello")
    assert strip_reasoning(cleaned) == cleaned == "hello"
