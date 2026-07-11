"""Tests for meaning-preserving prompt compression (Requirement 8.1)."""

from src.compress import compress


def test_collapses_whitespace():
    assert compress("Explain    this\t\tnow") == "Explain this now"


def test_collapses_blank_lines():
    assert compress("a\n\n\n\n\nb") == "a\n\nb"


def test_idempotent():
    prompts = [
        "Please   summarise this in one sentence:   The park opens soon.",
        "Fix the bug:\n```python\ndef f(): return 1+2\n```",
        "What is 2 plus 2?",
    ]
    for p in prompts:
        once = compress(p)
        assert compress(once) == once


def test_preserves_code_block_verbatim():
    p = "Fix:\n```python\ndef  add(a,b):\n    return   a+b\n```"
    out = compress(p)
    # The code block (with its internal spacing) is untouched.
    assert "```python\ndef  add(a,b):\n    return   a+b\n```" in out


def test_preserves_numbers():
    p = "A $2,400 warehouse sells 37% then 640 units. Compute the remainder."
    out = compress(p)
    for token in ("2,400", "37%", "640"):
        assert token in out


def test_strips_leading_filler():
    assert compress("Please answer: what is RAM?").lower().startswith("answer")
    assert compress("Could you please explain X").lower().startswith("explain")


def test_empty():
    assert compress("") == ""
