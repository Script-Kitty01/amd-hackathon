"""Tests for tiered model selection / escalation (T6).

Uses a fake client so no network or API key is required.
"""

from dataclasses import dataclass

import src.categories as categories
from src.categories import Category
from src.config import Config
from src.solver import Solver

# Prompts with known routing outcomes.
EASY_SENTIMENT = "Classify the sentiment of this review: great product, loved it"
COMPLEX_LOGIC = (
    "Alice, Bob, and Carol finished 1st, 2nd, 3rd in some order. "
    "Alice was not last. Carol beat Bob. Who came first?"
)


@dataclass
class FakeResult:
    text: str
    total_tokens: int


class FakeClient:
    """Returns a configured (text, tokens) per model, or raises if given an Exception."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def complete(self, model, system, user, max_tokens):
        self.calls.append(model)
        r = self.responses[model]
        if isinstance(r, Exception):
            raise r
        text, tokens = r
        return FakeResult(text=text, total_tokens=tokens)


def _cfg(models):
    return Config(api_key="x", base_url="http://x", models=models)


def setup_function(_):
    # Reset preferences to defaults (index 0) before each test.
    for c in Category:
        categories.MODEL_PREFERENCE[c] = 0


def test_easy_task_uses_cheap_primary_only():
    client = FakeClient({"cheap": ("Positive.", 12), "strong": ("x", 99)})
    solver = Solver(_cfg(["cheap", "strong"]), client)
    out = solver.solve("t1", EASY_SENTIMENT)
    assert client.calls == ["cheap"]
    assert out.answer == "Positive."
    assert out.total_tokens == 12


def test_logic_routes_to_reasoning_model_single_call():
    # Accuracy-first: logic goes straight to the reasoning model (minimax),
    # one call, no speculative cheap call first.
    client = FakeClient({"minimax-m3": ("Alice.", 40), "kimi-k2p7-code": ("x", 99)})
    solver = Solver(_cfg(["minimax-m3", "kimi-k2p7-code"]), client)
    out = solver.solve("t2", COMPLEX_LOGIC)
    assert client.calls == ["minimax-m3"]
    assert out.answer == "Alice."


def test_empty_primary_escalates_and_sums_tokens():
    client = FakeClient({"cheap": ("", 5), "strong": ("Positive.", 20)})
    solver = Solver(_cfg(["cheap", "strong"]), client)
    out = solver.solve("t3", EASY_SENTIMENT)
    assert client.calls == ["cheap", "strong"]
    assert out.answer == "Positive."
    assert out.total_tokens == 25  # both calls counted


def test_primary_error_escalates():
    client = FakeClient({"cheap": RuntimeError("boom"), "strong": ("Positive.", 20)})
    solver = Solver(_cfg(["cheap", "strong"]), client)
    out = solver.solve("t4", EASY_SENTIMENT)
    assert client.calls == ["cheap", "strong"]
    assert out.answer == "Positive."


def test_all_attempts_fail_returns_fallback():
    client = FakeClient({"cheap": ("", 5), "strong": ("", 7)})
    solver = Solver(_cfg(["cheap", "strong"]), client)
    out = solver.solve("t5", EASY_SENTIMENT)
    assert out.answer == "Unable to produce an answer."
    assert out.total_tokens == 12


def test_single_model_no_escalation():
    client = FakeClient({"only": ("Positive.", 10)})
    solver = Solver(_cfg(["only"]), client)
    out = solver.solve("t6", EASY_SENTIMENT)
    assert client.calls == ["only"]
    assert out.answer == "Positive."
