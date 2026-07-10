"""Tests for the cascade orchestrator (T26) and thresholds (T27/T28)."""

from dataclasses import dataclass

from src.cascade import Cascade
from src.categories import Category
from src.local_llm import LocalLLM
from src.thresholds import Thresholds, load_thresholds


@dataclass
class FWOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int


class FakeFireworks:
    def __init__(self, answer="fireworks-answer", tokens=50):
        self.answer = answer
        self.tokens = tokens
        self.called = 0

    def solve(self, task_id, prompt):
        self.called += 1
        # category doesn't matter for these tests
        return FWOutcome(task_id, self.answer, Category.FACTUAL, self.tokens)


def const_llm(answer, samples=1):
    return LocalLLM("gemma", lambda s, u, m, t: answer, samples=samples)


# --- thresholds ---

def test_default_thresholds_conservative_on_reasoning():
    thr = load_thresholds("does/not/exist.json")
    assert thr.local_llm(Category.LOGIC) >= 0.99
    assert thr.local_llm(Category.MATH) >= 0.99
    assert thr.local_solver(Category.MATH) <= 0.9


# --- cascade tiering ---

def test_math_answered_by_local_solver_zero_tokens():
    fw = FakeFireworks()
    c = Cascade(fireworks_solver=fw)
    out = c.solve("t", "Calculate the final price of a $120 item after a 15% discount.")
    assert out.tier == "local_solver"
    assert out.total_tokens == 0
    assert out.answer == "$102.00"
    assert fw.called == 0  # never escalated


def test_low_confidence_local_escalates_to_fireworks():
    # Logic has no local solver and local LLM is conservative-gated -> Fireworks.
    fw = FakeFireworks(answer="Alice", tokens=30)
    c = Cascade(fireworks_solver=fw, local_llm=const_llm("Alice?"))
    out = c.solve(
        "t",
        "Alice, Bob, and Carol finished 1st, 2nd, 3rd in some order. "
        "Alice was not last. Carol beat Bob. Who came first?",
    )
    assert out.tier == "fireworks"
    assert out.total_tokens == 30
    assert fw.called == 1


def test_local_llm_answers_factual_when_confident():
    fw = FakeFireworks()
    # factual prior 0.70 >= default local_llm threshold 0.65 -> accept locally
    c = Cascade(fireworks_solver=fw, local_llm=const_llm("Paris"))
    out = c.solve("t", "What is the capital of France?")
    assert out.tier == "local_llm"
    assert out.total_tokens == 0
    assert out.answer == "Paris"
    assert fw.called == 0


def test_no_fireworks_returns_best_effort_local():
    # Raise local_llm threshold so nothing is accepted; no fireworks configured.
    strict = Thresholds(local_solver_thr={}, local_llm_thr={Category.FACTUAL: 1.01})
    c = Cascade(thresholds=strict, local_llm=const_llm("Paris"))
    out = c.solve("t", "What is the capital of France?")
    assert out.tier == "best_effort"
    assert out.answer == "Paris"  # best local answer, not the canned fallback
    assert out.total_tokens == 0


def test_dedup_cache_reuses_identical_prompt():
    fw = FakeFireworks(answer="answer", tokens=40)
    c = Cascade(fireworks_solver=fw)  # no local LLM -> factual escalates to Fireworks
    prompt = "Explain what a binary search tree is."
    first = c.solve("a", prompt)
    second = c.solve("b", prompt)
    assert first.tier == "fireworks" and first.total_tokens == 40
    assert second.tier == "cache" and second.total_tokens == 0  # reused, no new tokens
    assert second.answer == first.answer
    assert fw.called == 1  # Fireworks hit only once for the duplicate


def test_sentiment_answered_locally():
    fw = FakeFireworks()
    c = Cascade(fireworks_solver=fw)
    # Realistic framing so the router classifies it as sentiment.
    out = c.solve(
        "t",
        "Classify the sentiment of this review: 'The battery dies in an hour, "
        "very disappointing.'",
    )
    assert out.tier == "local_solver"
    assert out.answer.startswith("Negative")
    assert fw.called == 0
