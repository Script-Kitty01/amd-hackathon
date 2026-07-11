"""Tests for the local LLM tier (T24). Uses a fake complete_fn (no network)."""

from src.categories import Category
from src.local_llm import LocalLLM


def const_fn(answer):
    return lambda system, user, max_tokens, temperature: answer


def test_single_sample_uses_prior_confidence():
    llm = LocalLLM("gemma", const_fn("Paris"), samples=1)
    s = llm.try_solve(Category.FACTUAL, "What is the capital of France?")
    assert s is not None
    assert s.answer == "Paris"
    assert s.confidence == 0.70  # factual prior


def test_empty_answer_abstains():
    llm = LocalLLM("gemma", const_fn("   "), samples=1)
    assert llm.try_solve(Category.FACTUAL, "anything") is None


def test_exception_abstains():
    def boom(system, user, max_tokens, temperature):
        raise RuntimeError("model down")

    llm = LocalLLM("gemma", boom, samples=1)
    assert llm.try_solve(Category.FACTUAL, "anything") is None


def test_self_consistency_full_agreement_boosts_confidence():
    # 3 identical answers -> agreement 1.0 -> confidence == prior * (0.5 + 0.5)
    llm = LocalLLM("gemma", const_fn("42"), samples=3)
    s = llm.try_solve(Category.MATH, "6 * 7")
    assert s is not None
    assert s.answer == "42"
    assert s.confidence == 0.5  # math prior 0.5 * 1.0


def test_self_consistency_disagreement_lowers_confidence():
    # Alternating answers -> majority agreement 2/3 -> lower confidence.
    seq = iter(["42", "43", "42"])
    fn = lambda system, user, max_tokens, temperature: next(seq)
    llm = LocalLLM("gemma", fn, samples=3)
    s = llm.try_solve(Category.MATH, "6 * 7")
    assert s is not None
    assert s.answer == "42"  # majority
    assert s.confidence < 0.5  # penalized for disagreement


def test_from_env_disabled_without_config(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    assert LocalLLM.from_env() is None
