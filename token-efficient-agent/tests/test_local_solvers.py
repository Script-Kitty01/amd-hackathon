"""Tests for the local answering layer (T20 interface, T21 math solver)."""

from src.categories import Category
from src.local_solvers import MathSolver, Solution, solvers_for


def solve(prompt):
    return MathSolver().try_solve(prompt)


def test_percentage_discount_on_price():
    s = solve("Calculate the final price of a $120 item after a 15% discount.")
    assert isinstance(s, Solution)
    assert s.answer == "$102.00"
    assert s.confidence >= 0.9


def test_discount_word_variant():
    s = solve("A shirt costs $40 and is discounted by 25%. What is the final price?")
    assert s is not None
    assert s.answer == "$30.00"


def test_percent_of():
    s = solve("What is 25% of 400?")
    assert s is not None
    assert s.answer == "100"


def test_percentage_increase():
    s = solve("A $200 fee is increased by 10%. What is the new amount?")
    assert s is not None
    assert s.answer == "$220.00"


def test_bare_arithmetic_expression():
    s = solve("12 * (3 + 4)")
    assert s is not None
    assert s.answer == "84"


def test_abstains_on_word_problem_without_clear_pattern():
    # No clean pattern -> must abstain so the cascade escalates.
    assert solve("If a train leaves at noon and travels 300 miles, when does it arrive?") is None


def test_abstains_on_non_math():
    assert solve("Explain what a hash table is.") is None


def test_registry_wires_math_solver():
    assert any(isinstance(s, MathSolver) for s in solvers_for(Category.MATH))
    assert solvers_for(Category.FACTUAL) == []
