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


def test_average_of_list():
    assert solve("What is the average of 4, 8, and 12?").answer == "8"


def test_sum_of_list():
    assert solve("Find the sum of 10, 20 and 30.").answer == "60"


def test_product_of_list():
    assert solve("What is the product of 6 and 7?").answer == "42"


def test_what_percent_of():
    assert solve("What percent of 200 is 50?").answer == "25%"


def test_word_arithmetic():
    assert solve("What is 15 plus 27?").answer == "42"
    assert solve("Compute 100 divided by 4.").answer == "25"
    assert solve("What is 9 times 8?").answer == "72"


def test_divide_by_zero_abstains():
    assert solve("What is 5 divided by 0?") is None


def test_abstains_on_multistep_discount():
    # Two percentages / "then additional" -> must abstain, not answer $200.
    s = solve(
        "A $250 jacket is discounted by 20%, then an additional 10% is taken "
        "off the reduced price. What is the final price?"
    )
    assert s is None


def test_abstains_on_word_problem_without_clear_pattern():
    # No clean pattern -> must abstain so the cascade escalates.
    assert solve("If a train leaves at noon and travels 300 miles, when does it arrive?") is None


def test_abstains_on_non_math():
    assert solve("Explain what a hash table is.") is None


def test_registry_wires_math_solver():
    assert any(isinstance(s, MathSolver) for s in solvers_for(Category.MATH))
    assert solvers_for(Category.FACTUAL) == []


# --- sentiment (T22) ---

from src.local_solvers import NERSolver, SentimentSolver


def test_sentiment_negative():
    s = SentimentSolver().try_solve(
        "The battery dies in an hour, very disappointing."
    )
    assert s is not None
    assert s.answer.startswith("Negative")


def test_sentiment_positive():
    s = SentimentSolver().try_solve("I love this product, it works great and is reliable.")
    assert s is not None
    assert s.answer.startswith("Positive")


def test_sentiment_negation_flips():
    s = SentimentSolver().try_solve("This is not good at all.")
    assert s is not None
    assert s.answer.startswith("Negative")


def test_sentiment_abstains_without_signal():
    assert SentimentSolver().try_solve("The meeting is scheduled for Tuesday.") is None


def test_sentiment_abstains_on_contrastive_review():
    # "slow ... but ... delicious" is nuanced -> must abstain (escalate), not
    # confidently mislabel it Negative.
    s = SentimentSolver().try_solve(
        "Classify the sentiment: 'The service was slow, but the food was "
        "absolutely delicious.'"
    )
    assert s is None


def test_sentiment_abstains_on_mixed_signals():
    # Both positive and negative cues without a contrast word -> abstain.
    s = SentimentSolver().try_solve("Classify the sentiment: 'great screen, terrible battery'")
    assert s is None


# --- NER (T23) ---

def test_ner_extracts_person_org_date():
    s = NERSolver().try_solve(
        "Extract named entities from: Satya Nadella, CEO of Microsoft Corporation, "
        "visited Paris on March 3, 2024."
    )
    assert s is not None
    import json

    data = json.loads(s.answer)
    assert set(data.keys()) == {"person", "org", "location", "date"}
    assert any("Satya Nadella" in p for p in data["person"])
    assert any("2024" in d for d in data["date"])


def test_ner_abstains_when_nothing_found():
    assert NERSolver().try_solve("please summarise the following text quickly") is None


# --- spaCy NER (T23 upgrade) ---

import pytest

from src.local_solvers import SpacyNERSolver, _get_spacy


@pytest.mark.skipif(_get_spacy() is None, reason="spaCy en_core_web_sm not installed")
def test_spacy_ner_extracts_entities():
    import json

    s = SpacyNERSolver().try_solve(
        "Extract named entities from: Satya Nadella, CEO of Microsoft, visited Paris on March 3, 2024."
    )
    assert s is not None
    data = json.loads(s.answer)
    assert set(data.keys()) == {"person", "org", "location", "date"}
    assert any("Nadella" in p for p in data["person"])
    assert any("Microsoft" in o for o in data["org"])
    assert any("Paris" in loc for loc in data["location"])
    assert any("2024" in d for d in data["date"])


def test_ner_registry_prefers_spacy_then_heuristic():
    from src.categories import Category
    from src.local_solvers import NERSolver, solvers_for

    ner_solvers = solvers_for(Category.NER)
    assert isinstance(ner_solvers[0], SpacyNERSolver)
    assert any(isinstance(s, NERSolver) for s in ner_solvers)
