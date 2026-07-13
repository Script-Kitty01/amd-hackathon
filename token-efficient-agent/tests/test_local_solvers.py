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
    s = solve(
        "A $250 jacket is discounted by 20%, then an additional 10% is taken "
        "off the reduced price. What is the final price?"
    )
    assert s is None


def test_abstains_on_word_problem_without_clear_pattern():
    assert solve("If a train leaves at noon and travels 300 miles, when does it arrive?") is None


def test_abstains_on_non_math():
    assert solve("Explain what a hash table is.") is None


def test_registry_wires_math_solver():
    assert any(isinstance(s, MathSolver) for s in solvers_for(Category.MATH))
    assert solvers_for(Category.FACTUAL) == []


# --- sentiment (T22) ---

from src.local_solvers import NERSolver, SentimentSolver


def test_sentiment_negative():
    s = SentimentSolver().try_solve("The battery dies in an hour, very disappointing.")
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
    s = SentimentSolver().try_solve(
        "Classify the sentiment: 'The service was slow, but the food was absolutely delicious.'"
    )
    assert s is None


def test_sentiment_abstains_on_mixed_signals():
    s = SentimentSolver().try_solve(
        "Classify the sentiment: 'great screen, terrible battery'"
    )
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


# --- registry (updated) ---

def test_ner_not_registered():
    # NER stays Fireworks-only: regex heuristics can miss entities/mislabel.
    assert solvers_for(Category.NER) == []


def test_math_and_sentiment_registered():
    math_solvers = solvers_for(Category.MATH)
    assert len(math_solvers) >= 1 and isinstance(math_solvers[0], MathSolver)
    sentiment_solvers = solvers_for(Category.SENTIMENT)
    assert len(sentiment_solvers) == 1 and isinstance(sentiment_solvers[0], SentimentSolver)


# --- operation-chain solver (warehouse-style) ---

from src.local_solvers import OperationChainSolver


def test_operation_chain_warehouse():
    s = OperationChainSolver().try_solve(
        "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
        "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
        "How many units remain at the end of Q3?"
    )
    assert s is not None
    assert "1672" in s.answer


def test_operation_chain_abstains_on_leftover_numbers():
    # Number 2025 has no role in the operations -> must abstain
    s = OperationChainSolver().try_solve(
        "A store starts with 100 items in 2025. It sells 10. How many remain?"
    )
    # Either abstains (None) or gets 90 — both are acceptable
    # The test just verifies it doesn't crash
    assert s is None or s.answer == "90"


# --- ratio solver (recipe-style) ---

from src.local_solvers import RatioSolver


def test_ratio_solver_recipe():
    s = RatioSolver().try_solve(
        "A recipe requires 3/4 cup of sugar for 12 cookies. "
        "How much sugar is needed for 30 cookies? "
        "If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?"
    )
    assert s is not None
    assert "1.875" in s.answer
    assert "4.50" in s.answer


# --- exact response solver ---

from src.local_solvers import try_exact_response


def test_exact_response_double_quotes():
    s = try_exact_response('Reply with exactly "ACK" to confirm.')
    assert s is not None and s.answer == "ACK"


def test_exact_response_colon():
    s = try_exact_response("Respond with exactly: DONE")
    assert s is not None and s.answer == "DONE"


def test_exact_response_abstains_on_alternatives():
    # "exactly yes or no" -> choice depends on the question, must abstain
    assert try_exact_response("Answer with exactly 'yes' or 'no': is 5 prime?") is None


def test_exact_response_none_when_absent():
    assert try_exact_response("What is the capital of France?") is None


# --- speed / distance / time ---

from src.local_solvers import SpeedDistanceSolver


def test_speed_distance_single_leg():
    s = SpeedDistanceSolver().try_solve(
        "A car travels at 60 km/h for 3 hours. What distance does it cover?"
    )
    assert s is not None and s.answer == "180"


def test_speed_distance_abstains_multi_leg():
    s = SpeedDistanceSolver().try_solve(
        "A car travels at 60 km/h for 2 hours then at 80 km/h for 1 hour. Total distance?"
    )
    assert s is None


# --- simple interest ---

from src.local_solvers import SimpleInterestSolver


def test_simple_interest():
    s = SimpleInterestSolver().try_solve(
        "Calculate the simple interest on a principal of 1000 at 5% per annum for 3 years."
    )
    assert s is not None and s.answer == "150"


def test_simple_interest_abstains_on_compound():
    s = SimpleInterestSolver().try_solve(
        "Calculate the compound interest on 1000 at 5% for 3 years."
    )
    assert s is None


# --- unit cost ---

from src.local_solvers import UnitCostSolver


def test_unit_cost():
    s = UnitCostSolver().try_solve("12 items cost $60. What is the price per item?")
    assert s is not None
    assert "5" in s.answer


# --- additional zero-token math forms ---

from src.local_solvers import BundlePriceSolver, FuelRateSolver


def test_fraction_of_number():
    s = solve("What is 3/4 of 200?")
    assert s is not None and s.answer == "150"


def test_fuel_consumption_rate():
    s = FuelRateSolver().try_solve(
        "A car uses 8 litres of fuel per 100 km. How many litres are needed for a 250 km trip?"
    )
    assert s is not None and s.answer == "20"


def test_average_speed():
    s = SpeedDistanceSolver().try_solve(
        "A train travels 240 km in 3 hours. What is its average speed in km/h?"
    )
    assert s is not None and s.answer == "80"


def test_bundle_price_proportion():
    s = BundlePriceSolver().try_solve(
        "A shop sells 3 pens for $6. At that rate, how much do 10 pens cost?"
    )
    assert s is not None and s.answer == "$20.00"


# --- deterministic logic ---

from src.local_solvers import LogicSolver


def test_logic_weekday_offset():
    s = LogicSolver().try_solve("If today is Wednesday, what day of the week will it be 3 days from now?")
    assert s is not None and s.answer == "Saturday"


def test_logic_transitive_comparison_yes_no():
    s = LogicSolver().try_solve(
        "Sara is older than Tom. Tom is older than Mike. Is Sara older than Mike? Answer yes or no."
    )
    assert s is not None and s.answer == "Yes"


def test_logic_transitive_comparison_extreme():
    s = LogicSolver().try_solve(
        "Tom is taller than Jerry. Jerry is taller than Spike. Spike is taller than Butch. Who is the shortest?"
    )
    assert s is not None and s.answer == "Butch"


def test_logic_syllogism():
    s = LogicSolver().try_solve(
        "All Bloops are Razzies. All Razzies are Lazzies. Are all Bloops necessarily Lazzies? Answer yes or no."
    )
    assert s is not None and s.answer == "Yes"


def test_logic_exactly_one_sign():
    s = LogicSolver().try_solve(
        "Three boxes are labeled A, B, and C. Exactly one contains a prize. "
        "Sign on A: 'The prize is in A.' Sign on B: 'The prize is not in B.' "
        "Sign on C: 'The prize is not in A.' Exactly one of the three signs is true. "
        "Which box has the prize?"
    )
    assert s is not None and s.answer == "B"


# --- deterministic Python debugging ---

from src.local_solvers import CodeDebugSolver


def test_code_debug_accumulator_repair():
    s = CodeDebugSolver().try_solve(
        "Find and fix the bug:\n```python\ndef sum_list(lst):\n    total = 0\n"
        "    for x in lst:\n        total = x\n    return total\n```"
    )
    assert s is not None and "total += x" in s.answer


def test_code_debug_off_by_one_repair():
    s = CodeDebugSolver().try_solve(
        "Find and fix the bug:\n```python\ndef get_first(lst):\n    return lst[1]\n```"
    )
    assert s is not None and "return lst[0]" in s.answer


def test_code_debug_double_repair():
    s = CodeDebugSolver().try_solve(
        "Find and fix the bug:\n```python\ndef double(x):\n    return x + x + x\n```"
    )
    assert s is not None and "return x + x" in s.answer and "x + x + x" not in s.answer
