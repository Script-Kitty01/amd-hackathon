"""Concurrency safety of the cascade (T12).

Runs many tasks through the cascade in a thread pool (as main.py does) and
verifies every task gets a valid, non-empty answer with no lost/duplicated ids.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from src.cascade import Cascade
from src.categories import Category


@dataclass
class FWOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int


class FakeFireworks:
    def solve(self, task_id, prompt):
        return FWOutcome(task_id, f"fw:{task_id}", Category.FACTUAL, 10)


def test_cascade_concurrent_completeness():
    cascade = Cascade(fireworks_solver=FakeFireworks())

    prompts = [
        "What is 15% of 200?",
        "Classify the sentiment of this review: 'great product, love it'",
        "Extract named entities from: 'Ada Lovelace visited London on May 1, 1843.'",
        "Explain what a binary search is.",
        "Summarise the following text in one sentence: the park opens in spring.",
        "Write a function that reverses a string.",
    ]
    tasks = [(f"t{i}", prompts[i % len(prompts)]) for i in range(60)]

    answers: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(cascade.solve, tid, p): tid for tid, p in tasks}
        for fut in as_completed(futures):
            out = fut.result()
            answers[out.task_id] = out.answer

    assert len(answers) == len(tasks)  # no lost/duplicated ids
    assert all(a and a.strip() for a in answers.values())  # every answer non-empty
