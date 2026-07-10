"""End-to-end pipeline test (offline): read -> cascade -> write -> validate.

Mirrors what src.main does, using a fake Fireworks fallback so every task is
answered without network. Proves results.json is always valid and complete.
"""

import json
from dataclasses import dataclass

from src.cascade import Cascade
from src.categories import Category
from src.io_utils import read_tasks, write_results


@dataclass
class FWOutcome:
    task_id: str
    answer: str
    category: Category
    total_tokens: int


class FakeFireworks:
    def solve(self, task_id, prompt):
        return FWOutcome(task_id, f"answer-for-{task_id}", Category.FACTUAL, 20)


def test_full_pipeline_writes_valid_complete_results(tmp_path):
    # Write a small tasks file (mirrors /input/tasks.json).
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(
        json.dumps(
            [
                {"task_id": "t1", "prompt": "What is 15% of 200?"},
                {
                    "task_id": "t2",
                    "prompt": "Classify the sentiment of this review: 'love it, works great'",
                },
                {"task_id": "t3", "prompt": "Explain what recursion is."},
            ]
        ),
        encoding="utf-8",
    )

    tasks = read_tasks(str(tasks_file))
    cascade = Cascade(fireworks_solver=FakeFireworks())

    answers = {t.task_id: "Unable to produce an answer." for t in tasks}
    for t in tasks:
        answers[t.task_id] = cascade.solve(t.task_id, t.prompt).answer

    results = [{"task_id": t.task_id, "answer": answers[t.task_id]} for t in tasks]
    out_file = tmp_path / "out" / "results.json"
    write_results(str(out_file), results)

    # Validate: parseable, one entry per task, correct keys, non-empty answers.
    reloaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(reloaded) == 3
    assert [r["task_id"] for r in reloaded] == ["t1", "t2", "t3"]
    assert all(set(r.keys()) == {"task_id", "answer"} for r in reloaded)
    assert all(r["answer"].strip() for r in reloaded)
    # t1 solved locally (zero tokens): should be the exact math value.
    assert reloaded[0]["answer"] == "30"


def test_real_input_file_parses():
    # The committed sample input mirror should read cleanly.
    tasks = read_tasks("data/input/tasks.json")
    assert len(tasks) >= 1
    assert all(t.task_id and isinstance(t.prompt, str) for t in tasks)
