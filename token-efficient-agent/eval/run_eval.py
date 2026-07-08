"""Run the agent over a local dataset and print an accuracy/token report.

Usage:
    python -m eval.run_eval [path/to/dataset.json]

Dataset format (expected optional for scoring):
    [{ "task_id": "e1", "prompt": "...", "expected": "..." }, ...]
"""

from __future__ import annotations

import json
import sys

from src.config import load_config
from src.google_client import GoogleClient
from src.router import classify
from src.solver import Solver

from .scorer import EvalRecord, EvalReport, check_match

DEFAULT_DATASET = "eval/datasets/sample_tasks.json"


def main() -> None:
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    cfg = load_config()
    client = GoogleClient(cfg)
    solver = Solver(cfg, client)
    report = EvalReport()

    for item in items:
        outcome = solver.solve(str(item["task_id"]), str(item["prompt"]))
        report.records.append(
            EvalRecord(
                task_id=outcome.task_id,
                category=classify(item["prompt"]).value,
                answer=outcome.answer,
                total_tokens=outcome.total_tokens,
                passed=check_match(outcome.answer, item.get("expected")),
            )
        )

    print(report.summary())


if __name__ == "__main__":
    main()
