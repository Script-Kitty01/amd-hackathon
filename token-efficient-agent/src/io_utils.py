"""Read tasks and write results with strict, valid-JSON guarantees."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str


def read_tasks(path: str) -> list[Task]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks = []
    for item in raw:
        tasks.append(Task(task_id=str(item["task_id"]), prompt=str(item["prompt"])))
    return tasks


def write_results(path: str, results: list[dict]) -> None:
    """Atomically write valid JSON results."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)
