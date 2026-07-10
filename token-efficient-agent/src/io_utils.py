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
    """Read tasks tolerantly: skip non-dict items, default missing fields.

    Never raises on malformed content — returns whatever valid tasks it can so
    the run proceeds and still emits a complete results.json.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []

    tasks: list[Task] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id", i)
        prompt = item.get("prompt", "")
        try:
            tasks.append(Task(task_id=str(task_id), prompt=str(prompt)))
        except Exception:
            continue
    return tasks


def write_results(path: str, results: list[dict]) -> None:
    """Atomically write valid JSON results."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(tmp, path)
