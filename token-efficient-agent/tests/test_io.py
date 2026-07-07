"""Tests for task/result I/O."""

import json

from src.io_utils import read_tasks, write_results


def test_read_tasks(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([{"task_id": "t1", "prompt": "hello"}]), encoding="utf-8")
    tasks = read_tasks(str(p))
    assert len(tasks) == 1
    assert tasks[0].task_id == "t1"
    assert tasks[0].prompt == "hello"


def test_write_results_valid_json(tmp_path):
    p = tmp_path / "out" / "results.json"
    write_results(str(p), [{"task_id": "t1", "answer": "hi"}])
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data == [{"task_id": "t1", "answer": "hi"}]
