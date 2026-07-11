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


# --- hardening (T11) ---

def test_read_tasks_missing_file_returns_empty():
    assert read_tasks("does/not/exist.json") == []


def test_read_tasks_skips_malformed_items(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(
        json.dumps(
            [
                {"task_id": "t1", "prompt": "ok"},
                "not-a-dict",
                {"prompt": "no id here"},          # missing task_id -> index used
                {"task_id": "t4"},                 # missing prompt -> ""
            ]
        ),
        encoding="utf-8",
    )
    tasks = read_tasks(str(p))
    ids = [t.task_id for t in tasks]
    assert "t1" in ids and "t4" in ids
    assert all(isinstance(t.prompt, str) for t in tasks)


def test_read_tasks_non_list_returns_empty(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps({"oops": 1}), encoding="utf-8")
    assert read_tasks(str(p)) == []


def test_read_tasks_unicode(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps([{"task_id": "t1", "prompt": "café ☕ 日本語"}]), encoding="utf-8")
    tasks = read_tasks(str(p))
    assert tasks[0].prompt == "café ☕ 日本語"
