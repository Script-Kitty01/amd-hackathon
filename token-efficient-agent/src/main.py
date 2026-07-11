"""Entrypoint: read tasks -> solve concurrently -> write results -> exit.

Robustness contract (the grading harness fails the run on any crash, timeout, or
malformed output). This module is written so that NONE of those can happen:

  - We ALWAYS write a valid, complete /output/results.json (one entry per input
    task, safe fallback answers pre-filled) even if config is missing, Fireworks
    is unreachable, or individual tasks raise.
  - We respect a wall-clock budget well under the 10-minute hard cap and never
    block on stragglers: pending work is cancelled and we os._exit() so a hung
    network thread (non-daemon) can't keep the process alive past the cap.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from . import config
from .cascade import Cascade
from .categories import load_model_preference
from .config import load_config
from .fireworks_client import FireworksClient
from .io_utils import read_tasks, write_results
from .local_llm import LocalLLM
from .solver import Solver
from .thresholds import load_thresholds

MAX_WORKERS = 8
_FALLBACK = "Unable to produce an answer."


def _build_cascade() -> Cascade | None:
    """Construct the solving cascade, or None if config is unavailable.

    A missing/invalid config must NOT crash the run: we still emit a complete
    results file (all fallback answers) and exit cleanly.
    """
    try:
        cfg = load_config()
    except Exception:
        return None
    try:
        load_model_preference()  # overlay launch-day sweep output if present
        fireworks_solver = Solver(cfg, FireworksClient(cfg))
        return Cascade(
            thresholds=load_thresholds(),
            fireworks_solver=fireworks_solver,
            local_llm=LocalLLM.from_env(),  # None unless a local model is configured
        )
    except Exception:
        return None


def run() -> None:
    start = time.monotonic()
    tasks = read_tasks(config.INPUT_PATH)

    # Preserve input order; default every task to a safe fallback answer so the
    # results file is always complete regardless of what happens below.
    answers: dict[str, str] = {t.task_id: _FALLBACK for t in tasks}

    try:
        cascade = _build_cascade()
        if cascade is not None and tasks:
            _solve_all(cascade, tasks, answers, start)
    finally:
        # Always write results, even if solving was interrupted or errored.
        results = [{"task_id": t.task_id, "answer": answers[t.task_id]} for t in tasks]
        write_results(config.OUTPUT_PATH, results)


def _solve_all(cascade: Cascade, tasks, answers: dict[str, str], start: float) -> None:
    """Solve tasks concurrently within the wall-clock budget.

    Uses an explicit deadline and `wait(..., timeout=...)` so we stop collecting
    the moment the budget is hit, then cancel outstanding work. Unfinished tasks
    keep their pre-filled fallback answer.
    """
    budget = config.RUNTIME_BUDGET_SECONDS
    deadline = start + budget
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        futures = {pool.submit(cascade.solve, t.task_id, t.prompt): t.task_id for t in tasks}
        pending = set(futures)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break  # budget exhausted
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    outcome = fut.result()
                    answers[outcome.task_id] = outcome.answer
                except Exception:
                    pass  # keep the fallback answer for this task
    finally:
        # Drop queued work; do not wait for in-flight network calls.
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    try:
        run()
    except Exception:  # last-resort guard: try to leave a valid (empty) results file
        traceback.print_exc()
        try:
            write_results(config.OUTPUT_PATH, [])
        except Exception:
            pass
    finally:
        # Flush logs, then force-exit. os._exit bypasses waiting on any non-daemon
        # worker thread still blocked in a network call, guaranteeing we terminate
        # (exit 0) with results.json already on disk.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
