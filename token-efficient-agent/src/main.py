"""Entrypoint: read tasks -> solve concurrently -> write results -> exit.

Robustness contract (the grading harness kills or fails the run on any crash):
  - We ALWAYS write a valid, complete /output/results.json and exit 0, even if
    config is missing, Fireworks is unreachable, or individual tasks fail.
  - We respect a wall-clock budget well under the 10-minute hard cap and never
    block on stragglers: pending work is cancelled so the process can exit.
"""

from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    results file (all fallback answers) and exit 0.
    """
    try:
        cfg = load_config()
    except Exception:
        return None
    load_model_preference()  # overlay launch-day sweep output if present
    fireworks_solver = Solver(cfg, FireworksClient(cfg))
    return Cascade(
        thresholds=load_thresholds(),
        fireworks_solver=fireworks_solver,
        local_llm=LocalLLM.from_env(),  # None unless a local model is configured
    )


def run() -> int:
    start = time.monotonic()
    tasks = read_tasks(config.INPUT_PATH)

    # Preserve input order; default every task to a safe fallback answer so the
    # results file is always complete regardless of what happens below.
    answers: dict[str, str] = {t.task_id: _FALLBACK for t in tasks}

    cascade = _build_cascade()
    if cascade is not None and tasks:
        _solve_all(cascade, tasks, answers, start)

    results = [{"task_id": t.task_id, "answer": answers[t.task_id]} for t in tasks]
    write_results(config.OUTPUT_PATH, results)
    return 0


def _solve_all(cascade: Cascade, tasks, answers: dict[str, str], start: float) -> None:
    """Solve tasks concurrently within the wall-clock budget.

    On budget exhaustion we cancel outstanding work (never block on shutdown) so
    the process can write results and exit before the harness's hard cap.
    """
    budget = config.RUNTIME_BUDGET_SECONDS
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        futures = {pool.submit(cascade.solve, t.task_id, t.prompt): t.task_id for t in tasks}
        for fut in as_completed(futures):
            remaining = budget - (time.monotonic() - start)
            if remaining <= 0:
                break
            try:
                outcome = fut.result(timeout=max(0.1, remaining))
                answers[outcome.task_id] = outcome.answer
            except Exception:
                pass  # keep the fallback answer for this task
    finally:
        # Drop any queued/running work; do not wait for slow network calls.
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    try:
        code = run()
    except Exception:  # last-resort guard: still try to leave a valid results file
        traceback.print_exc()
        try:
            write_results(config.OUTPUT_PATH, [])
        except Exception:
            pass
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
