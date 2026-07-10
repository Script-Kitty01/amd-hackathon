"""Entrypoint: read tasks -> solve concurrently -> write results -> exit.

Exit code 0 on success, non-zero on failure. Respects a wall-clock budget so
we always write whatever we have before the 10-minute hard cap.
"""

from __future__ import annotations

import sys
import time
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


def run() -> int:
    start = time.monotonic()
    cfg = load_config()
    load_model_preference()  # overlay launch-day sweep output if present
    tasks = read_tasks(config.INPUT_PATH)

    # Build the local-first cascade: local solvers -> local LLM -> Fireworks.
    fireworks_solver = Solver(cfg, FireworksClient(cfg))
    cascade = Cascade(
        thresholds=load_thresholds(),
        fireworks_solver=fireworks_solver,
        local_llm=LocalLLM.from_env(),  # None if no local model configured
    )

    # Preserve input order; default every task to a safe fallback answer.
    answers: dict[str, str] = {t.task_id: "Unable to produce an answer." for t in tasks}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(cascade.solve, t.task_id, t.prompt): t.task_id for t in tasks}
        for fut in as_completed(futures):
            if time.monotonic() - start > config.RUNTIME_BUDGET_SECONDS:
                break
            try:
                outcome = fut.result()
                answers[outcome.task_id] = outcome.answer
            except Exception:
                pass  # keep the fallback answer

    results = [{"task_id": t.task_id, "answer": answers[t.task_id]} for t in tasks]
    write_results(config.OUTPUT_PATH, results)
    return 0


def main() -> None:
    try:
        sys.exit(run())
    except Exception as exc:  # noqa: BLE001
        print(f"fatal: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
