"""Run the agent over a local dataset and report accuracy/tokens.

Usage:
    python -m eval.run_eval [path/to/dataset.json]            # single run
    python -m eval.run_eval --sweep [path/to/dataset.json]    # model sweep (T8)

The sweep runs every ALLOWED_MODEL against every task, prints an accuracy/token
table per (category, model), and writes a recommended MODEL_PREFERENCE mapping
to config/model_preference.json. On ties it prefers a gemma model to capture the
"best use of gemma 4" bonus (T19).

Dataset format:
    [{ "task_id": "e1", "prompt": "...", "expected": "..." }, ...]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import time

from src.cascade import Cascade
from src.categories import Category
from src.config import load_config
from src.fireworks_client import FireworksClient
from src.local_llm import LocalLLM
from src.prompts import spec_for
from src.router import classify
from src.solver import Solver
from src.thresholds import load_thresholds

from .judge import Judge
from .scorer import EvalRecord, EvalReport, check_match

DEFAULT_DATASET = "eval/datasets/sample_tasks.json"
PREF_OUTPUT = "config/model_preference.json"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader for local dev (not used in the shipped image)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _build_fireworks_solver() -> Solver | None:
    """Build the Fireworks/fallback solver if its env is configured, else None."""
    try:
        cfg = load_config()
    except KeyError:
        return None
    return Solver(cfg, FireworksClient(cfg))


# --- single run ------------------------------------------------------------

def run_single(dataset_path: str) -> None:
    _load_dotenv()
    items = _load(dataset_path)

    fireworks_solver = _build_fireworks_solver()
    local_llm = LocalLLM.from_env()
    judge = Judge.from_env()

    cascade = Cascade(
        thresholds=load_thresholds(),
        fireworks_solver=fireworks_solver,
        local_llm=local_llm,
    )

    print(
        f"config: local_llm={'on' if local_llm else 'off'} "
        f"fallback={'on' if fireworks_solver else 'off'} "
        f"judge={'on' if judge else 'off (substring match)'}"
    )

    report = EvalReport()
    start = time.time()
    for item in items:
        prompt = str(item["prompt"])
        outcome = cascade.solve(str(item["task_id"]), prompt)
        expected = item.get("expected")
        if judge is not None:
            passed = judge.passed(prompt, outcome.answer, expected)
        else:
            passed = check_match(outcome.answer, expected)
        report.records.append(
            EvalRecord(
                task_id=outcome.task_id,
                category=outcome.category.value,
                answer=outcome.answer,
                total_tokens=outcome.total_tokens,
                passed=passed,
                tier=outcome.tier,
            )
        )
        print(f"  {outcome.task_id:<22} {outcome.tier:<12} "
              f"{'PASS' if passed else 'FAIL'}  tokens={outcome.total_tokens}")

    elapsed = time.time() - start
    print("\n" + report.summary())
    print(f"wall time:    {elapsed:.1f}s")
    if judge is not None:
        print(f"judge tokens: {judge.total_tokens} (not counted toward agent score)")


# --- model sweep (T8) ------------------------------------------------------

def _is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


def run_sweep(dataset_path: str) -> None:
    items = _load(dataset_path)
    cfg = load_config()
    client = FireworksClient(cfg)

    # stats[(category, model)] = {"pass": int, "n": int, "tokens": int}
    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"pass": 0, "n": 0, "tokens": 0}
    )

    for model in cfg.models:
        for item in items:
            prompt = str(item["prompt"])
            category = classify(prompt)
            spec = spec_for(category)
            try:
                result = client.complete(
                    model=model, system=spec.system, user=prompt, max_tokens=spec.max_tokens
                )
                answer, tokens = result.text, result.total_tokens
            except Exception:
                answer, tokens = "", 0
            passed = check_match(answer, item.get("expected"))
            s = stats[(category.value, model)]
            s["n"] += 1
            s["tokens"] += tokens
            if passed:
                s["pass"] += 1

    _print_sweep_table(cfg.models, stats)
    recommendation = _recommend(cfg.models, stats)
    _write_recommendation(cfg.models, recommendation)


def _print_sweep_table(models: list[str], stats: dict[tuple[str, str], dict[str, int]]) -> None:
    print("\n=== model sweep: pass-rate / total-tokens per category ===")
    for cat in Category:
        rows = [(m, stats.get((cat.value, m))) for m in models]
        if not any(r[1] for r in rows):
            continue
        print(f"\n{cat.value}")
        for model, s in rows:
            if not s or s["n"] == 0:
                continue
            rate = s["pass"] / s["n"]
            print(f"  {model:<48} pass={rate:>5.0%}  tokens={s['tokens']}")


def _recommend(
    models: list[str], stats: dict[tuple[str, str], dict[str, int]]
) -> dict[Category, int]:
    """Per category, pick the model index with the best pass-rate, then fewest
    tokens. Prefer a gemma model when it is within 10% tokens of the best (T19).
    """
    recommendation: dict[Category, int] = {}
    for cat in Category:
        candidates = []
        for idx, model in enumerate(models):
            s = stats.get((cat.value, model))
            if not s or s["n"] == 0:
                continue
            rate = s["pass"] / s["n"]
            candidates.append((idx, model, rate, s["tokens"]))
        if not candidates:
            continue

        best_rate = max(c[2] for c in candidates)
        passers = [c for c in candidates if c[2] == best_rate]
        best_tokens = min(c[3] for c in passers)

        # Gemma bonus: prefer gemma among passers if within 10% of best tokens.
        gemma = [c for c in passers if _is_gemma(c[1]) and c[3] <= best_tokens * 1.10]
        chosen = min(gemma, key=lambda c: c[3]) if gemma else min(passers, key=lambda c: c[3])
        recommendation[cat] = chosen[0]
    return recommendation


def _write_recommendation(models: list[str], recommendation: dict[Category, int]) -> None:
    os.makedirs(os.path.dirname(PREF_OUTPUT) or ".", exist_ok=True)
    payload = {cat.value: idx for cat, idx in recommendation.items()}
    with open(PREF_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nrecommended MODEL_PREFERENCE written to {PREF_OUTPUT}:")
    for cat, idx in recommendation.items():
        print(f"  {cat.value:<16} -> [{idx}] {models[idx]}")


# --- shared ----------------------------------------------------------------

def _load(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--sweep":
        dataset = args[1] if len(args) > 1 else DEFAULT_DATASET
        run_sweep(dataset)
    else:
        dataset = args[0] if args else DEFAULT_DATASET
        run_single(dataset)


if __name__ == "__main__":
    main()
