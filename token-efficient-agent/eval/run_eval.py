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

from src.categories import Category
from src.config import load_config
from src.fireworks_client import FireworksClient
from src.prompts import spec_for
from src.router import classify
from src.solver import Solver

from .scorer import EvalRecord, EvalReport, check_match

DEFAULT_DATASET = "eval/datasets/sample_tasks.json"
PREF_OUTPUT = "config/model_preference.json"


# --- single run ------------------------------------------------------------

def run_single(dataset_path: str) -> None:
    items = _load(dataset_path)
    cfg = load_config()
    solver = Solver(cfg, FireworksClient(cfg))
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
