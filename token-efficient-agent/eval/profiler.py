"""Lightweight token profiler for evaluation runs."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CallStats:
    task_id: str
    category: str
    model: str
    prompt: int
    completion: int
    total: int


class TokenProfiler:
    def __init__(self) -> None:
        self.is_active = os.environ.get("DEBUG_TOKEN_STATS", "").lower() == "true"
        self.calls: list[CallStats] = []
        self.retries = 0

    def record(self, **kwargs) -> None:
        if not self.is_active:
            return
        self.calls.append(CallStats(**kwargs))

    def record_retry(self) -> None:
        if not self.is_active:
            return
        self.retries += 1

    def print_report(self) -> None:
        if not self.is_active or not self.calls:
            return

        cat_totals: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        model_totals: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        
        total_prompt = 0
        total_completion = 0

        for call in self.calls:
            cat_totals[call.category]["prompt"] += call.prompt
            cat_totals[call.category]["completion"] += call.completion
            cat_totals[call.category]["total"] += call.total
            cat_totals[call.category]["count"] += 1

            model_totals[call.model]["prompt"] += call.prompt
            model_totals[call.model]["completion"] += call.completion
            model_totals[call.model]["total"] += call.total
            model_totals[call.model]["count"] += 1
            
            total_prompt += call.prompt
            total_completion += call.completion

        print("\n--- Token Profiler Report ---")

        print("\nCategory Totals:")
        for cat, totals in sorted(cat_totals.items()):
            print(
                f"  {cat:<16} "
                f"prompt={totals['prompt']:<6} "
                f"completion={totals['completion']:<6} "
                f"total={totals['total']:<6}"
            )

        print("\nModel Totals:")
        for model, totals in sorted(model_totals.items()):
            print(
                f"  {model:<48} "
                f"prompt={totals['prompt']:<6} "
                f"completion={totals['completion']:<6} "
                f"total={totals['total']:<6}"
            )

        print(f"\nRetry count: {self.retries}")

        print("\nAverage Tokens per Category:")
        for cat, totals in sorted(cat_totals.items()):
            count = totals['count']
            avg_prompt = totals['prompt'] / count
            avg_completion = totals['completion'] / count
            print(
                f"  {cat:<16} "
                f"prompt={avg_prompt:<6.1f} "
                f"completion={avg_completion:<6.1f}"
            )
        
        if self.calls:
            print(f"\nAverage prompt tokens: {total_prompt / len(self.calls):.1f}")
            print(f"Average completion tokens: {total_completion / len(self.calls):.1f}")
        
        print("---------------------------\n")