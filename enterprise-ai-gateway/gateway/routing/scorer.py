"""Multi-factor model scoring engine.

Scores every available model across:
  - Cost (lower = better)
  - Expected quality (per category)
  - Latency (estimated)
  - Current availability (healthy/unhealthy)
  - Department policy (allowed/blocked)
  - Remaining budget (department-level)
  - User priority tier

Score = w1*CostScore + w2*QualityScore + w3*LatencyScore + w4*AvailabilityScore
        + w5*PolicyScore + w6*BudgetScore + w7*PriorityScore

All scores are normalized 0..1 where 1 = best.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..providers.registry import ProviderRegistry


# --- Category definitions (from the hackathon router) ---

class TaskCategory:
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    LOGIC = "logic"
    CODE_GEN = "code_gen"
    GENERAL = "general"


# Per-category model quality scores (0..1, higher = better at this category).
# These are approximate — in production, calibrate from eval runs.
# Based on known benchmark strengths of common models.
CATEGORY_QUALITY: dict[str, dict[str, float]] = {
    # model_name -> {category: quality_score}
    "gpt-4o": {
        TaskCategory.FACTUAL: 0.92, TaskCategory.MATH: 0.88,
        TaskCategory.SENTIMENT: 0.90, TaskCategory.SUMMARIZATION: 0.91,
        TaskCategory.NER: 0.93, TaskCategory.CODE_DEBUG: 0.90,
        TaskCategory.LOGIC: 0.89, TaskCategory.CODE_GEN: 0.91,
        TaskCategory.GENERAL: 0.90,
    },
    "gpt-4o-mini": {
        TaskCategory.FACTUAL: 0.85, TaskCategory.MATH: 0.78,
        TaskCategory.SENTIMENT: 0.84, TaskCategory.SUMMARIZATION: 0.86,
        TaskCategory.NER: 0.87, TaskCategory.CODE_DEBUG: 0.80,
        TaskCategory.LOGIC: 0.76, TaskCategory.CODE_GEN: 0.82,
        TaskCategory.GENERAL: 0.83,
    },
    "claude-sonnet-4-20250514": {
        TaskCategory.FACTUAL: 0.93, TaskCategory.MATH: 0.90,
        TaskCategory.SENTIMENT: 0.91, TaskCategory.SUMMARIZATION: 0.94,
        TaskCategory.NER: 0.92, TaskCategory.CODE_DEBUG: 0.93,
        TaskCategory.LOGIC: 0.91, TaskCategory.CODE_GEN: 0.94,
        TaskCategory.GENERAL: 0.92,
    },
    "claude-3-5-haiku-20241022": {
        TaskCategory.FACTUAL: 0.84, TaskCategory.MATH: 0.76,
        TaskCategory.SENTIMENT: 0.83, TaskCategory.SUMMARIZATION: 0.85,
        TaskCategory.NER: 0.86, TaskCategory.CODE_DEBUG: 0.82,
        TaskCategory.LOGIC: 0.74, TaskCategory.CODE_GEN: 0.83,
        TaskCategory.GENERAL: 0.82,
    },
    "gemini-2.0-flash": {
        TaskCategory.FACTUAL: 0.86, TaskCategory.MATH: 0.82,
        TaskCategory.SENTIMENT: 0.85, TaskCategory.SUMMARIZATION: 0.87,
        TaskCategory.NER: 0.88, TaskCategory.CODE_DEBUG: 0.84,
        TaskCategory.LOGIC: 0.80, TaskCategory.CODE_GEN: 0.85,
        TaskCategory.GENERAL: 0.85,
    },
    "gemma-4-31b-it": {
        TaskCategory.FACTUAL: 0.82, TaskCategory.MATH: 0.78,
        TaskCategory.SENTIMENT: 0.81, TaskCategory.SUMMARIZATION: 0.83,
        TaskCategory.NER: 0.84, TaskCategory.CODE_DEBUG: 0.78,
        TaskCategory.LOGIC: 0.74, TaskCategory.CODE_GEN: 0.80,
        TaskCategory.GENERAL: 0.80,
    },
}

# Default quality for unknown models
_DEFAULT_QUALITY = 0.75

# Estimated latency in ms (approximate, for scoring)
_ESTIMATED_LATENCY: dict[str, float] = {
    "gpt-4o": 800, "gpt-4o-mini": 400,
    "claude-sonnet-4-20250514": 900, "claude-3-5-haiku-20241022": 350,
    "gemini-2.0-flash": 300, "gemma-4-31b-it": 500,
}
_DEFAULT_LATENCY = 600


@dataclass
class ModelScore:
    """Scored model with breakdown."""
    model: str
    provider: str
    total_score: float
    cost_score: float
    quality_score: float
    latency_score: float
    availability_score: float
    policy_score: float
    budget_score: float
    priority_score: float
    estimated_cost_usd: float
    estimated_latency_ms: float


@dataclass
class RoutingContext:
    """Context passed to the scorer for each request."""
    category: str = TaskCategory.GENERAL
    department: str = "default"
    user_priority: int = 0  # 0=normal, 1=high, 2=critical
    remaining_budget_usd: float = 1000.0
    monthly_budget_usd: float = 1000.0
    blocked_models: list[str] = field(default_factory=list)
    preferred_providers: list[str] = field(default_factory=list)


class ModelScorer:
    """Scores every available model across 7 dimensions."""

    # Default weights (tunable per deployment)
    DEFAULT_WEIGHTS = {
        "cost": 0.25,
        "quality": 0.30,
        "latency": 0.10,
        "availability": 0.15,
        "policy": 0.10,
        "budget": 0.05,
        "priority": 0.05,
    }

    def __init__(
        self,
        registry: ProviderRegistry,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._registry = registry
        self._weights = weights or dict(self.DEFAULT_WEIGHTS)

    def score_all(self, ctx: RoutingContext) -> list[ModelScore]:
        """Score every available model and return sorted (best first)."""
        scores = []
        for model in self._registry.all_models:
            info = self._registry.model_info(model)
            if info is None:
                continue
            score = self._score_one(model, info, ctx)
            if score is not None:
                scores.append(score)

        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores

    def _score_one(
        self, model: str, info: dict, ctx: RoutingContext
    ) -> Optional[ModelScore]:
        """Score a single model across all dimensions."""
        provider_name = info["provider"]

        # --- Policy: blocked models get score 0 ---
        if model in ctx.blocked_models:
            return None
        policy_score = 1.0
        if ctx.preferred_providers and provider_name not in ctx.preferred_providers:
            policy_score = 0.5

        # --- Availability (from model_info, no client creation needed) ---
        availability_score = 1.0 if info.get("healthy", True) else 0.0

        # --- Quality ---
        model_qualities = CATEGORY_QUALITY.get(model, {})
        quality_score = model_qualities.get(ctx.category, _DEFAULT_QUALITY)

        # --- Cost ---
        cost_input = info.get("cost_per_1m_input", 0)
        cost_output = info.get("cost_per_1m_output", 0)
        avg_cost = (cost_input + cost_output) / 2 if (cost_input + cost_output) > 0 else 0

        # Normalize cost: cheaper = higher score. Use log scale to avoid
        # extreme differences (e.g. $15 vs $0.15).
        if avg_cost > 0:
            import math
            cost_score = 1.0 / (1.0 + math.log2(1 + avg_cost))
        else:
            cost_score = 1.0  # free/unknown cost = best

        # Estimated cost for this request (assume ~500 prompt + ~200 completion)
        estimated_cost = (500 / 1_000_000) * cost_input + (200 / 1_000_000) * cost_output

        # --- Latency ---
        est_latency = _ESTIMATED_LATENCY.get(model, _DEFAULT_LATENCY)
        # Normalize: faster = higher score. 100ms -> 1.0, 2000ms -> ~0.3
        latency_score = 1.0 / (1.0 + est_latency / 500.0)

        # --- Budget ---
        budget_ratio = ctx.remaining_budget_usd / max(ctx.monthly_budget_usd, 1.0)
        budget_score = min(1.0, budget_ratio)

        # --- Priority ---
        priority_map = {0: 0.5, 1: 0.75, 2: 1.0}
        priority_score = priority_map.get(ctx.user_priority, 0.5)

        # --- Weighted total ---
        total = (
            self._weights["cost"] * cost_score
            + self._weights["quality"] * quality_score
            + self._weights["latency"] * latency_score
            + self._weights["availability"] * availability_score
            + self._weights["policy"] * policy_score
            + self._weights["budget"] * budget_score
            + self._weights["priority"] * priority_score
        )

        return ModelScore(
            model=model,
            provider=provider_name,
            total_score=round(total, 4),
            cost_score=round(cost_score, 4),
            quality_score=round(quality_score, 4),
            latency_score=round(latency_score, 4),
            availability_score=round(availability_score, 4),
            policy_score=round(policy_score, 4),
            budget_score=round(budget_score, 4),
            priority_score=round(priority_score, 4),
            estimated_cost_usd=round(estimated_cost, 6),
            estimated_latency_ms=est_latency,
        )
