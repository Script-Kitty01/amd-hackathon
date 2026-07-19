"""Routing engine — classifies prompts, scores models, picks the best one.

Integrates the category classifier from the hackathon project with the
multi-factor model scorer. Supports fallback to next-best model on failure.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from ..providers.base import ProviderResponse
from ..providers.registry import ProviderRegistry
from .scorer import ModelScore, ModelScorer, RoutingContext, TaskCategory


# --- Lightweight category classifier (from hackathon router) ---

_CODE_FENCE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|=>|;\s*$", re.MULTILINE)
_DEBUG_HINT = re.compile(r"\b(bug|fix|error|wrong|broken|debug|fails?|exception)\b", re.I)
_LOGIC_HINT = re.compile(
    r"\b(puzzle|deduce|arrange|rank(ing|ed)?|in some order|order them|"
    r"who (came|finished|won|sits?|stands?)|beat|taller|shorter|"
    r"older|younger|faster|slower|next to|seated|constraint|neither)\b",
    re.I,
)
_MATH_SIGNAL = re.compile(
    r"\b(?:average|mean|sum|product)\s+of\s+[-\d]|"
    r"\bwhat\s+percent(?:age)?\s+of\b|"
    r"-?\d+(?:\.\d+)?\s+(?:plus|minus|times|multiplied by|divided by)\s+-?\d",
    re.I,
)

_KEYWORDS: dict[str, tuple[str, ...]] = {
    TaskCategory.SUMMARIZATION: (
        "summarise", "summarize", "summary", "one sentence", "tl;dr",
        "condense", "in a sentence", "in brief", "bullet point",
    ),
    TaskCategory.SENTIMENT: (
        "sentiment", "positive or negative", "tone", "emotion",
        "classify the sentiment", "positive, negative",
    ),
    TaskCategory.NER: (
        "named entit", "extract", "entities", "person, org",
        "person, organization, location",
    ),
    TaskCategory.MATH: (
        "calculate", "percent", "%", "how many", "how much",
        "sum of", "compute", "product of", "divided by",
        "interest", "rate of", "total cost",
    ),
    TaskCategory.FACTUAL: (
        "explain", "what is", "what are", "who ", "how does",
        "why does", "define", "definition",
    ),
}


def classify_prompt(prompt: str) -> str:
    """Classify a prompt into a task category (zero-token, local)."""
    p = prompt.lower()
    scores: dict[str, float] = {c: 0.0 for c in [
        TaskCategory.FACTUAL, TaskCategory.MATH, TaskCategory.SENTIMENT,
        TaskCategory.SUMMARIZATION, TaskCategory.NER,
        TaskCategory.CODE_DEBUG, TaskCategory.CODE_GEN, TaskCategory.LOGIC,
    ]}

    # Code signals
    if _CODE_FENCE.search(prompt) or "function" in p or "code" in p:
        if _DEBUG_HINT.search(prompt):
            scores[TaskCategory.CODE_DEBUG] += 3.0
        else:
            scores[TaskCategory.CODE_GEN] += 3.0

    # Math signals
    if _MATH_SIGNAL.search(prompt):
        scores[TaskCategory.MATH] += 2.0

    # Logic signals
    scores[TaskCategory.LOGIC] += 2.0 * len(_LOGIC_HINT.findall(prompt))

    # Keyword signals
    for cat, words in _KEYWORDS.items():
        scores[cat] += sum(1.0 for w in words if w in p)

    total = sum(scores.values())
    if total <= 0:
        return TaskCategory.GENERAL

    # Highest score wins; tie-break by specificity
    priority = [
        TaskCategory.CODE_DEBUG, TaskCategory.CODE_GEN, TaskCategory.LOGIC,
        TaskCategory.MATH, TaskCategory.NER, TaskCategory.SENTIMENT,
        TaskCategory.SUMMARIZATION, TaskCategory.FACTUAL,
    ]
    best = max(scores, key=lambda c: (scores[c], -priority.index(c)))
    return best


# --- Routing engine ---

@dataclass
class RouteDecision:
    """The result of a routing decision."""
    model: str
    provider: str
    category: str
    score_breakdown: ModelScore
    all_scores: list[ModelScore]
    fallback_chain: list[str] = field(default_factory=list)


class RoutingEngine:
    """Classify -> Score -> Pick best model -> Execute with fallback."""

    def __init__(
        self,
        registry: ProviderRegistry,
        scorer: ModelScorer | None = None,
        max_retries: int = 2,
    ) -> None:
        self._registry = registry
        self._scorer = scorer or ModelScorer(registry)
        self._max_retries = max_retries

    def decide(
        self,
        prompt: str,
        department: str = "default",
        user_priority: int = 0,
        remaining_budget: float = 1000.0,
        monthly_budget: float = 1000.0,
        blocked_models: list[str] | None = None,
        preferred_providers: list[str] | None = None,
    ) -> RouteDecision:
        """Classify the prompt and pick the best model."""
        category = classify_prompt(prompt)

        ctx = RoutingContext(
            category=category,
            department=department,
            user_priority=user_priority,
            remaining_budget_usd=remaining_budget,
            monthly_budget_usd=monthly_budget,
            blocked_models=blocked_models or [],
            preferred_providers=preferred_providers or [],
        )

        all_scores = self._scorer.score_all(ctx)
        if not all_scores:
            raise RuntimeError("No available models to route to")

        best = all_scores[0]
        return RouteDecision(
            model=best.model,
            provider=best.provider,
            category=category,
            score_breakdown=best,
            all_scores=all_scores,
        )

    async def execute(
        self,
        decision: RouteDecision,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> tuple[ProviderResponse, RouteDecision]:
        """Execute the decision with fallback to next-best models on failure."""
        tried: list[str] = []
        last_error: Optional[str] = None

        # Build fallback chain: best model first, then next-best healthy models
        chain = [s for s in decision.all_scores if s.model not in tried]
        chain = chain[:self._max_retries + 1]

        for score in chain:
            provider = self._registry.get(score.provider)
            if provider is None or not provider.healthy:
                tried.append(score.model)
                continue

            try:
                resp = await provider.complete(
                    model=score.model,
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                decision.fallback_chain = tried
                return resp, decision
            except Exception as exc:
                last_error = str(exc)
                tried.append(score.model)
                provider.mark_unhealthy(last_error)

        raise RuntimeError(
            f"All models failed. Tried: {tried}. Last error: {last_error}"
        )
