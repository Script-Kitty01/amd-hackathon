"""Abstract base for all LLM providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderResponse:
    """Normalized response from any provider."""
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cost_usd: float = 0.0


class ProviderClient(ABC):
    """Every provider implements this interface."""

    def __init__(
        self,
        name: str,
        api_key: str,
        base_url: str = "",
        models: list[str] | None = None,
        cost_per_1m_input: float = 0.0,
        cost_per_1m_output: float = 0.0,
    ) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
        self.models = models or []
        self.cost_per_1m_input = cost_per_1m_input
        self.cost_per_1m_output = cost_per_1m_output
        self._healthy = True
        self._last_error: Optional[str] = None

    @property
    def healthy(self) -> bool:
        return self._healthy

    def mark_unhealthy(self, reason: str) -> None:
        self._healthy = False
        self._last_error = reason

    def mark_healthy(self) -> None:
        self._healthy = True
        self._last_error = None

    @abstractmethod
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> ProviderResponse:
        """Send a chat completion and return a normalized response."""
        ...

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate USD cost from token counts."""
        input_cost = (prompt_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (completion_tokens / 1_000_000) * self.cost_per_1m_output
        return round(input_cost + output_cost, 6)

    def _timed(self) -> float:
        """Return a timer start value."""
        return time.perf_counter()

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 1)
