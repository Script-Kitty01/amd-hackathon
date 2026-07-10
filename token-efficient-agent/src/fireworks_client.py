"""Thin Fireworks AI wrapper (OpenAI-compatible).

All inference MUST go through FIREWORKS_BASE_URL. This is the only place that
talks to the network, so token accounting and retry policy live here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import Config

_MAX_RETRIES = 4
_BASE_DELAY = 5.0  # seconds; exponential backoff for transient/rate-limit errors


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _is_retryable(exc: Exception) -> bool:
    """Rate limits (429) and transient 5xx / connection errors are retryable."""
    name = type(exc).__name__
    if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"):
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 500, 502, 503, 504)


class FireworksClient:
    def __init__(self, cfg: Config) -> None:
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        # Pass provider reasoning control via extra_body when configured.
        self._extra_body = (
            {"reasoning_effort": cfg.reasoning_effort} if cfg.reasoning_effort else None
        )

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        """Single deterministic chat completion. Retries transient/rate-limit errors."""
        resp = self._create_with_retry(model, system, user, max_tokens)
        usage = resp.usage
        return LLMResult(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    def _create_with_retry(self, model: str, system: str, user: str, max_tokens: int):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs = {}
                if self._extra_body:
                    kwargs["extra_body"] = self._extra_body
                return self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
        raise last_exc  # pragma: no cover
