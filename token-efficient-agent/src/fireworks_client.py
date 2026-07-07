"""Thin Fireworks AI wrapper (OpenAI-compatible).

All inference MUST go through FIREWORKS_BASE_URL. This is the only place that
talks to the network, so token accounting and retry policy live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .config import Config


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FireworksClient:
    def __init__(self, cfg: Config) -> None:
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        """Single deterministic chat completion. Raises on API error."""
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        usage = resp.usage
        return LLMResult(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
        )
