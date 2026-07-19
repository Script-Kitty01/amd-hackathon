"""Anthropic Claude provider."""

from __future__ import annotations

from typing import Optional

from .base import ProviderClient, ProviderResponse


class AnthropicProvider(ProviderClient):
    """Anthropic Claude API."""

    def __init__(self, **kwargs) -> None:
        from anthropic import AsyncAnthropic
        super().__init__(**kwargs)
        self._client = AsyncAnthropic(
            api_key=self.api_key,
            base_url=self.base_url or None,
            timeout=60.0,
            max_retries=0,
        )

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> ProviderResponse:
        start = self._timed()

        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop

        resp = await self._client.messages.create(**kwargs)

        usage = resp.usage
        text = ""
        for block in resp.content:
            if block.type == "text":
                text += block.text

        prompt_tokens = usage.input_tokens if usage else 0
        completion_tokens = usage.output_tokens if usage else 0

        return ProviderResponse(
            text=text.strip(),
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=self._elapsed_ms(start),
            finish_reason=resp.stop_reason or "stop",
            cost_usd=self._compute_cost(prompt_tokens, completion_tokens),
        )
