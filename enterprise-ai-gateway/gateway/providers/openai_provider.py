"""OpenAI-compatible provider (covers OpenAI, Fireworks, Groq, Together, etc.)."""

from __future__ import annotations

from typing import Optional

from .base import ProviderClient, ProviderResponse


class OpenAIProvider(ProviderClient):
    """OpenAI and OpenAI-compatible APIs (Fireworks, Groq, Together, vLLM, etc.)."""

    def __init__(self, **kwargs) -> None:
        from openai import AsyncOpenAI
        self._AsyncOpenAI = AsyncOpenAI
        super().__init__(**kwargs)
        self._client = self._AsyncOpenAI(
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

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            kwargs["stop"] = stop

        resp = await self._client.chat.completions.create(**kwargs)

        usage = resp.usage
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        return ProviderResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=self._elapsed_ms(start),
            finish_reason=choice.finish_reason or "stop",
            cost_usd=self._compute_cost(prompt_tokens, completion_tokens),
        )
