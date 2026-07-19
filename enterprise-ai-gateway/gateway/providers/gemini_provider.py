"""Google Gemini provider."""

from __future__ import annotations

from typing import Optional

from .base import ProviderClient, ProviderResponse


class GeminiProvider(ProviderClient):
    """Google Gemini API (via google-generativeai SDK)."""

    def __init__(self, **kwargs) -> None:
        import google.generativeai as genai
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> ProviderResponse:
        import asyncio
        import google.generativeai as genai
        start = self._timed()

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system if system else None,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "stop_sequences": stop or [],
            },
        )

        # google-generativeai is sync; run in thread pool
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: gemini_model.generate_content(user)
        )

        text = (resp.text or "").strip()

        # Gemini doesn't expose token counts in the same way; estimate
        # Rough estimate: ~4 chars per token
        prompt_chars = len(system or "") + len(user)
        completion_chars = len(text)
        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, completion_chars // 4)

        return ProviderResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=self._elapsed_ms(start),
            finish_reason="stop",
            cost_usd=self._compute_cost(prompt_tokens, completion_tokens),
        )
