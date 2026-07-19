"""Provider registry — maps provider names to client instances."""

from __future__ import annotations

from typing import Optional, Sequence, Union

from ..config import GatewayConfig, ProviderConfig
from .base import ProviderClient


def _get_provider_class(name: str):
    """Lazy-import provider classes so missing SDKs don't break everything."""
    if name in ("openai", "fireworks", "groq", "together"):
        from .openai_provider import OpenAIProvider
        return OpenAIProvider
    elif name in ("anthropic", "claude"):
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider
    elif name in ("gemini", "google"):
        from .gemini_provider import GeminiProvider
        return GeminiProvider
    raise ValueError(f"Unknown provider type: {name}")


_PROVIDER_ALIASES = {
    "openai": "openai",
    "fireworks": "openai",
    "groq": "openai",
    "together": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
}


class ProviderRegistry:
    """Holds all configured provider clients and their models."""

    def __init__(self, config: Union[GatewayConfig, Sequence[ProviderConfig]]) -> None:
        self._providers: dict[str, ProviderClient] = {}
        self._model_to_provider: dict[str, str] = {}  # model_name -> provider_name
        self._configs: dict[str, ProviderConfig] = {}  # stored for lazy init

        providers = config.providers if hasattr(config, 'providers') else config
        for pc in providers:
            self._configs[pc.name] = pc
            for model in pc.models:
                self._model_to_provider[model] = pc.name

    def _ensure_provider(self, name: str) -> Optional[ProviderClient]:
        """Lazily create a provider client on first access."""
        if name in self._providers:
            return self._providers[name]
        pc = self._configs.get(name)
        if pc is None:
            return None
        provider_type = _PROVIDER_ALIASES.get(pc.name.lower())
        if provider_type is None:
            return None
        cls = _get_provider_class(provider_type)
        client = cls(
            name=pc.name,
            api_key=pc.api_key,
            base_url=pc.base_url,
            models=pc.models,
            cost_per_1m_input=pc.cost_per_1m_input,
            cost_per_1m_output=pc.cost_per_1m_output,
        )
        self._providers[name] = client
        return client

    def get(self, name: str) -> Optional[ProviderClient]:
        return self._ensure_provider(name)

    def provider_for_model(self, model: str) -> Optional[ProviderClient]:
        provider_name = self._model_to_provider.get(model)
        if provider_name:
            return self._ensure_provider(provider_name)
        return None

    @property
    def all_models(self) -> list[str]:
        return list(self._model_to_provider.keys())

    @property
    def all_providers(self) -> list[ProviderClient]:
        return [self._ensure_provider(n) for n in self._configs if self._ensure_provider(n)]

    @property
    def healthy_providers(self) -> list[ProviderClient]:
        return [p for p in self.all_providers if p and p.healthy]

    def model_info(self, model: str) -> Optional[dict]:
        """Return cost/latency info for a model (no client creation needed)."""
        provider_name = self._model_to_provider.get(model)
        if provider_name is None:
            return None
        pc = self._configs.get(provider_name)
        if pc is None:
            return None
        return {
            "model": model,
            "provider": provider_name,
            "cost_per_1m_input": pc.cost_per_1m_input,
            "cost_per_1m_output": pc.cost_per_1m_output,
            "healthy": True,  # assumed healthy until proven otherwise
        }
