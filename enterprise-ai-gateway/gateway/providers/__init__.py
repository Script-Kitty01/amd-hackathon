"""Multi-provider LLM abstraction layer.

Every provider implements the same interface so the gateway can call any model
through a single code path. Adding a provider = one new file in this package.

Provider classes are lazy-imported so missing SDKs don't break the entire gateway.
"""

from .base import ProviderClient, ProviderResponse
from .registry import ProviderRegistry

__all__ = [
    "ProviderClient",
    "ProviderResponse",
    "ProviderRegistry",
]
