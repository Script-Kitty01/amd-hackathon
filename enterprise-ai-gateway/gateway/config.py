"""Central configuration loaded from environment / .env file.

All secrets, provider keys, and policy settings live here.
Never hardcode credentials — everything comes from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "gateway.db"
BUDGETS_PATH = DATA_DIR / "budgets.json"
USERS_PATH = DATA_DIR / "users.json"
LOG_DIR = DATA_DIR / "logs"


@dataclass
class ProviderConfig:
    """Configuration for one LLM provider."""
    name: str
    api_key: str
    base_url: str
    models: list[str] = field(default_factory=list)
    # Cost per 1M tokens (input, output) — approximate, for scoring
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    # Provider-specific extra headers or params
    extra_headers: dict = field(default_factory=dict)


@dataclass
class GatewayConfig:
    """Top-level gateway configuration."""
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Auth
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"

    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.92
    redis_url: str = ""

    # Security
    pii_scan_enabled: bool = True
    pii_mask_enabled: bool = True
    prompt_injection_check: bool = True
    blocked_domains: list[str] = field(default_factory=list)

    # Budget
    budget_enabled: bool = True
    default_monthly_budget_usd: float = 1000.0

    # Routing
    routing_strategy: str = "scored"  # "scored" | "cheapest" | "fastest" | "round_robin"
    fallback_enabled: bool = True
    max_retries: int = 2

    # Providers
    providers: list[ProviderConfig] = field(default_factory=list)

    # Observability
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader."""
    env_path = BASE_DIR / path
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config() -> GatewayConfig:
    """Build GatewayConfig from environment variables."""
    _load_dotenv()

    cfg = GatewayConfig(
        host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
        port=int(os.environ.get("GATEWAY_PORT", "8000")),
        debug=os.environ.get("GATEWAY_DEBUG", "").lower() == "true",
        jwt_secret=os.environ.get("JWT_SECRET", "dev-secret-change-me"),
        jwt_expire_minutes=int(os.environ.get("JWT_EXPIRE_MINUTES", "60")),
        rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60")),
        rate_limit_burst=int(os.environ.get("RATE_LIMIT_BURST", "10")),
        cache_enabled=os.environ.get("CACHE_ENABLED", "true").lower() == "true",
        cache_ttl_seconds=int(os.environ.get("CACHE_TTL_SECONDS", "3600")),
        cache_similarity_threshold=float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.92")),
        redis_url=os.environ.get("REDIS_URL", ""),
        pii_scan_enabled=os.environ.get("PII_SCAN_ENABLED", "true").lower() == "true",
        pii_mask_enabled=os.environ.get("PII_MASK_ENABLED", "true").lower() == "true",
        prompt_injection_check=os.environ.get("PROMPT_INJECTION_CHECK", "true").lower() == "true",
        budget_enabled=os.environ.get("BUDGET_ENABLED", "true").lower() == "true",
        default_monthly_budget_usd=float(os.environ.get("DEFAULT_MONTHLY_BUDGET_USD", "1000.0")),
        routing_strategy=os.environ.get("ROUTING_STRATEGY", "scored"),
        fallback_enabled=os.environ.get("FALLBACK_ENABLED", "true").lower() == "true",
        max_retries=int(os.environ.get("MAX_RETRIES", "2")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        log_format=os.environ.get("LOG_FORMAT", "json"),
    )

    # Parse providers from env: PROVIDERS=openai,anthropic,gemini
    provider_names = [
        n.strip()
        for n in os.environ.get("PROVIDERS", "openai").split(",")
        if n.strip()
    ]

    for name in provider_names:
        prefix = name.upper()
        api_key = os.environ.get(f"{prefix}_API_KEY", "")
        base_url = os.environ.get(f"{prefix}_BASE_URL", "")
        models_raw = os.environ.get(f"{prefix}_MODELS", "")
        models = [m.strip() for m in models_raw.split(",") if m.strip()]

        if not api_key:
            continue  # skip unconfigured providers

        cfg.providers.append(ProviderConfig(
            name=name,
            api_key=api_key,
            base_url=base_url,
            models=models,
            cost_per_1m_input=float(os.environ.get(f"{prefix}_COST_INPUT", "0")),
            cost_per_1m_output=float(os.environ.get(f"{prefix}_COST_OUTPUT", "0")),
        ))

    return cfg
