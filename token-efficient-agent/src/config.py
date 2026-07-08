"""Runtime configuration loaded purely from environment variables.

The grading harness injects GOOGLE_API_KEY, GOOGLE_BASE_URL and
GOOGLE_MODELS at runtime. Never hardcode these or bundle a .env in the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# Paths are fixed by the challenge contract.
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Overall wall-clock budget (seconds). The hard cap is 10 minutes; leave margin.
RUNTIME_BUDGET_SECONDS = int(os.environ.get("RUNTIME_BUDGET_SECONDS", "540"))


@dataclass(frozen=True)
class Config:
    api_key: str
    base_url: str
    models: list[str]

    @property
    def default_model(self) -> str:
        return self.models[0]


def load_config() -> Config:
    """Read required env vars. Raises KeyError if any are missing."""
    cfg = Config(
        api_key=os.environ["GOOGLE_API_KEY"],
        base_url=os.environ.get("GOOGLE_BASE_URL", ""),
        models=[m.strip() for m in os.environ["GOOGLE_MODELS"].split(",") if m.strip()],
    )
    print(f"Loaded Config: API Key (first 5 chars): {cfg.api_key[:5]}, Base URL: {cfg.base_url}, Models: {cfg.models}, Default Model: {cfg.default_model}")
    return cfg