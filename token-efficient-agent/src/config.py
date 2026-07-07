"""Runtime configuration loaded purely from environment variables.

The grading harness injects FIREWORKS_API_KEY, FIREWORKS_BASE_URL and
ALLOWED_MODELS at runtime. Never hardcode these or bundle a .env in the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


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
    return Config(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        models=[m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()],
    )
