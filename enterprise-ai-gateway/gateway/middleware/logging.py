"""Structured logging middleware — JSON logs for every request.

Logs: timestamp, user, department, model used, tokens, cost, latency, cache hit,
      route decision, PII flags, and more.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import LOG_DIR


class GatewayLogger:
    """Structured JSON logger for the gateway."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir or LOG_DIR
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_request(self, entry: dict) -> None:
        """Write a structured log entry as JSON lines."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        entry.setdefault("type", "request")

        # Write to daily log file
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_path = self._log_dir / f"gateway-{date_str}.jsonl"

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_error(self, error: str, context: dict | None = None) -> None:
        self.log_request({
            "type": "error",
            "error": error,
            "context": context or {},
        })


# --- Request tracking middleware ---

class RequestTracker:
    """Tracks per-request metadata through the middleware pipeline."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.user: Optional[str] = None
        self.department: Optional[str] = None
        self.category: Optional[str] = None
        self.model: Optional[str] = None
        self.provider: Optional[str] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.cost_usd: float = 0.0
        self.cache_hit: bool = False
        self.pii_detected: bool = False
        self.pii_masked: bool = False
        self.injection_detected: bool = False
        self.fallback_used: bool = False
        self.fallback_chain: list[str] = []
        self.route_scores: list[dict] = []

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self.start_time) * 1000, 1)

    def to_dict(self) -> dict:
        return {
            "user": self.user,
            "department": self.department,
            "category": self.category,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "pii_detected": self.pii_detected,
            "pii_masked": self.pii_masked,
            "injection_detected": self.injection_detected,
            "fallback_used": self.fallback_used,
            "fallback_chain": self.fallback_chain,
            "route_scores": self.route_scores,
        }
