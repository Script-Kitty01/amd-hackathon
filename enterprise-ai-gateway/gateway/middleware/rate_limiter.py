"""Rate limiting middleware — per-user and global limits.

Uses a sliding-window in-memory store. For production, swap to Redis.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    burst_size: int = 10
    window_seconds: int = 60


class RateLimiter:
    """Sliding-window rate limiter (in-memory)."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self._config = config or RateLimitConfig()
        # user_id -> list of timestamps
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._global_timestamps: list[float] = []

    def check(self, user_id: str, request=None) -> bool:
        """Check if the request is allowed. Raises HTTPException if rate limited."""
        from fastapi import HTTPException, status
        now = time.monotonic()
        window = self._config.window_seconds

        # Clean old entries
        self._windows[user_id] = [
            ts for ts in self._windows[user_id] if now - ts < window
        ]
        self._global_timestamps = [
            ts for ts in self._global_timestamps if now - ts < window
        ]

        user_count = len(self._windows[user_id])

        # Burst check: if user has sent burst_size in the last second, throttle
        recent_user = sum(1 for ts in self._windows[user_id] if now - ts < 1.0)
        if recent_user >= self._config.burst_size:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded (burst). Slow down.",
                headers={"Retry-After": "1"},
            )

        # Per-minute check
        if user_count >= self._config.requests_per_minute:
            retry_after = int(window - (now - self._windows[user_id][0]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded (per-minute).",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        # Record
        self._windows[user_id].append(now)
        self._global_timestamps.append(now)
        return True

    def get_stats(self, user_id: str) -> dict:
        """Return current usage stats for a user."""
        now = time.monotonic()
        window = self._config.window_seconds
        user_recent = [ts for ts in self._windows[user_id] if now - ts < window]
        return {
            "requests_last_minute": len(user_recent),
            "limit_per_minute": self._config.requests_per_minute,
            "remaining": max(0, self._config.requests_per_minute - len(user_recent)),
        }
