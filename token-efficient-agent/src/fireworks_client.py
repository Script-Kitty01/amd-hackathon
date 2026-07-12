"""Thin Fireworks AI wrapper (OpenAI-compatible).

All inference MUST go through FIREWORKS_BASE_URL. This is the only place that
talks to the network, so token accounting and retry policy live here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import Config

_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds; exponential backoff for transient/rate-limit errors
_MAX_DELAY = 20.0  # cap a single backoff so retries can't blow the wall-clock budget
_REQUEST_TIMEOUT = 60.0  # hard per-call timeout; a hung call fails fast instead of blocking


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str = ""  # "stop" (complete) | "length" (truncated) | ...


def _is_retryable(exc: Exception) -> bool:
    """Rate limits (429) and transient 5xx / connection errors are retryable."""
    name = type(exc).__name__
    if name in ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"):
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 500, 502, 503, 504)


def _is_reasoning_param_error(exc: Exception) -> bool:
    """A 400 that looks like the model doesn't accept reasoning_effort/extra params."""
    status = getattr(exc, "status_code", None)
    if status not in (400, 422) and type(exc).__name__ != "BadRequestError":
        return False
    msg = str(exc).lower()
    return (
        "reasoning" in msg
        or "invalid_request" in msg
        or "unknown" in msg
        or "unexpected" in msg
        or "not support" in msg
        or status in (400, 422)
    )


class FireworksClient:
    def __init__(self, cfg: Config) -> None:
        # Hard per-request timeout so a stuck call can never hang the run past
        # the 10-minute cap. We manage retries ourselves, so disable the SDK's.
        self._client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=_REQUEST_TIMEOUT,
            max_retries=0,
        )
        # Pass provider reasoning control via extra_body when configured. Thinking
        # models (e.g. minimax) otherwise burn the max_tokens budget on hidden
        # reasoning and truncate the answer; reasoning_effort=none fixes that.
        self._extra_body = (
            {"reasoning_effort": cfg.reasoning_effort} if cfg.reasoning_effort else None
        )
        # Models that reject reasoning_effort with a 400 (e.g. non-thinking models).
        # Learned at runtime so we send the param at most once to such a model,
        # then skip it. A 400 returns no usage, so this costs 0 tokens.
        self._no_extra_body: set[str] = set()

    def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        """Single deterministic chat completion. Retries transient/rate-limit errors."""
        resp = self._create_with_retry(model, system, user, max_tokens)
        usage = resp.usage
        choice = resp.choices[0]
        # Some thinking models (kimi-k2p7-code) put the answer in `content` and
        # reasoning in a separate field. Others (minimax-m3 via vLLM) may leak
        # reasoning tags directly into `content`. We read content only — the
        # caller's strip_reasoning handles any leaked tags.
        text = (choice.message.content or "").strip()
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )

    def _create_once(self, model: str, system: str, user: str, max_tokens: int):
        kwargs = {}
        if self._extra_body and model not in self._no_extra_body:
            kwargs["extra_body"] = self._extra_body
        # Merge the instruction into a single user turn instead of a separate
        # "system" role. Gemma-family models don't support a system role and can
        # mishandle it; a merged user message is accepted by every model
        # (minimax/kimi/gemma alike). This is the same approach the local LLM uses.
        content = f"{system}\n\n{user}" if system else user
        return self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )

    def _create_with_retry(self, model: str, system: str, user: str, max_tokens: int):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._create_once(model, system, user, max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # If the model rejects reasoning_effort, drop it and retry now
                # (doesn't consume a retry attempt; 400 costs 0 tokens).
                if (
                    self._extra_body
                    and model not in self._no_extra_body
                    and _is_reasoning_param_error(exc)
                ):
                    self._no_extra_body.add(model)
                    try:
                        return self._create_once(model, system, user, max_tokens)
                    except Exception as exc2:  # noqa: BLE001
                        last_exc = exc2
                        exc = exc2
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
                time.sleep(delay)
        raise last_exc  # pragma: no cover
