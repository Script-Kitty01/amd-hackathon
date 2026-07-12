"""Thin Fireworks AI wrapper (OpenAI-compatible).

All inference MUST go through FIREWORKS_BASE_URL. This is the only place that
talks to the network, so token accounting and retry policy live here.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from .config import Config

if TYPE_CHECKING:
    from .categories import Category


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
    def __init__(self, cfg: Config, profiler: Any | None = None) -> None:
        # Hard per-request timeout so a stuck call can never hang the run past
        # the 10-minute cap. We manage retries ourselves, so disable the SDK's.
        self._client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=_REQUEST_TIMEOUT,
            max_retries=0,
        )
        self._profiler = profiler
        # Per-call reasoning control via extra_body:
        #  - non-reasoning categories -> "none" (no CoT tax)
        #  - math/logic -> env override if set, else "low" (enough to reason on
        #    arithmetic word problems without emitting a huge chain-of-thought)
        self._reasoning_override = cfg.reasoning_effort or None
        # Models that reject reasoning_effort with a 400 (e.g. non-thinking models).
        # Learned at runtime so we send the param at most once to such a model,
        # then skip it. A 400 returns no usage, so this costs 0 tokens.
        self._no_extra_body: set[str] = set()

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        stop: list[str] | None = None,
        needs_reasoning: bool = False,
        # Profiling metadata:
        task_id: str = "unknown",
        category: "Category | None" = None,
    ) -> LLMResult:
        """Single deterministic chat completion. Retries transient/rate-limit errors."""
        resp = self._create_with_retry(model, system, user, max_tokens, stop, needs_reasoning)
        usage = resp.usage
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        result = LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )
        if self._profiler:
            self._profiler.record(
                task_id=task_id,
                category=str(category.value) if category else "unknown",
                model=model,
                prompt=result.prompt_tokens,
                completion=result.completion_tokens,
                total=result.total_tokens,
            )
        return result

    def _create_once(self, model: str, system: str, user: str, max_tokens: int,
                     stop: list[str] | None = None, needs_reasoning: bool = False):
        kwargs: dict = {}
        # Build per-call reasoning_effort:
        # - Non-reasoning categories on any model: suppress thinking (=none).
        # - Reasoning categories (math/logic): use global config or default.
        # Gemma models benefit most from reasoning=none (they can silently enter
        # <|think|> mode). Minimax/kimi are reasoning models — suppress only when
        # the category genuinely doesn't need step-by-step thinking.
        if model not in self._no_extra_body:
            if not needs_reasoning:
                # Cheap categories: suppress reasoning entirely (no CoT tax).
                effort = "none"
            else:
                # Math/logic: limited reasoning — enough for arithmetic word
                # problems, far fewer tokens than full chain-of-thought.
                effort = self._reasoning_override or "low"
            kwargs["extra_body"] = {"reasoning_effort": effort}
        if stop:
            kwargs["stop"] = stop
        content = f"{system}\n\n{user}" if system else user
        return self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )

    def _create_with_retry(self, model: str, system: str, user: str, max_tokens: int,
                           stop: list[str] | None = None, needs_reasoning: bool = False):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._create_once(model, system, user, max_tokens, stop, needs_reasoning)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # If the model rejects reasoning_effort (e.g. a non-thinking
                # model), drop the param and retry immediately (0 tokens on 400).
                if model not in self._no_extra_body and _is_reasoning_param_error(exc):
                    self._no_extra_body.add(model)
                    try:
                        return self._create_once(model, system, user, max_tokens, stop, needs_reasoning)
                    except Exception as exc2:  # noqa: BLE001
                        last_exc = exc2
                        exc = exc2
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    raise
                if self._profiler:
                    self._profiler.record_retry()
                delay = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
                time.sleep(delay)
        raise last_exc  # pragma: no cover
