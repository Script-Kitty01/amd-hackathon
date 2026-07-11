"""LLM-judge for local accuracy measurement.

Grades each agent answer against the task's intent using an OpenAI-compatible
endpoint (e.g. Google AI Studio's Gemini API). This approximates the real
challenge judge so we can measure accuracy locally. Judge tokens are tracked
separately and are NOT part of the agent's token score.

Enabled when JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL are set; else None.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

from src.fireworks_client import _is_retryable

_MAX_RETRIES = 4
_BASE_DELAY = 5.0

_JUDGE_SYS = (
    "You grade an AI assistant's answer to a task. Judge only whether the answer "
    "satisfies the task's intent (correct and appropriately formatted). Reply with "
    "exactly one word: PASS or FAIL."
)

_JUDGE_TMPL = (
    "TASK:\n{prompt}\n\n"
    "PROPOSED ANSWER:\n{answer}\n\n"
    "{expected_block}"
    "Does the proposed answer correctly satisfy the task's intent? "
    "Reply with exactly one word: PASS or FAIL."
)


class Judge:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self.total_tokens = 0

    @classmethod
    def from_env(cls) -> Optional["Judge"]:
        base = os.environ.get("JUDGE_BASE_URL")
        key = os.environ.get("JUDGE_API_KEY")
        model = os.environ.get("JUDGE_MODEL")
        if not (base and key and model):
            return None
        return cls(base, key, model)

    def passed(self, prompt: str, answer: str, expected: Optional[str] = None) -> bool:
        expected_block = (
            f"REFERENCE (for guidance, may be partial): {expected}\n\n"
            if expected
            else ""
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYS},
            {
                "role": "user",
                "content": _JUDGE_TMPL.format(
                    prompt=prompt, answer=answer, expected_block=expected_block
                ),
            },
        ]
        resp = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model, messages=messages, temperature=0, max_tokens=5
                )
                break
            except Exception as exc:  # noqa: BLE001
                if not _is_retryable(exc) or attempt == _MAX_RETRIES - 1:
                    return False
                time.sleep(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
        if resp is None:
            return False
        usage = resp.usage
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0
        verdict = (resp.choices[0].message.content or "").strip().upper()
        return verdict.startswith("PASS")
