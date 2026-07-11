"""Local LLM tier of the cascade (e.g. gemma via Ollama on ROCm).

Structurally identical to the Fireworks client but points at a local,
OpenAI-compatible endpoint. Because inference is local, its tokens count as
**zero** — so we never track usage here.

Enabled only when LOCAL_LLM_BASE_URL and LOCAL_LLM_MODEL are set; otherwise
`from_env()` returns None and the cascade simply skips this tier and falls
through to Fireworks. This keeps the agent runnable with or without a local
model present (important while the runtime/hardware is still being set up).

Confidence: a per-category prior, optionally sharpened by self-consistency
(AutoMix-style) when LOCAL_LLM_SAMPLES > 1 — sample a few times and let answer
agreement drive confidence.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Callable, Optional

from .categories import Category
from .local_solvers import Solution
from .prompts import spec_for

# Prior confidence for a single local-LLM answer, per category. Generative,
# non-reasoning tasks get higher priors; reasoning/code get lower so they
# escalate more readily. Tuned/overridden by calibration (T27).
_DEFAULT_PRIOR: dict[Category, float] = {
    Category.FACTUAL: 0.70,
    Category.SUMMARIZATION: 0.70,
    Category.SENTIMENT: 0.70,
    Category.NER: 0.60,
    Category.MATH: 0.50,
    Category.CODE_GEN: 0.55,
    Category.CODE_DEBUG: 0.50,
    Category.LOGIC: 0.45,
}

# complete_fn(system, user, max_tokens, temperature) -> answer text
CompleteFn = Callable[[str, str, int, float], str]

# Minimum token budget for local generation (free tokens; leaves room for the
# model's hidden reasoning so it doesn't hit the cap before emitting an answer).
_LOCAL_MIN_TOKENS = 512


class LocalLLM:
    def __init__(
        self,
        model: str,
        complete_fn: CompleteFn,
        samples: int = 1,
        priors: Optional[dict[Category, float]] = None,
    ) -> None:
        self._model = model
        self._complete = complete_fn
        self._samples = max(1, samples)
        self._priors = priors or dict(_DEFAULT_PRIOR)

    @classmethod
    def from_env(cls) -> Optional["LocalLLM"]:
        base_url = os.environ.get("LOCAL_LLM_BASE_URL")
        model = os.environ.get("LOCAL_LLM_MODEL")
        if not base_url or not model:
            return None
        samples = int(os.environ.get("LOCAL_LLM_SAMPLES", "1"))
        return cls(model, _openai_complete_fn(base_url, model), samples=samples)

    def try_solve(self, category: Category, prompt: str) -> Optional[Solution]:
        spec = spec_for(category)
        temperature = 0.0 if self._samples == 1 else 0.4
        # Local tokens are free: give the model room to "think" and still answer.
        # (Gemma 3n spends hidden reasoning tokens; a tight cap yields empty output.)
        max_tokens = max(spec.max_tokens, _LOCAL_MIN_TOKENS)

        answers: list[str] = []
        for _ in range(self._samples):
            try:
                text = self._complete(spec.system, prompt, max_tokens, temperature)
            except Exception:
                continue
            if text and text.strip():
                answers.append(text.strip())

        if not answers:
            return None  # local model unavailable/failed -> escalate

        best, agreement = _majority(answers)
        prior = self._priors.get(category, 0.5)
        if self._samples > 1:
            confidence = prior * (0.5 + 0.5 * agreement)
        else:
            confidence = prior
        return Solution(best, confidence=round(confidence, 3))


def _majority(answers: list[str]) -> tuple[str, float]:
    """Most common answer (by normalized form) and its agreement fraction."""
    norm = [a.strip().lower() for a in answers]
    counts = Counter(norm)
    top_norm, top_count = counts.most_common(1)[0]
    # Return the first original answer whose normalized form is the winner.
    best = next(a for a in answers if a.strip().lower() == top_norm)
    return best, top_count / len(answers)


def _openai_complete_fn(base_url: str, model: str) -> CompleteFn:
    """Build a completion function backed by an OpenAI-compatible local endpoint."""
    from openai import OpenAI  # lazy import so tests need no openai/network

    client = OpenAI(api_key=os.environ.get("LOCAL_LLM_API_KEY", "local"), base_url=base_url)

    def _complete(system: str, user: str, max_tokens: int, temperature: float) -> str:
        # Merge instruction into the user turn: Gemma 3n emits fewer hidden
        # reasoning tokens this way and reliably returns the answer within budget.
        content = f"{system}\n\n{user}" if system else user
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    return _complete
