"""Local answering layer — tier 0 of the cascade (zero tokens).

Each solver attempts a task locally and either returns a `Solution` (with a
confidence in 0..1) or **abstains** (returns None) so the orchestrator escalates
to the next tier (local LLM, then Fireworks).

Confidence is "by construction": a solver returns an answer only when its own
signals say it is reliable. Thresholds are applied by the orchestrator (M4).

This module intentionally has no third-party dependencies so the deterministic
solvers (e.g. math) run anywhere and are trivially testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

from .categories import Category


@dataclass(frozen=True)
class Solution:
    answer: str
    confidence: float  # 0..1


class LocalSolver(Protocol):
    """A per-category local solver."""

    category: Category

    def try_solve(self, prompt: str) -> Optional[Solution]:
        """Return a Solution if confidently solvable locally, else None (abstain)."""
        ...


# --- deterministic math solver (T21) --------------------------------------

_PCT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*\$?\s*(\d[\d,]*(?:\.\d+)?)", re.I)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.I)
_PRICE = re.compile(r"\$\s*(\d[\d,]*(?:\.\d+)?)")
_PURE_EXPR = re.compile(r"^[\s\d+\-*/().]+$")

_DISCOUNT_WORDS = ("discount", "off", "reduced", "markdown")
_INCREASE_WORDS = ("increase", "increased", "markup", "marked up", "more", "raise", "raised")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _fmt(value: float, currency: bool) -> str:
    # Round money to 2 dp; otherwise emit a clean number.
    if currency:
        return f"${value:,.2f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


class MathSolver:
    """Handles arithmetic, `X% of Y`, and percentage discount/increase on a price.

    Conservative by design: returns None (abstain) whenever the pattern isn't a
    clean, unambiguous match, so uncertain problems escalate rather than risk the
    accuracy gate with a wrong local answer.
    """

    category = Category.MATH

    def try_solve(self, prompt: str) -> Optional[Solution]:
        p = prompt.strip()
        low = p.lower()

        # 1) "X% of Y"
        m = _PCT_OF.search(p)
        if m:
            pct, base = _num(m.group(1)), _num(m.group(2))
            currency = "$" in p[max(0, m.start() - 2): m.end() + 2]
            return Solution(_fmt(pct / 100.0 * base, currency), confidence=0.95)

        # 2) percentage discount / increase on a $ price
        pct_m = _PERCENT.search(p)
        price_m = _PRICE.search(p)
        if pct_m and price_m:
            pct = _num(pct_m.group(1))
            price = _num(price_m.group(1))
            if any(w in low for w in _DISCOUNT_WORDS):
                return Solution(_fmt(price * (1 - pct / 100.0), True), confidence=0.9)
            if any(w in low for w in _INCREASE_WORDS):
                return Solution(_fmt(price * (1 + pct / 100.0), True), confidence=0.9)

        # 3) a bare arithmetic expression, e.g. "12 * (3 + 4)"
        core = p.rstrip("=?. ")
        if _PURE_EXPR.match(core) and any(op in core for op in "+-*/") and re.search(r"\d", core):
            try:
                value = eval(core, {"__builtins__": {}}, {})  # noqa: S307 - digits/ops only
            except (SyntaxError, ZeroDivisionError, TypeError, NameError):
                return None
            if isinstance(value, (int, float)):
                return Solution(_fmt(float(value), "$" in p), confidence=0.9)

        return None  # abstain -> escalate


# --- registry --------------------------------------------------------------

_SOLVERS: dict[Category, list[LocalSolver]] = {
    Category.MATH: [MathSolver()],
}


def solvers_for(category: Category) -> list[LocalSolver]:
    """Local solvers registered for a category (empty if none yet)."""
    return _SOLVERS.get(category, [])
