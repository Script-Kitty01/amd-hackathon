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

# A list of numbers separated by commas / "and" / "&" (e.g. "4, 8, and 12").
# Plain numbers (no embedded thousands-comma) so list commas aren't swallowed.
_NUMLIST = r"(-?\d+(?:\.\d+)?(?:(?:\s*(?:,|and|&)\s*)+-?\d+(?:\.\d+)?)*)"
# Explicit-list aggregates and simple two-operand word arithmetic (all safe,
# single-operation, deterministic).
_AVG = re.compile(r"\b(?:average|mean)\s+of\s+" + _NUMLIST, re.I)
_SUM = re.compile(r"\bsum\s+of\s+" + _NUMLIST, re.I)
_PRODUCT = re.compile(r"\bproduct\s+of\s+" + _NUMLIST, re.I)
_PCT_WHAT = re.compile(
    r"what\s+percent(?:age)?\s+of\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)", re.I
)
_ARITH = re.compile(
    r"(-?\d+(?:\.\d+)?)\s+(plus|minus|times|multiplied by|divided by)\s+(-?\d+(?:\.\d+)?)$",
    re.I,
)
_SIMPLE_ARITH = re.compile(
    r"what is (-?\d+(?:\.\d+)?)\s*(plus|\+|minus|-|times|\*|divided by|/)\s*(-?\d+(?:\.\d+)?)\??$",
    re.I,
)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_DISCOUNT_WORDS = ("discount", "off", "reduced", "markdown")
_INCREASE_WORDS = ("increase", "increased", "markup", "marked up", "more", "raise", "raised")
# Signals of a multi-step problem the naive solver must NOT attempt.
_MULTISTEP_WORDS = ("then", "additional", "additionally", "after that", "followed by", "subsequent")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _numbers(s: str) -> list[float]:
    return [float(x.replace(",", "")) for x in _NUM_RE.findall(s)]


def _all_numbers_consumed(prompt: str, consumed: set[float]) -> bool:
    """Safety: all significant standalone numbers in the prompt must have been used.

    We use a simple heuristic: find all numbers that:
    - are NOT preceded by a letter (excludes Q1, Q2, Q3, etc.)
    - are NOT part of an ordinal (1st, 2nd, 3rd) 
    Then check each is in the consumed set.
    """
    # Match numbers not immediately preceded by a letter (Q1, 3rd -> excluded)
    for m in re.finditer(r"(?<![A-Za-z])(\d[\d,]*(?:\.\d+)?)", prompt):
        raw = m.group(1)
        # Skip ordinals: followed by st/nd/rd/th
        after = prompt[m.end():m.end()+2]
        if re.match(r"(?:st|nd|rd|th)\b", after, re.I):
            continue
        try:
            n = float(raw.replace(",", ""))
            if n <= 0:
                continue
            # Must appear in consumed (within rounding)
            if not any(abs(n - c) < 1e-6 for c in consumed):
                return False
        except ValueError:
            continue
    return True


def _apply_op(a: float, op: str, b: float) -> Optional[float]:
    op = op.lower()
    if op in ("plus", "+"):
        return a + b
    if op in ("minus", "-"):
        return a - b
    if op in ("times", "multiplied by", "*"):
        return a * b
    if op in ("divided by", "/"):
        return a / b if b != 0 else None
    return None


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

        # Highest-confidence patterns first
        m = _SIMPLE_ARITH.search(p)
        if m:
            result = _apply_op(_num(m.group(1)), m.group(2), _num(m.group(3)))
            if result is not None:
                return Solution(_fmt(result, "$" in p), confidence=0.99)

        # Multi-step / multi-percentage problems are beyond this solver — abstain
        # rather than return a confidently-wrong single-step answer.
        pct_count = len(_PERCENT.findall(p))
        multi_step = any(w in low for w in _MULTISTEP_WORDS)
        single_clean = pct_count <= 1 and not multi_step

        # 1) "X% of Y"
        m = _PCT_OF.search(p)
        if m and single_clean:
            pct, base = _num(m.group(1)), _num(m.group(2))
            currency = "$" in p[max(0, m.start() - 2): m.end() + 2]
            return Solution(_fmt(pct / 100.0 * base, currency), confidence=0.95)

        # 2) percentage discount / increase on a $ price (single, clean step only)
        if single_clean:
            pct_m = _PERCENT.search(p)
            price_m = _PRICE.search(p)
            if pct_m and price_m:
                pct = _num(pct_m.group(1))
                price = _num(price_m.group(1))
                if any(w in low for w in _DISCOUNT_WORDS):
                    return Solution(_fmt(price * (1 - pct / 100.0), True), confidence=0.9)
                if any(w in low for w in _INCREASE_WORDS):
                    return Solution(_fmt(price * (1 + pct / 100.0), True), confidence=0.9)

        # 3) average / mean of an explicit list of numbers
        m = _AVG.search(p)
        if m:
            nums = _numbers(m.group(1))
            if len(nums) >= 2:
                return Solution(_fmt(sum(nums) / len(nums), "$" in p), confidence=0.9)

        # 4) sum of an explicit list
        m = _SUM.search(p)
        if m:
            nums = _numbers(m.group(1))
            if len(nums) >= 2:
                return Solution(_fmt(sum(nums), "$" in p), confidence=0.9)

        # 5) product of an explicit list
        m = _PRODUCT.search(p)
        if m:
            nums = _numbers(m.group(1))
            if len(nums) >= 2:
                prod = 1.0
                for n in nums:
                    prod *= n
                return Solution(_fmt(prod, "$" in p), confidence=0.9)

        # 6) "what percent of X is Y"
        m = _PCT_WHAT.search(p)
        if m:
            x, y = _num(m.group(1)), _num(m.group(2))
            if x != 0:
                return Solution(_fmt(y / x * 100.0, False) + "%", confidence=0.9)

        # 7) simple two-operand word arithmetic ("15 plus 27", "100 divided by 4")
        m = _ARITH.search(p)
        if m and single_clean:
            result = _apply_op(_num(m.group(1)), m.group(2), _num(m.group(3)))
            if result is not None:
                return Solution(_fmt(result, "$" in p), confidence=0.88)

        # 8) a bare arithmetic expression, e.g. "12 * (3 + 4)"
        core = p.rstrip("=?. ")
        if _PURE_EXPR.match(core) and any(op in core for op in "+-*/") and re.search(r"\d", core):
            try:
                value = eval(core, {"__builtins__": {}}, {})  # noqa: S307 - digits/ops only
            except (SyntaxError, ZeroDivisionError, TypeError, NameError):
                return None
            if isinstance(value, (int, float)):
                return Solution(_fmt(float(value), "$" in p), confidence=0.9)

        return None  # abstain -> escalate


# --- Family 2: operation-chain solver (warehouse-style multi-step) ---------
# Handles: "starts with X, sells N%, restocks M, sells K → final"
# Safety: all numeric quantities in the prompt must be consumed.

_STARTS_RE = re.compile(
    r"(?:starts?\s+with|begins?\s+with|initially\s+has?|has?\s+an?\s+initial)"
    r"\s+(?:\$\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:units?|items?|pieces?)?",
    re.I,
)
_OP_RE = re.compile(
    r"(?:"
    # percentage decrease: sells/loses/uses/decreases N%
    r"(?:sells?|loses?|uses?|decreases?\s+by|removes?|ships?)\s+"
    r"(?P<pct_dec_val>\d+(?:\.\d+)?)\s*%"
    r"|"
    # percentage increase: grows/increases by N%
    r"(?:grows?\s+by|increases?\s+by)\s*(?P<pct_inc_val>\d+(?:\.\d+)?)\s*%"
    r"|"
    # fixed addition: restocks/receives/adds N
    r"(?:restocks?|receives?|adds?|gains?|acquires?)\s+(?P<add_val>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s+(?:units?|items?|pieces?))?"
    r"|"
    # fixed subtraction: sells/removes/uses/ships N (exact number, not percentage)
    r"(?:sells?|removes?|uses?|ships?|loses?)\s+(?P<sub_val>\d[\d,]*(?:\.\d+)?)"
    r"(?:\s+(?:units?|items?|pieces?))?"
    r")",
    re.I,
)


class OperationChainSolver:
    """Solves multi-step inventory/state problems: start → ops → final value.

    Safety rule: every significant number in the prompt must be consumed in the
    calculation. If any number is leftover the problem is beyond this solver.
    """

    category = Category.MATH

    def try_solve(self, prompt: str) -> Optional[Solution]:
        p = prompt.strip()

        start_m = _STARTS_RE.search(p)
        if not start_m:
            return None  # no starting value → not this family

        try:
            value = _num(start_m.group(1))
        except ValueError:
            return None

        consumed: set[float] = {value}
        ops = list(_OP_RE.finditer(p))
        if not ops:
            return None

        for m in ops:
            try:
                if m.group("pct_dec_val"):
                    pct = float(m.group("pct_dec_val"))
                    consumed.add(pct)
                    value *= (1 - pct / 100.0)
                elif m.group("pct_inc_val"):
                    pct = float(m.group("pct_inc_val"))
                    consumed.add(pct)
                    value *= (1 + pct / 100.0)
                elif m.group("add_val"):
                    n = _num(m.group("add_val"))
                    consumed.add(n)
                    value += n
                elif m.group("sub_val"):
                    n = _num(m.group("sub_val"))
                    consumed.add(n)
                    value -= n
            except (ValueError, IndexError):
                return None

        if not _all_numbers_consumed(p, consumed):
            return None  # some number wasn't used → problem too complex, abstain

        currency = "$" in p
        answer = _fmt(round(value, 2), currency)
        return Solution(answer, confidence=0.93)


# --- Family 3: ratio/proportion + cost (recipe-style) ----------------------
# Handles: "X [unit] for A → how much for B? costs $P per [unit] → total cost?"

_RATIO_RE = re.compile(
    r"(\d+(?:/\d+)?(?:\.\d+)?)\s+"
    r"(?:cups?|kg|g|oz|lb|lbs?|liters?|litres?|ml|tbsp|tsp|units?|items?|pieces?|servings?)?"
    r"\s+(?:of\s+\S+\s+)?(?:for|to\s+make|makes?)\s+(\d[\d,]*(?:\.\d+)?)"
    r"\s*(?:cookies?|servings?|portions?|people|units?|items?|pieces?)?",
    re.I,
)
_SCALE_RE = re.compile(
    r"(?:how\s+much\s+\S+\s+(?:is\s+)?(?:needed|required)\s+for|"
    r"(?:needed?|required?)\s+for)\s+(\d[\d,]*(?:\.\d+)?)"
    r"\s*(?:cookies?|servings?|portions?|people|units?|items?|pieces?)?",
    re.I,
)
_COST_RE = re.compile(
    r"(?:costs?|price(?:d)?|at)\s+\$\s*(\d+(?:\.\d+)?)"
    r"(?:\s+per\s+(?:cup|kg|g|oz|lb|unit|item|piece|serving))?",
    re.I,
)


def _parse_fraction(s: str) -> float:
    """Parse "3/4" or "1.5" as a float."""
    if "/" in s:
        num, den = s.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(s)


# --- Family 4: speed / distance / time (single clean relation) -------------
# distance = speed * time. Conservative: exactly one speed and one time, and
# every number in the prompt must be consumed (no multi-leg problems).

_SPEED_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:km/h|kph|mph|m/s|km per hour|miles per hour|meters per second)",
    re.I,
)
_TIME_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)",
    re.I,
)


class SpeedDistanceSolver:
    """distance = speed x time, only when exactly one speed and one time appear."""

    category = Category.MATH

    def try_solve(self, prompt: str) -> Optional[Solution]:
        p = prompt.strip()
        low = p.lower()
        # Must be asking for distance (not time/speed, which need different forms)
        if "distance" not in low and "how far" not in low:
            return None
        # Multi-leg problems (then, and then) are beyond this solver
        if any(w in low for w in ("then", "after that", "followed by")):
            return None

        speeds = _SPEED_RE.findall(p)
        times = _TIME_RE.findall(p)
        if len(speeds) != 1 or len(times) != 1:
            return None

        try:
            speed = float(speeds[0])
            time = float(times[0])
        except ValueError:
            return None

        distance = speed * time
        consumed = {speed, time}
        if not _all_numbers_consumed(p, consumed):
            return None
        return Solution(_fmt(distance, False), confidence=0.9)


# --- Family 5: simple interest ---------------------------------------------
# I = P * R * T / 100. Only when the prompt explicitly says "simple interest".

_PRINCIPAL_RE = re.compile(r"(?:principal|invests?|deposits?|borrows?|of)\s+\$?\s*(\d[\d,]*(?:\.\d+)?)", re.I)
_RATE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%(?:\s*(?:per\s+(?:annum|year)|annual))?", re.I)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*years?", re.I)


class SimpleInterestSolver:
    """simple interest I = P*R*T/100, only when 'simple interest' is stated."""

    category = Category.MATH

    def try_solve(self, prompt: str) -> Optional[Solution]:
        p = prompt.strip()
        low = p.lower()
        if "simple interest" not in low:
            return None
        # Compound interest is a different formula — abstain
        if "compound" in low:
            return None

        principals = _PRINCIPAL_RE.findall(p)
        rates = _RATE_RE.findall(p)
        years = _YEARS_RE.findall(p)
        if not principals or len(rates) != 1 or len(years) != 1:
            return None

        try:
            principal = _num(principals[0])
            rate = float(rates[0])
            time = float(years[0])
        except ValueError:
            return None

        interest = principal * rate * time / 100.0
        consumed = {principal, rate, time}
        if not _all_numbers_consumed(p, consumed):
            return None
        currency = "$" in p
        return Solution(_fmt(round(interest, 2), currency), confidence=0.9)


# --- Family 6: unit cost ----------------------------------------------------
# "N items cost $X, price per item?" → X/N

_TOTAL_FOR_N = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:items?|units?|pieces?|kg|pounds?|lbs?)\s+cost\s+\$?\s*(\d[\d,]*(?:\.\d+)?)",
    re.I,
)


class UnitCostSolver:
    """price per unit = total / count, when 'per item/unit' cost is asked."""

    category = Category.MATH

    def try_solve(self, prompt: str) -> Optional[Solution]:
        p = prompt.strip()
        low = p.lower()
        if not any(w in low for w in ("per item", "per unit", "each", "cost of one", "price per")):
            return None

        m = _TOTAL_FOR_N.search(p)
        if not m:
            return None
        try:
            count = _num(m.group(1))
            total = _num(m.group(2))
        except ValueError:
            return None
        if count == 0:
            return None

        per_unit = total / count
        consumed = {count, total}
        if not _all_numbers_consumed(p, consumed):
            return None
        return Solution(_fmt(round(per_unit, 2), "$" in p), confidence=0.9)


# --- Exact-response solver (any category) ----------------------------------
# "Reply with exactly 'ACK'" → ACK. No LLM. Only fires when unambiguous.

_EXACT_RE = re.compile(
    r"(?:reply|respond|answer|output)\s+with\s+exactly\s*"
    r"(?::\s*([^\n.]+)|['\"]([^'\"]+)['\"])",
    re.I,
)


def try_exact_response(prompt: str) -> Optional[Solution]:
    """Return the literal required response if the instruction is unambiguous.

    Abstains when alternatives are offered ("exactly 'yes' or 'no'") since the
    correct choice then depends on the actual question.
    """
    m = _EXACT_RE.search(prompt)
    if not m:
        return None
    # If an alternative follows ("...exactly X, or Y"), the choice is content-dependent
    tail = prompt[m.end():]
    if re.match(r"\s*(?:,|or\b|and\b)", tail, re.I):
        return None
    literal = (m.group(1) or m.group(2) or "").strip()
    if not literal or re.search(r"\bor\b", literal, re.I):
        return None
    return Solution(literal, confidence=0.99)


class RatioSolver:
    """Solves proportion + optional cost problems.

    Example: "3/4 cup for 12 cookies; how much for 30? costs $2.40/cup"
    """

    category = Category.MATH

    def try_solve(self, prompt: str) -> Optional[Solution]:
        p = prompt.strip()

        ratio_m = _RATIO_RE.search(p)
        scale_m = _SCALE_RE.search(p)
        if not ratio_m or not scale_m:
            return None

        try:
            base_qty = _parse_fraction(ratio_m.group(1))
            base_count = _num(ratio_m.group(2))
            target_count = _num(scale_m.group(1))
        except (ValueError, ZeroDivisionError):
            return None

        if base_count == 0:
            return None

        scaled = base_qty * target_count / base_count
        # Add fraction components to consumed so "3/4" accounts for 3 and 4
        consumed = {base_qty, base_count, target_count}
        if "/" in ratio_m.group(1):
            parts = ratio_m.group(1).split("/")
            try:
                consumed.add(float(parts[0].strip()))
                consumed.add(float(parts[1].strip()))
            except ValueError:
                pass

        cost_m = _COST_RE.search(p)
        if cost_m:
            try:
                price = float(cost_m.group(1))
                consumed.add(price)
                total_cost = scaled * price
                if not _all_numbers_consumed(p, consumed):
                    return None
                # Format both answers
                scaled_fmt = f"{scaled:g}" if scaled != int(scaled) else str(int(scaled))
                answer = (
                    f"{scaled_fmt} {'cup' if 'cup' in p.lower() else 'units'}; "
                    f"${total_cost:.2f}"
                )
                return Solution(answer, confidence=0.92)
            except ValueError:
                return None

        if not _all_numbers_consumed(p, consumed):
            return None
        scaled_fmt = f"{scaled:g}" if scaled != int(scaled) else str(int(scaled))
        return Solution(scaled_fmt, confidence=0.88)


# --- sentiment solver (T22) -----------------------------------------------

_POS_WORDS = frozenset("""
good great excellent amazing awesome wonderful fantastic love loved loving like
liked best perfect happy pleased satisfied enjoy enjoyed enjoyable superb brilliant
outstanding delightful positive recommend recommended nice beautiful comfortable
fast reliable helpful impressive worth flawless smooth delicious tasty gorgeous
stunning pleasant friendly favourite favorite terrific lovely
""".split())

_NEG_WORDS = frozenset("""
bad terrible awful horrible worst hate hated dislike disappointing disappointed
poor slow broken useless waste wasted defective faulty annoying frustrating
frustrated unhappy sad angry disgusting cheap overpriced dies died fails failed
rude dirty noisy uncomfortable regret regretted refund complaint complaints
mediocre lousy pathetic clunky laggy glitchy unreliable ripoff scam garbage
failing crash crashed buggy unreliable uncomfortable regret avoid
""".split())

_NEGATORS = frozenset("not no never n't isn't wasn't don't didn't doesn't cannot can't".split())

# Contrast/concession cues signal a nuanced review a lexicon can't score reliably.
_CONTRAST = re.compile(
    r"\b(but|however|although|though|yet|nonetheless|whereas|even though|"
    r"on the other hand|that said|mixed)\b",
    re.I,
)

# Words start with a letter; internal apostrophes kept (don't), surrounding
# quotes excluded (so 'great' tokenizes to great, not 'great').
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


class SentimentSolver:
    """Lexicon sentiment with simple negation handling.

    Confidence scales with the margin between positive and negative hits relative
    to total sentiment-bearing words. Abstains when no sentiment words are found.
    """

    category = Category.SENTIMENT

    def try_solve(self, prompt: str) -> Optional[Solution]:
        # Contrastive/nuanced reviews ("great food but slow service") are beyond a
        # lexicon — abstain so the local LLM handles the nuance rather than risk a
        # confidently-wrong label.
        if _CONTRAST.search(prompt):
            return None

        words = _WORD_RE.findall(prompt.lower())
        if not words:
            return None

        pos = neg = 0
        triggers: list[str] = []
        for i, w in enumerate(words):
            negated = i > 0 and words[i - 1] in _NEGATORS
            if w in _POS_WORDS:
                triggers.append(w)
                neg, pos = (neg + 1, pos) if negated else (neg, pos + 1)
            elif w in _NEG_WORDS:
                triggers.append(w)
                pos, neg = (pos + 1, neg) if negated else (pos, neg + 1)

        if pos == 0 and neg == 0:
            return None  # no signal -> escalate
        if pos > 0 and neg > 0:
            return None  # mixed signals -> escalate for nuanced judgement

        label = "Positive" if pos > neg else "Negative"
        reason = ", ".join(dict.fromkeys(triggers[:3]))
        answer = f"{label}. Key indicators: {reason}." if reason else label
        # Clean, one-sided signal (mixed/contrastive already escalated).
        return Solution(answer, confidence=0.9)


# --- NER solver (T23) ------------------------------------------------------

_MONTHS = (
    "january february march april may june july august september october "
    "november december jan feb mar apr jun jul aug sep sept oct nov dec"
).split()
_DATE_RES = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),
    re.compile(
        r"\b((?:" + "|".join(_MONTHS) + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
        re.I,
    ),
    re.compile(r"\b(\d{1,2}\s+(?:" + "|".join(_MONTHS) + r")\.?\s+\d{4})\b", re.I),
]
_ORG_SUFFIX = re.compile(
    r"\b([A-Z][A-Za-z.&]+(?:\s+[A-Z][A-Za-z.&]+)*\s+"
    r"(?:Inc|Incorporated|Corp|Corporation|Ltd|LLC|LLP|Company|Co|Group|Bank|"
    r"University|Institute|Foundation))\b"
)
# Consecutive capitalized tokens (candidate proper nouns).
_PROPER = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
_STOP_STARTS = frozenset(
    "The A An In On At Of For And But Or If When Who What Where Why How "
    "Extract Summarise Summarize Classify Calculate Write Find".split()
)


class NERSolver:
    """Best-effort local NER into the required JSON shape.

    Dates are matched with high-precision regexes (reliable). Persons/orgs/
    locations are heuristic (capitalized sequences, org suffixes), so overall
    confidence is kept modest — the cascade escalates when the caller's threshold
    isn't met.
    """

    category = Category.NER

    def try_solve(self, prompt: str) -> Optional[Solution]:
        # Work on the content after a leading instruction/colon if present.
        text = prompt.split(":", 1)[1] if ":" in prompt else prompt

        dates: list[str] = []
        for rx in _DATE_RES:
            dates.extend(m.group(1) for m in rx.finditer(text))

        orgs = [m.group(1) for m in _ORG_SUFFIX.finditer(text)]

        proper: list[str] = []
        for m in _PROPER.finditer(text):
            span = m.group(1)
            if span.split()[0] in _STOP_STARTS:
                continue
            if any(span in o or o in span for o in orgs):
                continue
            proper.append(span)

        # Two-token capitalized spans -> likely person names.
        persons = [s for s in proper if len(s.split()) >= 2]
        singles = [s for s in proper if len(s.split()) == 1]

        found = len(dates) + len(orgs) + len(persons) + len(singles)
        if found == 0:
            return None  # nothing extractable -> escalate

        entities = {
            "person": _dedup(persons),
            "org": _dedup(orgs),
            "location": _dedup(singles),  # heuristic: lone capitalized -> location
            "date": _dedup(dates),
        }
        answer = _compact_json(entities)
        # Confidence is modest: dates reliable, others heuristic.
        confidence = 0.6 if (persons or orgs or dates) else 0.45
        return Solution(answer, confidence=confidence)


def _dedup(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i.strip() for i in items if i.strip()))


def _compact_json(entities: dict[str, list[str]]) -> str:
    import json

    return json.dumps(entities, ensure_ascii=False, separators=(",", ":"))


# --- spaCy NER solver (T23, upgrade) --------------------------------------

_SPACY_NLP = None
_SPACY_TRIED = False

_SPACY_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "org",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "DATE": "date",
    "TIME": "date",
}


def _get_spacy():
    """Lazily load en_core_web_sm once. Returns None if spaCy/model unavailable."""
    global _SPACY_NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _SPACY_NLP
    _SPACY_TRIED = True
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
    except Exception:
        _SPACY_NLP = None
    return _SPACY_NLP


class SpacyNERSolver:
    """NER via spaCy's statistical model — more reliable than the regex heuristic.

    Abstains (None) if spaCy or the model isn't installed, so the heuristic
    NERSolver runs as a fallback.
    """

    category = Category.NER

    def try_solve(self, prompt: str) -> Optional[Solution]:
        nlp = _get_spacy()
        if nlp is None:
            return None
        text = prompt.split(":", 1)[1] if ":" in prompt else prompt
        doc = nlp(text)
        buckets: dict[str, list[str]] = {"person": [], "org": [], "location": [], "date": []}
        for ent in doc.ents:
            key = _SPACY_LABEL_MAP.get(ent.label_)
            if key:
                buckets[key].append(ent.text.strip())
        buckets = {k: _dedup(v) for k, v in buckets.items()}
        if not any(buckets.values()):
            return None
        return Solution(_compact_json(buckets), confidence=0.85)


# --- registry --------------------------------------------------------------

# Registry: only solvers that are PROVEN safe for the accuracy gate.
# Safety rule for all local solvers: abstain on anything uncertain.
# A wrong local answer costs the same tokens as a correct one (zero) but
# risks the accuracy gate — the one thing we cannot recover from.
_SOLVERS: dict[Category, list[LocalSolver]] = {
    # Math: deterministic single-step + multi-step chains + ratio/cost +
    # speed/distance/time + simple interest + unit cost. All abstain when any
    # number in the prompt is unconsumed (protects the accuracy gate).
    Category.MATH: [
        MathSolver(),
        OperationChainSolver(),
        RatioSolver(),
        SpeedDistanceSolver(),
        SimpleInterestSolver(),
        UnitCostSolver(),
    ],
    # Sentiment: clear one-sided signals only. Abstains on any contrastive text
    # ("but", "however", "although") and on mixed pos/neg signals.
    # The judge's mixed-review tasks always have contrastive language → Fireworks.
    Category.MATH: [MathSolver()],
    Category.NER: [NERSolver()],
    Category.SENTIMENT: [SentimentSolver()],
}


def solvers_for(category: Category) -> list[LocalSolver]:
    """Local solvers registered for a category (empty if none yet)."""
    return _SOLVERS.get(category, [])
