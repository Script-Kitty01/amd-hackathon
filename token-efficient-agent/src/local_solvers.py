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
# Signals of a multi-step problem the naive solver must NOT attempt.
_MULTISTEP_WORDS = ("then", "additional", "additionally", "after that", "followed by", "subsequent")


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


# --- sentiment solver (T22) -----------------------------------------------

_POS_WORDS = frozenset("""
good great excellent amazing awesome wonderful fantastic love loved loving like
liked best perfect happy pleased satisfied enjoy enjoyed enjoyable superb brilliant
outstanding delightful positive recommend recommended nice beautiful comfortable
fast reliable helpful impressive worth flawless smooth
""".split())

_NEG_WORDS = frozenset("""
bad terrible awful horrible worst hate hated dislike disappointing disappointed
poor slow broken useless waste wasted defective faulty annoying frustrating
frustrated unhappy sad angry disgusting cheap overpriced dies died fails failed
failing crash crashed buggy unreliable uncomfortable regret avoid
""".split())

_NEGATORS = frozenset("not no never n't isn't wasn't don't didn't doesn't cannot can't".split())

_WORD_RE = re.compile(r"[a-z']+")


class SentimentSolver:
    """Lexicon sentiment with simple negation handling.

    Confidence scales with the margin between positive and negative hits relative
    to total sentiment-bearing words. Abstains when no sentiment words are found.
    """

    category = Category.SENTIMENT

    def try_solve(self, prompt: str) -> Optional[Solution]:
        words = _WORD_RE.findall(prompt.lower())
        if not words:
            return None

        pos = neg = 0
        triggers: list[str] = []
        for i, w in enumerate(words):
            negated = i > 0 and words[i - 1] in _NEGATORS
            if w in _POS_WORDS:
                triggers.append(w)
                if negated:
                    neg += 1
                else:
                    pos += 1
            elif w in _NEG_WORDS:
                triggers.append(w)
                if negated:
                    pos += 1
                else:
                    neg += 1

        total = pos + neg
        if total == 0:
            return None  # no signal -> escalate

        if pos > neg:
            label = "Positive"
        elif neg > pos:
            label = "Negative"
        else:
            label = "Neutral"

        margin = abs(pos - neg) / total
        confidence = 0.55 + 0.4 * margin  # 0.55..0.95
        reason = ", ".join(dict.fromkeys(triggers[:3]))
        answer = f"{label}. Key indicators: {reason}." if reason else label
        return Solution(answer, confidence=round(confidence, 3))


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

# NER: try spaCy first, fall back to the regex heuristic if the model is absent.
_SOLVERS: dict[Category, list[LocalSolver]] = {
    Category.MATH: [MathSolver()],
    Category.SENTIMENT: [SentimentSolver()],
    Category.NER: [SpacyNERSolver(), NERSolver()],
}


def solvers_for(category: Category) -> list[LocalSolver]:
    """Local solvers registered for a category (empty if none yet)."""
    return _SOLVERS.get(category, [])
