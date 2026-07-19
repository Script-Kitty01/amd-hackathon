"""PII (Personally Identifiable Information) scanner and masker.

Detects and optionally masks sensitive data before it leaves the organization:
  - Email addresses
  - Phone numbers (international formats)
  - SSN (US), Aadhaar (India), PAN (India)
  - Credit card numbers
  - IP addresses
  - Physical addresses (basic patterns)

Uses Microsoft Presidio for advanced detection when available,
falls back to regex patterns for zero-dependency operation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PIIMatch:
    """A detected PII entity."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float  # 0..1 confidence


@dataclass
class PIIScanResult:
    """Result of a PII scan."""
    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    masked_text: str = ""
    original_text: str = ""


# --- Regex-based PII patterns (zero-dependency fallback) ---

_PII_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    # (entity_type, pattern, confidence)
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0.95),
    ("PHONE_INTERNATIONAL", re.compile(r"\+\d{1,3}[\s-]?\d{3,14}[\s-]?\d{3,14}"), 0.85),
    ("PHONE_US", re.compile(r"\b\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), 0.80),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.90),
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), 0.75),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), 0.85),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), 0.85),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.90),
    # Basic address: number + street name + city/state/zip pattern
    ("ADDRESS", re.compile(
        r"\b\d{1,5}\s+[A-Za-z]+(?:\s+[A-Za-z]+)*"
        r"(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Blvd|Boulevard|Court|Ct|Way))"
        r"(?:[,\s]+[A-Za-z]+(?:[,\s]+[A-Z]{2})?(?:[,\s]+\d{5}(?:-\d{4})?)?)?\b",
        re.I,
    ), 0.60),
]

# Masking characters
_MASK_CHAR = "*"


class PIIScanner:
    """Scans text for PII and optionally masks it."""

    def __init__(self, use_presidio: bool = True) -> None:
        self._use_presidio = use_presidio
        self._presidio_analyzer = None
        self._presidio_anonymizer = None

        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                self._presidio_analyzer = AnalyzerEngine()
                self._presidio_anonymizer = AnonymizerEngine()
            except ImportError:
                pass  # Fall back to regex

    def scan(self, text: str) -> PIIScanResult:
        """Scan text for PII. Returns matches without modifying text."""
        matches = self._find_matches(text)
        return PIIScanResult(
            has_pii=len(matches) > 0,
            matches=matches,
            original_text=text,
            masked_text=text,
        )

    def scan_and_mask(self, text: str) -> PIIScanResult:
        """Scan text for PII and return a masked version."""
        if self._presidio_analyzer and self._presidio_anonymizer:
            return self._presidio_scan(text)

        matches = self._find_matches(text)
        masked = self._apply_masking(text, matches)
        return PIIScanResult(
            has_pii=len(matches) > 0,
            matches=matches,
            masked_text=masked,
            original_text=text,
        )

    def _find_matches(self, text: str) -> list[PIIMatch]:
        """Find all PII matches using regex patterns."""
        matches: list[PIIMatch] = []
        seen_spans: set[tuple[int, int]] = set()

        for entity_type, pattern, confidence in _PII_PATTERNS:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                # Avoid overlapping matches (keep higher confidence)
                overlapping = any(
                    not (span[1] <= s[0] or span[0] >= s[1])
                    for s in seen_spans
                )
                if overlapping:
                    continue
                seen_spans.add(span)
                matches.append(PIIMatch(
                    entity_type=entity_type,
                    text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    score=confidence,
                ))

        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches

    def _apply_masking(self, text: str, matches: list[PIIMatch]) -> str:
        """Replace PII spans with masked versions."""
        if not matches:
            return text

        result = list(text)
        for m in matches:
            # Keep first char, mask rest (e.g., "j***@g****.com")
            entity_text = m.text
            if m.entity_type == "EMAIL":
                at_idx = entity_text.index("@")
                masked = (
                    entity_text[0] + _MASK_CHAR * (at_idx - 1)
                    + "@"
                    + entity_text[at_idx + 1]
                    + _MASK_CHAR * (len(entity_text) - at_idx - 2)
                )
            elif m.entity_type in ("CREDIT_CARD", "SSN", "AADHAAR"):
                # Show last 4
                masked = _MASK_CHAR * (len(entity_text) - 4) + entity_text[-4:]
            elif m.entity_type == "PAN":
                # Show first 3, last 1
                masked = entity_text[:3] + _MASK_CHAR * 5 + entity_text[-1]
            elif m.entity_type == "PHONE_INTERNATIONAL":
                # Show country code + last 3
                masked = entity_text[:3] + _MASK_CHAR * (len(entity_text) - 6) + entity_text[-3:]
            else:
                # Generic: show first 2, mask rest
                masked = entity_text[:2] + _MASK_CHAR * max(1, len(entity_text) - 2)

            for i in range(m.start, m.end):
                if i - m.start < len(masked):
                    result[i] = masked[i - m.start]

        return "".join(result)

    def _presidio_scan(self, text: str) -> PIIScanResult:
        """Use Microsoft Presidio for advanced PII detection."""
        try:
            results = self._presidio_analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                    "CREDIT_CARD", "IP_ADDRESS", "PERSON",
                    "LOCATION", "US_DRIVER_LICENSE", "IBAN_CODE",
                ],
            )
            matches = [
                PIIMatch(
                    entity_type=r.entity_type,
                    text=text[r.start:r.end],
                    start=r.start,
                    end=r.end,
                    score=r.score,
                )
                for r in results
            ]

            anonymized = self._presidio_anonymizer.anonymize(
                text=text, analyzer_results=results
            )

            return PIIScanResult(
                has_pii=len(matches) > 0,
                matches=matches,
                masked_text=anonymized.text,
                original_text=text,
            )
        except Exception:
            # Fall back to regex on Presidio failure
            return self.scan_and_mask.__wrapped__(self, text) if hasattr(self.scan_and_mask, '__wrapped__') else self._regex_fallback(text)

    def _regex_fallback(self, text: str) -> PIIScanResult:
        matches = self._find_matches(text)
        masked = self._apply_masking(text, matches)
        return PIIScanResult(
            has_pii=len(matches) > 0,
            matches=matches,
            masked_text=masked,
            original_text=text,
        )
