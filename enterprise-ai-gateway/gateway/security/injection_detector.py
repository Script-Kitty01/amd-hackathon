"""Prompt injection detection.

Detects common prompt injection patterns:
  - "Ignore previous instructions"
  - "You are now DAN" / jailbreak attempts
  - "Reveal your system prompt"
  - "Forget all previous"
  - Role-playing as a different system
  - Hidden text / encoding tricks
  - Repetition attacks

Returns a risk score 0..1 and a list of detected patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InjectionMatch:
    pattern_name: str
    matched_text: str
    severity: str  # "low" | "medium" | "high" | "critical"


@dataclass
class InjectionResult:
    is_injection: bool
    risk_score: float  # 0..1
    matches: list[InjectionMatch] = field(default_factory=list)
    blocked: bool = False


# --- Injection patterns ---

# Critical: direct override attempts
_CRITICAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "direct_override",
        re.compile(
            r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above|earlier|"
            r"the\s+)?\s*(?:instructions?|prompts?|commands?|directives?|rules?|"
            r"context|conversation)",
            re.I,
        ),
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"(?:reveal|tell\s+me|show\s+me|print|output|display|what\s+(?:is|are))\s+"
            r"(?:your\s+)?(?:system\s+(?:prompt|message|instructions?)|"
            r"initial\s+(?:prompt|instructions?)|hidden\s+(?:prompt|instructions?))",
            re.I,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you\s+are)|"
            r"you\s+will\s+now\s+(?:be|act|play)|from\s+now\s+on\s+you\s+(?:are|will))"
            r"\s+(?:DAN|jailbreak|evil|unethical|without\s+restrictions?|"
            r"no\s+(?:rules?|limits?|restrictions?|filters?|ethics?))",
            re.I,
        ),
    ),
]

# High severity: manipulation attempts
_HIGH_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "token_leak",
        re.compile(
            r"(?:what\s+(?:is|are)\s+your\s+|reveal\s+your\s+|tell\s+me\s+your\s+|"
            r"show\s+(?:me\s+)?your\s+|give\s+me\s+your\s+)"
            r"(?:API\s+key|token|secret|password|credential)",
            re.I,
        ),
    ),
    (
        "harmful_request",
        re.compile(
            r"\b(?:how\s+to\s+(?:make|build|create|manufacture)\s+(?:a\s+)?"
            r"(?:bomb|weapon|drug|poison|malware|virus|ransomware)|"
            r"instructions?\s+for\s+(?:hacking|cracking|phishing))",
            re.I,
        ),
    ),
    (
        "encoding_trick",
        re.compile(
            r"(?:base64|rot13|hex\s+encode|decode\s+(?:this|the\s+following)|"
            r"translate\s+from\s+(?:base64|hex|binary))",
            re.I,
        ),
    ),
]

# Medium severity: suspicious patterns
_MEDIUM_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "repetition_attack",
        re.compile(r"(.{20,}?)\1{4,}"),  # Same 20+ char string repeated 5+ times
    ),
    (
        "delimiter_override",
        re.compile(
            r"={3,}.*?(?:system|instruction|prompt|command).*?={3,}",
            re.I | re.S,
        ),
    ),
    (
        "nested_instruction",
        re.compile(
            r"(?:new\s+instructions?|updated\s+instructions?|revised\s+instructions?|"
            r"override\s+instructions?)\s*:",
            re.I,
        ),
    ),
]

# Severity weights for risk score calculation
_SEVERITY_WEIGHTS = {
    "critical": 0.35,
    "high": 0.25,
    "medium": 0.15,
    "low": 0.05,
}

# Thresholds
_BLOCK_THRESHOLD = 0.5   # Block request if risk >= this
_WARN_THRESHOLD = 0.25   # Log warning if risk >= this


class InjectionDetector:
    """Detects prompt injection attempts."""

    def __init__(
        self,
        block_threshold: float = _BLOCK_THRESHOLD,
        warn_threshold: float = _WARN_THRESHOLD,
    ) -> None:
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold

    def scan(self, prompt: str) -> InjectionResult:
        """Scan a prompt for injection patterns."""
        matches: list[InjectionMatch] = []

        # Check critical patterns
        for name, pattern in _CRITICAL_PATTERNS:
            for m in pattern.finditer(prompt):
                matches.append(InjectionMatch(
                    pattern_name=name,
                    matched_text=m.group(0)[:100],
                    severity="critical",
                ))

        # Check high patterns
        for name, pattern in _HIGH_PATTERNS:
            for m in pattern.finditer(prompt):
                matches.append(InjectionMatch(
                    pattern_name=name,
                    matched_text=m.group(0)[:100],
                    severity="high",
                ))

        # Check medium patterns
        for name, pattern in _MEDIUM_PATTERNS:
            for m in pattern.finditer(prompt):
                matches.append(InjectionMatch(
                    pattern_name=name,
                    matched_text=m.group(0)[:100],
                    severity="medium",
                ))

        # Calculate risk score (capped at 1.0)
        risk = sum(
            _SEVERITY_WEIGHTS.get(m.severity, 0.05)
            for m in matches
        )
        risk = min(1.0, risk)

        return InjectionResult(
            is_injection=risk >= self._warn_threshold,
            risk_score=round(risk, 3),
            matches=matches,
            blocked=risk >= self._block_threshold,
        )
