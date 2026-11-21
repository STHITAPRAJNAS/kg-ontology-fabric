from __future__ import annotations

import hashlib
import hmac
import os
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PIIRedactionMode(str, Enum):
    """Controls how detected PII spans are transformed before embedding or graph extraction."""

    REPLACE_TAG = "replace_tag"  # e.g. <US_SSN>, <EMAIL_ADDRESS>, <BEARER_JWT_TOKEN>
    HMAC_PSEUDONYMIZE = "hmac_pseudonymize"  # Deterministic keyed token e.g. <EMAIL_ADDRESS_9f3a1c2b> for safe cross-KB linkage
    MASK = "mask"  # Partial masking e.g. ***-**-6789


class PIIDetectionSpan(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    engine: str = "presidio"


class PIIRedactionResult(BaseModel):
    scrubbed_text: str
    redaction_counts: dict[str, int] = Field(default_factory=dict)
    total_redactions: int = 0
    engine_used: str = "presidio_compatible_fallback"
    protected_technical_tokens_preserved: int = 0


# Technical enterprise identifiers that stock NLP/Presidio PII models often misclassify
# as PHONE_NUMBER, US_SSN, US_DRIVER_LICENSE, or IP_ADDRESS.
_PROTECTED_TECHNICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bCVE-\d{4}-\d{3,7}\b", re.I),
    re.compile(r"\bINC-\d{3,8}\b", re.I),
    re.compile(r"\bCC-\d{2,5}(?:-[A-Za-z0-9]+)*\b", re.I),
    re.compile(r"\bPCI-DSS-\d+(?:\.\d+)*\b", re.I),
    re.compile(r"\bGDPR-Article-\d+(?:\.\d+)*\b", re.I),
    re.compile(r"\bISO-\d{4,5}(?:-\d+)?\b", re.I),
    re.compile(r"\bSOC2-Type-I{1,2}\b", re.I),
    re.compile(r"\bWAF-[A-Za-z0-9\-]+\b", re.I),
    re.compile(r"\bSLA-[A-Za-z0-9\-]+\b", re.I),
    re.compile(r"\b(?:us|eu|ap|sa|ca|me|af)-(?:east|west|central|north|south|northeast|southeast)-\d\b", re.I),
]

_PEM_HEADER_REGEX = r"-{5}BEGIN (?:RSA |EC |OPENSSH )?PRIVATE" + r" KEY-{5}"

# Custom enterprise secret & PII patterns registered BOTH as Presidio PatternRecognizers
# and used by the zero-dependency Presidio-compatible fallback engine.
_ENTERPRISE_PII_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    ("BEARER_JWT_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+(?:\.[A-Za-z0-9\-_]+)?\b"), 0.99),
    ("BEARER_JWT_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9\-_\.]{18,}\b"), 0.95),
    ("AWS_ACCESS_KEY_ID", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0.99),
    ("PRIVATE_KEY_HEADER", re.compile(_PEM_HEADER_REGEX), 0.99),
    (
        "GENERIC_API_SECRET",
        re.compile(r"\b(?:api[_-]?key|client[_-]?secret|db[_-]?password|password|secret[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_\.\/\+]{10,}['\"]?", re.I),
        0.95,
    ),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), 0.95),
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0.95),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), 0.92),
    ("PHONE_NUMBER", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), 0.85),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), 0.90),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"), 0.85),
]


class PresidioPIIRedactor:
    """
    Enterprise-grade PII & Secret Redaction Engine powered by Microsoft Presidio
    (`presidio-analyzer` + `presidio-anonymizer`) with:
      1. Custom Enterprise Presidio `PatternRecognizer`s for JWT Bearer tokens, AWS IAM keys,
         PEM key headers, and inline API credentials.
      2. Technical Entity Protection Guard (`_PROTECTED_TECHNICAL_PATTERNS`) so Presidio
         NEVER corrupts Knowledge Graph identifiers like `CVE-2026-9981`, `PCI-DSS-4.0`,
         `CC-209-Global`, `WAF-Shield-Rule-904`, or `us-east-1`.
      3. Configurable transformation modes: `REPLACE_TAG` (`<US_SSN>`),
         `HMAC_PSEUDONYMIZE` (`<EMAIL_ADDRESS_a91f3c2d>` for privacy-preserving entity linkage),
         or `MASK`.
      4. Seamless fallback to Presidio-compatible regex/Luhn recognizers when `presidio-analyzer`
         or spaCy models are not installed in lightweight/offline test environments.
    """

    DEFAULT_ENTITIES: list[str] = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IBAN_CODE",
        "IP_ADDRESS",
        "BEARER_JWT_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "PRIVATE_KEY_HEADER",
        "GENERIC_API_SECRET",
    ]

    def __init__(
        self,
        mode: PIIRedactionMode = PIIRedactionMode.REPLACE_TAG,
        entities: list[str] | None = None,
        score_threshold: float = 0.70,
        hmac_secret_key: str | None = None,
        language: str = "en",
        custom_allow_list: list[str] | None = None,
        prefer_presidio: bool = True,
    ):
        self.mode = mode
        self.entities = entities or list(self.DEFAULT_ENTITIES)
        self.score_threshold = score_threshold
        resolved_key = hmac_secret_key or os.environ.get("PII_HMAC_KEY", "ephemeral-local-hmac-salt")
        self.hmac_secret_key = resolved_key.encode("utf-8")
        self.language = language
        self.custom_allow_list = set(custom_allow_list or [])
        self._presidio_analyzer: Any = None
        self._presidio_anonymizer: Any = None
        self._using_native_presidio = False

        if prefer_presidio:
            self._try_init_presidio()

    def _try_init_presidio(self) -> None:
        """
        Initializes Microsoft Presidio `AnalyzerEngine` & `AnonymizerEngine` with custom
        Enterprise Secret `PatternRecognizer`s if `presidio-analyzer` & `presidio-anonymizer`
        are installed in the runtime environment (e.g., EKS production container).
        """
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_anonymizer import AnonymizerEngine

            analyzer = AnalyzerEngine()
            # Register custom enterprise recognizers for secrets Presidio does not ship natively
            custom_defs = [
                ("BEARER_JWT_TOKEN", r"\bBearer\s+[A-Za-z0-9\-_\.]{18,}\b", 0.95),
                ("AWS_ACCESS_KEY_ID", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0.99),
                ("PRIVATE_KEY_HEADER", _PEM_HEADER_REGEX, 0.99),
                (
                    "GENERIC_API_SECRET",
                    r"\b(?:api[_-]?key|client[_-]?secret|db[_-]?password|password|secret[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9\-_\.\/\+]{10,}['\"]?",
                    0.95,
                ),
            ]
            for entity_name, regex_str, score in custom_defs:
                recognizer = PatternRecognizer(
                    supported_entity=entity_name,
                    patterns=[Pattern(name=f"{entity_name.lower()}_pattern", regex=regex_str, score=score)],
                )
                analyzer.registry.add_recognizer(recognizer)

            self._presidio_analyzer = analyzer
            self._presidio_anonymizer = AnonymizerEngine()
            self._using_native_presidio = True
        except Exception:
            # Presidio or spaCy model not installed locally -> fall back cleanly
            self._presidio_analyzer = None
            self._presidio_anonymizer = None
            self._using_native_presidio = False

    @property
    def engine_name(self) -> str:
        return "microsoft_presidio" if self._using_native_presidio else "presidio_compatible_fallback"

    def _find_protected_spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for pat in _PROTECTED_TECHNICAL_PATTERNS:
            for m in pat.finditer(text):
                spans.append((m.start(), m.end()))
        for token in self.custom_allow_list:
            if not token:
                continue
            for m in re.finditer(re.escape(token), text, re.I):
                spans.append((m.start(), m.end()))
        return spans

    @staticmethod
    def _overlaps_any(start: int, end: int, protected_spans: list[tuple[int, int]]) -> bool:
        for p_start, p_end in protected_spans:
            if start < p_end and end > p_start:
                return True
        return False

    @staticmethod
    def _luhn_check(digits_only: str) -> bool:
        if not digits_only.isdigit() or len(digits_only) < 13 or len(digits_only) > 19:
            return False
        total = 0
        reverse_digits = digits_only[::-1]
        for idx, ch in enumerate(reverse_digits):
            n = int(ch)
            if idx % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    def _format_replacement(self, entity_type: str, raw_value: str) -> str:
        if self.mode == PIIRedactionMode.HMAC_PSEUDONYMIZE:
            digest = hmac.new(
                self.hmac_secret_key,
                f"{entity_type}:{raw_value.strip().lower()}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:8]
            return f"<{entity_type}_{digest}>"
        if self.mode == PIIRedactionMode.MASK:
            if len(raw_value) > 4:
                return ("*" * (len(raw_value) - 4)) + raw_value[-4:]
            return "****"
        return f"<{entity_type}>"

    def analyze_and_redact(self, text: str) -> PIIRedactionResult:
        if not text:
            return PIIRedactionResult(scrubbed_text="", engine_used=self.engine_name)

        protected_spans = self._find_protected_spans(text)
        detected: list[PIIDetectionSpan] = []

        if self._using_native_presidio and self._presidio_analyzer is not None:
            raw_results = self._presidio_analyzer.analyze(
                text=text,
                entities=self.entities,
                language=self.language,
                score_threshold=self.score_threshold,
            )
            for res in raw_results:
                if self._overlaps_any(res.start, res.end, protected_spans):
                    continue
                detected.append(
                    PIIDetectionSpan(
                        entity_type=res.entity_type,
                        start=res.start,
                        end=res.end,
                        score=float(res.score),
                        engine="microsoft_presidio",
                    )
                )
        else:
            allowed_entity_set = set(self.entities)
            for entity_type, pattern, score in _ENTERPRISE_PII_PATTERNS:
                if entity_type not in allowed_entity_set or score < self.score_threshold:
                    continue
                for m in pattern.finditer(text):
                    if self._overlaps_any(m.start(), m.end(), protected_spans):
                        continue
                    matched_str = m.group(0)
                    if entity_type == "CREDIT_CARD":
                        digits = re.sub(r"\D", "", matched_str)
                        if not self._luhn_check(digits):
                            continue
                    detected.append(
                        PIIDetectionSpan(
                            entity_type=entity_type,
                            start=m.start(),
                            end=m.end(),
                            score=score,
                            engine="presidio_compatible_fallback",
                        )
                    )

        # Deduplicate overlapping spans (keep highest score, then longest span)
        detected.sort(key=lambda s: (s.start, -s.score, -(s.end - s.start)))
        non_overlapping: list[PIIDetectionSpan] = []
        last_end = -1
        for span in detected:
            if span.start >= last_end:
                non_overlapping.append(span)
                last_end = span.end

        # Apply replacements from right to left so character offsets remain exact
        scrubbed = text
        counts: dict[str, int] = {}
        for span in reversed(non_overlapping):
            raw_val = scrubbed[span.start : span.end]
            replacement = self._format_replacement(span.entity_type, raw_val)
            scrubbed = scrubbed[: span.start] + replacement + scrubbed[span.end :]
            counts[span.entity_type] = counts.get(span.entity_type, 0) + 1

        return PIIRedactionResult(
            scrubbed_text=scrubbed,
            redaction_counts=counts,
            total_redactions=sum(counts.values()),
            engine_used=self.engine_name,
            protected_technical_tokens_preserved=len(protected_spans),
        )

    def scrub(self, text: str) -> str:
        return self.analyze_and_redact(text).scrubbed_text


# Default singleton instance + backward-compatible static wrapper
_DEFAULT_REDACTOR = PresidioPIIRedactor(mode=PIIRedactionMode.REPLACE_TAG)


class PIIRedactor:
    """Backward-compatible facade delegating to `PresidioPIIRedactor`."""

    @classmethod
    def scrub(cls, text: str) -> str:
        return _DEFAULT_REDACTOR.scrub(text)

    @classmethod
    def analyze_and_redact(cls, text: str) -> PIIRedactionResult:
        return _DEFAULT_REDACTOR.analyze_and_redact(text)
