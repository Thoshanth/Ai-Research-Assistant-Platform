import re
from backend.guardrails.pii_detector import redact_pii
from backend.logger import get_logger

logger = get_logger("guardrails.output")

# Phrases that suggest the LLM is hallucinating or uncertain
HALLUCINATION_SIGNALS = [
    r"i think\s+it\s+(might|could|may)",
    r"i('m|\s+am)\s+not\s+sure\s+but",
    r"i\s+believe\s+but\s+(i('m|\s+am)\s+not\s+certain)",
    r"as\s+far\s+as\s+i\s+know",
    r"i\s+cannot\s+verify",
    r"i\s+don'?t\s+have\s+(access|information)",
]

_hallucination_compiled = [
    re.compile(p, re.IGNORECASE) for p in HALLUCINATION_SIGNALS
]


class OutputGuardResult:
    def __init__(
        self,
        safe_text: str,
        was_modified: bool,
        pii_redacted: dict,
        hallucination_warning: bool,
        warnings: list[str],
    ):
        self.safe_text = safe_text
        self.was_modified = was_modified
        self.pii_redacted = pii_redacted
        self.hallucination_warning = hallucination_warning
        self.warnings = warnings

    def to_dict(self):
        return {
            "safe_text": self.safe_text,
            "was_modified": self.was_modified,
            "pii_redacted": self.pii_redacted,
            "hallucination_warning": self.hallucination_warning,
            "warnings": self.warnings,
        }


def check_output(text: str, expect_citations: bool = False) -> OutputGuardResult:
    """
    Runs all output checks:
    1. PII redaction
    2. Hallucination signal detection
    3. Citation check (optional — for RAG responses)

    Always returns a safe version of the text,
    never blocks output entirely.
    """
    logger.debug(f"Output guard checking | chars={len(text)}")

    warnings = []
    was_modified = False

    # Check 1: Redact PII from output
    safe_text, redactions = redact_pii(text)
    if redactions:
        was_modified = True
        warnings.append(f"PII redacted from output: {list(redactions.keys())}")
        logger.warning(f"PII redacted from output | types={list(redactions.keys())}")

    # Check 2: Hallucination signals
    hallucination_warning = False
    for pattern in _hallucination_compiled:
        if pattern.search(safe_text):
            hallucination_warning = True
            warnings.append("Response may contain uncertain or unverified information")
            logger.warning("Hallucination signal detected in output")
            break

    # Check 3: Citation check for RAG responses
    if expect_citations:
        has_citation = bool(re.search(r"\[Source\s*\d+", safe_text))
        if not has_citation:
            warnings.append("RAG response missing source citations")
            logger.warning("RAG response has no citations")

    # Check 4: Empty response
    if not safe_text or len(safe_text.strip()) < 10:
        warnings.append("Response is empty or too short")
        safe_text = "I was unable to generate a response. Please try rephrasing your question."
        was_modified = True

    logger.debug(f"Output guard complete | modified={was_modified} | warnings={len(warnings)}")

    return OutputGuardResult(
        safe_text=safe_text,
        was_modified=was_modified,
        pii_redacted=redactions,
        hallucination_warning=hallucination_warning,
        warnings=warnings,
    )