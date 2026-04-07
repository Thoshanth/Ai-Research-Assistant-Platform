import re
from backend.logger import get_logger

logger = get_logger("guardrails.pii")

# ── PII Patterns ──────────────────────────────────────────────────
PII_PATTERNS = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "phone_international": re.compile(
        r"\+?[0-9]{1,3}[\s\-]?[0-9]{7,12}"
    ),
    "phone_india": re.compile(
        r"\b[6-9]\d{9}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d[ \-]?){13,16}\b"
    ),
    "aadhaar": re.compile(
        r"\b\d{4}\s\d{4}\s\d{4}\b"
    ),
    "pan_india": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
    ),
    "ip_address": re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
}


def detect_pii(text: str) -> dict:
    """
    Scans text for PII patterns.
    Returns a dict of found PII types and their matches.
    Does NOT return the actual values for security.
    """
    found = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[pii_type] = len(matches)
            logger.warning(
                f"PII detected | type={pii_type} | count={len(matches)}"
            )
    return found


def redact_pii(text: str) -> tuple[str, dict]:
    """
    Replaces PII in text with [REDACTED: type].
    Returns (redacted_text, dict of what was redacted).
    
    Used on LLM outputs — we let the answer through
    but remove any sensitive data before it reaches the user.
    """
    redacted = text
    redactions = {}

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(redacted)
        if matches:
            redacted = pattern.sub(f"[REDACTED:{pii_type.upper()}]", redacted)
            redactions[pii_type] = len(matches)
            logger.info(f"PII redacted | type={pii_type} | count={len(matches)}")

    return redacted, redactions


def has_pii(text: str) -> bool:
    return bool(detect_pii(text))