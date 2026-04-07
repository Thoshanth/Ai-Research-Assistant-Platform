import re
from backend.logger import get_logger

logger = get_logger("guardrails.input")

# ── Prompt Injection Patterns ─────────────────────────────────────
# These are phrases commonly used to override system prompts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|context)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are\s+)?a",
    r"pretend\s+(you\s+are|to\s+be)",
    r"your\s+new\s+(instructions|rules|purpose)",
    r"disregard\s+(all\s+)?(previous|prior|your)",
    r"override\s+(your\s+)?(instructions|rules|system)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"developer\s+mode",
]

# ── Harmful Content Patterns ──────────────────────────────────────
HARMFUL_PATTERNS = [
    r"\b(how\s+to\s+)?(make|build|create|synthesize)\s+(a\s+)?(bomb|weapon|explosive|poison|drug)",
    r"\b(hack|exploit|attack|breach)\s+(a\s+)?(system|server|database|network)",
    r"child\s+(abuse|exploitation|pornography)",
    r"\b(kill|murder|harm|hurt)\s+(a\s+)?(person|people|human|someone)",
    r"(suicide|self.harm)\s+(method|way|how)",
]

# ── Compiled patterns for speed ───────────────────────────────────
_injection_compiled = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]
_harmful_compiled = [
    re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS
]


class InputGuardResult:
    def __init__(self, is_safe: bool, reason: str = "", threat_type: str = ""):
        self.is_safe = is_safe
        self.reason = reason
        self.threat_type = threat_type

    def to_dict(self):
        return {
            "is_safe": self.is_safe,
            "reason": self.reason,
            "threat_type": self.threat_type,
        }


def check_input(text: str) -> InputGuardResult:
    """
    Runs all input checks in sequence.
    Returns on the first failed check for efficiency.
    """
    logger.debug(f"Input guard checking | chars={len(text)}")

    # Check 1: Empty or too short
    if not text or len(text.strip()) < 2:
        return InputGuardResult(
            is_safe=False,
            reason="Input is empty or too short",
            threat_type="invalid_input",
        )

    # Check 2: Too long (prevent context stuffing attacks)
    if len(text) > 5000:
        return InputGuardResult(
            is_safe=False,
            reason="Input exceeds maximum length of 5000 characters",
            threat_type="input_too_long",
        )

    # Check 3: Prompt injection
    for pattern in _injection_compiled:
        if pattern.search(text):
            logger.warning(f"Prompt injection detected | pattern='{pattern.pattern[:40]}'")
            return InputGuardResult(
                is_safe=False,
                reason="Prompt injection attempt detected",
                threat_type="prompt_injection",
            )

    # Check 4: Harmful content
    for pattern in _harmful_compiled:
        if pattern.search(text):
            logger.warning(f"Harmful content detected | pattern='{pattern.pattern[:40]}'")
            return InputGuardResult(
                is_safe=False,
                reason="Harmful content detected in input",
                threat_type="harmful_content",
            )

    logger.debug("Input guard passed")
    return InputGuardResult(is_safe=True)