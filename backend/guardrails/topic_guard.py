import os
from groq import Groq
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("guardrails.topic")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Fast keyword check — if any of these are in the question
# it's clearly relevant to a research assistant
ALLOWED_KEYWORDS = [
    "document", "resume", "report", "pdf", "file",
    "summarize", "explain", "what", "who", "when",
    "where", "how", "why", "which", "compare",
    "analyze", "research", "find", "search", "tell",
    "show", "list", "give", "describe", "define",
    "calculate", "compute", "skill", "experience",
    "project", "education", "work", "study",
]

# Clearly off-topic requests
BLOCKED_KEYWORDS = [
    "love poem", "romantic", "sex", "adult content",
    "write a song", "recipe for", "sports score",
    "stock price", "lottery",
]


def is_on_topic(question: str) -> tuple[bool, str]:
    """
    Two-step topic check:

    Step 1 — Fast keyword scan
    If clearly allowed → pass immediately
    If clearly blocked → reject immediately

    Step 2 — LLM judge for ambiguous cases
    Ask the LLM if this question is appropriate for
    a document research assistant.

    Returns (is_allowed, reason)
    """
    question_lower = question.lower()

    # Fast pass — clearly relevant
    for keyword in ALLOWED_KEYWORDS:
        if keyword in question_lower:
            logger.debug(f"Topic guard: fast pass | keyword='{keyword}'")
            return True, "allowed"

    # Fast block — clearly irrelevant
    for keyword in BLOCKED_KEYWORDS:
        if keyword in question_lower:
            logger.warning(f"Topic guard: blocked | keyword='{keyword}'")
            return False, f"Off-topic request detected: '{keyword}'"

    # LLM judge for ambiguous cases
    logger.debug("Topic guard: using LLM judge")
    prompt = f"""You are a content moderator for an AI research assistant.
This assistant helps users search and analyze uploaded documents like resumes, reports, and PDFs.

Is this question appropriate for a document research assistant?
Question: "{question}"

Answer with exactly one word: ALLOWED or BLOCKED"""

    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )

    decision = response.choices[0].message.content.strip().upper()
    is_allowed = "ALLOWED" in decision

    logger.info(f"Topic guard LLM decision | question='{question[:50]}' | decision={decision}")
    return is_allowed, "allowed" if is_allowed else "Question is outside the scope of this assistant"