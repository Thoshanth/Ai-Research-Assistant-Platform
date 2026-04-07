from fastapi import HTTPException
from backend.guardrails.input_guard import check_input
from backend.guardrails.pii_detector import has_pii, detect_pii
from backend.guardrails.topic_guard import is_on_topic
from backend.guardrails.output_guard import check_output
from backend.logger import get_logger

logger = get_logger("guardrails.pipeline")


def run_input_guards(question: str) -> None:
    """
    Runs all input guards before the question reaches the LLM.
    Raises HTTPException if any guard fails — request is blocked.
    """
    logger.info(f"Running input guards | question='{question[:50]}'")

    # Guard 1: Basic input validation + injection detection
    input_result = check_input(question)
    if not input_result.is_safe:
        logger.warning(
            f"Input blocked | threat='{input_result.threat_type}' | reason='{input_result.reason}'"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input blocked by safety guardrail",
                "reason": input_result.reason,
                "threat_type": input_result.threat_type,
            }
        )

    # Guard 2: PII in input
    pii_found = detect_pii(question)
    if pii_found:
        logger.warning(f"PII in input | types={list(pii_found.keys())}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Input contains sensitive personal information",
                "pii_types": list(pii_found.keys()),
                "suggestion": "Please remove personal information from your question",
            }
        )

    # Guard 3: Topic relevance
    is_allowed, reason = is_on_topic(question)
    if not is_allowed:
        logger.warning(f"Topic blocked | reason='{reason}'")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Question is outside the scope of this assistant",
                "reason": reason,
                "suggestion": "This assistant is designed for document research and analysis",
            }
        )

    logger.info("All input guards passed")


def run_output_guards(
    answer: str,
    expect_citations: bool = False
) -> dict:
    """
    Runs all output guards after the LLM responds.
    Never blocks — always returns a safe version of the answer.
    Returns dict with safe answer + any warnings.
    """
    logger.info(f"Running output guards | chars={len(answer)}")

    result = check_output(answer, expect_citations=expect_citations)

    if result.was_modified:
        logger.info(f"Output was modified | warnings={result.warnings}")

    return {
        "answer": result.safe_text,
        "guardrail_warnings": result.warnings if result.warnings else [],
        "pii_redacted": result.pii_redacted,
        "hallucination_warning": result.hallucination_warning,
    }