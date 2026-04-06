import os
from groq import Groq
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("langchain.router")

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Keywords that strongly suggest document search is needed
DOCUMENT_KEYWORDS = [
    "resume", "document", "pdf", "file", "uploaded",
    "report", "cv", "summary", "content", "says",
    "mention", "according", "skill", "experience",
    "college", "university", "project", "objective",
    "certificate", "qualification", "work", "job"
]


def should_search_documents(question: str) -> bool:
    """
    Two-step routing:

    Step 1 — Fast keyword check
    If the question contains obvious document-related words,
    immediately return True without calling the LLM.
    This saves API calls and is faster.

    Step 2 — LLM routing for ambiguous questions
    Only call the LLM router when keywords don't match.
    """
    question_lower = question.lower()

    # Step 1: Fast keyword check
    for keyword in DOCUMENT_KEYWORDS:
        if keyword in question_lower:
            logger.info(f"Router: keyword match '{keyword}' → DOCUMENTS")
            return True

    # Step 2: LLM decides for ambiguous questions
    router_prompt = f"""You are a query router for a document search system.
A user has uploaded personal documents like resumes, reports, and PDFs.

Decide if this question needs to search those uploaded documents or can be answered from general knowledge.

Question: "{question}"

Rules:
- Questions about a specific person, their details, skills, education, work → DOCUMENTS
- Questions about general concepts, definitions, technology, science → GENERAL
- When in doubt → DOCUMENTS

Respond with exactly one word only: DOCUMENTS or GENERAL"""

    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": router_prompt}],
        max_tokens=5,
        temperature=0,
    )

    decision = response.choices[0].message.content.strip().upper()
    use_docs = "DOCUMENTS" in decision

    logger.info(f"Router: LLM decision='{decision}' | use_docs={use_docs} | question='{question[:50]}'")
    return use_docs