import os
from groq import Groq
from dotenv import load_dotenv
from backend.langchain.memory import save_to_memory, get_history_as_text
from backend.langchain.router import should_search_documents
from backend.logger import get_logger

load_dotenv()
logger = get_logger("langchain.chat_pipeline")

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def chat(
    question: str,
    session_id: str = "default",
    document_id: int = None,
) -> dict:
    """
    Main conversational entry point.

    Flow:
    1. Load conversation history for this session
    2. Router decides: search docs or answer directly
    3a. If docs → use LlamaIndex query engine
    3b. If general → answer directly with LLM + history
    4. Save Q&A to memory
    5. Return answer
    """
    logger.info(f"Chat request | session='{session_id}' | question='{question[:60]}'")

    # Step 1: Get conversation history
    history = get_history_as_text(session_id)

    # Step 2: Router decides
    use_documents = should_search_documents(question)

    if use_documents:
        answer = _answer_from_documents(question, history, document_id)
        source = "documents"
    else:
        answer = _answer_from_llm(question, history)
        source = "general_knowledge"

    # Step 3: Save to memory
    save_to_memory(session_id, question, answer)

    logger.info(f"Chat complete | session='{session_id}' | source={source}")

    return {
        "answer": answer,
        "source": source,
        "session_id": session_id,
        "question": question,
    }


def _answer_from_documents(question: str, history: str, document_id: int = None) -> str:
    """Uses LlamaIndex query engine to answer from uploaded documents."""
    try:
        from backend.llamaindex.indexer import load_existing_index
        from backend.llamaindex.query_engine import build_query_engine

        index = load_existing_index()
        query_engine = build_query_engine(index, top_k=5)

        # Inject conversation history into the question
        if history:
            augmented_question = f"""Previous conversation:
{history}

Current question: {question}"""
        else:
            augmented_question = question

        response = query_engine.query(augmented_question)
        answer = str(response)

        logger.info(f"Document answer generated | chars={len(answer)}")
        return answer

    except ValueError as e:
        logger.warning(f"Index not ready: {e}")
        return "No documents indexed yet. Please upload and index a document first."


def _answer_from_llm(question: str, history: str) -> str:
    """Answers general knowledge questions directly using Groq."""

    system_prompt = "You are a helpful assistant. Answer clearly and concisely."

    if history:
        user_prompt = f"""Previous conversation:
{history}

Current question: {question}"""
    else:
        user_prompt = question

    response = _groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    logger.info(f"General answer generated | chars={len(answer)}")
    return answer