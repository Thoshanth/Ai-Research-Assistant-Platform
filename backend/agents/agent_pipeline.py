from backend.agents.agent_loop import run_agent_loop
from backend.langchain.memory import get_history_as_text, save_to_memory
from backend.logger import get_logger

logger = get_logger("agents.pipeline")


def run_agent(
    question: str,
    session_id: str = "default",
    document_id: int = None,
    show_trace: bool = False,
) -> dict:
    """
    Entry point for the agent.

    Connects:
    - Stage 4 conversation memory (for context)
    - Stage 5 agent loop (for reasoning)
    - Saves result back to memory

    show_trace=True returns the full step-by-step reasoning.
    show_trace=False returns just the final answer (cleaner for users).
    """
    logger.info(f"Agent run | session='{session_id}' | question='{question[:60]}'")

    # Get conversation history from Stage 4 memory
    history = get_history_as_text(session_id)

    # Run the ReAct agent loop
    result = run_agent_loop(
        question=question,
        document_id=document_id,
        conversation_history=history,
    )

    # Save to memory so future turns have context
    save_to_memory(session_id, question, result["answer"])

    response = {
        "answer": result["answer"],
        "iterations_used": result["iterations"],
        "session_id": session_id,
        "question": question,
    }

    # Optionally include the full reasoning trace
    if show_trace:
        response["trace"] = result["trace"]

    return response