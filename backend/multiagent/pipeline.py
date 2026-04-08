from backend.multiagent.graph import research_graph
from backend.langchain.memory import get_history_as_text, save_to_memory
from backend.logger import get_logger

logger = get_logger("multiagent.pipeline")


def run_multiagent(
    question: str,
    session_id: str = "default",
    document_id: int = None,
    max_iterations: int = 3,
    show_agent_trace: bool = False,
) -> dict:
    """
    Runs the full multi-agent research pipeline.

    Initializes state, runs the LangGraph workflow,
    and returns the final compiled answer with metadata.

    The graph handles all routing internally —
    we just give it an initial state and it runs
    until it reaches END.
    """
    logger.info(
        f"Multi-agent run | question='{question[:60]}' | "
        f"session='{session_id}' | max_iter={max_iterations}"
    )

    # Get conversation history from Stage 4 memory
    history = get_history_as_text(session_id)
    augmented_question = question
    if history:
        augmented_question = (
            f"Context from previous conversation:\n{history}\n\n"
            f"Current question: {question}"
        )

    # Initialize state
    initial_state = {
        "question": augmented_question,
        "document_id": document_id,
        "research_findings": "",
        "sources": [],
        "analysis": "",
        "critique": "",
        "approved": False,
        "feedback_for_researcher": "",
        "iterations": 1,
        "max_iterations": max_iterations,
        "final_answer": "",
        "messages": [],
    }

    # Run the graph
    logger.info("Invoking LangGraph research workflow")
    final_state = research_graph.invoke(initial_state)

    final_answer = final_state.get("final_answer", "")
    messages = final_state.get("messages", [])
    iterations_used = final_state.get("iterations", 1) - 1
    approved = final_state.get("approved", False)

    logger.info(
        f"Multi-agent complete | "
        f"iterations={iterations_used} | "
        f"approved={approved} | "
        f"answer_chars={len(final_answer)}"
    )

    # Save to Stage 4 conversation memory
    save_to_memory(session_id, question, final_answer)

    response = {
        "answer": final_answer,
        "session_id": session_id,
        "question": question,
        "iterations_used": iterations_used,
        "approved_by_critic": approved,
        "agents_used": ["Researcher", "Analyst", "Critic", "Compiler"],
    }

    # Optionally include full agent trace
    if show_agent_trace:
        response["agent_trace"] = messages
        response["research_findings"] = final_state.get(
            "research_findings", ""
        )
        response["analysis"] = final_state.get("analysis", "")
        response["critique"] = final_state.get("critique", "")

    return response