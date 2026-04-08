from langgraph.graph import StateGraph, END
from backend.multiagent.state import ResearchState
from backend.multiagent.researcher import researcher_node
from backend.multiagent.analyst import analyst_node
from backend.multiagent.critic import critic_node
from backend.logger import get_logger

logger = get_logger("multiagent.graph")


def should_continue(state: dict) -> str:
    """
    Conditional edge function — decides where to route after Critic.

    This is the key routing logic of the entire multi-agent system.
    LangGraph calls this function after every Critic run and uses
    the return value to decide which node to go to next.

    Returns:
    - "researcher" → Critic rejected, send back for more research
    - "end"        → Critic approved OR max iterations reached
    """
    approved = state.get("approved", False)
    iterations = state.get("iterations", 1)
    max_iterations = state.get("max_iterations", 3)

    if approved:
        logger.info(f"Routing: APPROVED → END | iteration={iterations}")
        return "end"

    if iterations >= max_iterations:
        logger.warning(
            f"Routing: MAX ITERATIONS REACHED → END | iteration={iterations}"
        )
        return "end"

    logger.info(
        f"Routing: REJECTED → Researcher | iteration={iterations}"
    )
    return "researcher"


def compile_final_answer(state: dict) -> dict:
    """
    Final node — compiles the approved analysis into
    a clean, user-facing answer.
    """
    import os
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()

    _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Compiling final answer")

    question = state["question"]
    analysis = state["analysis"]
    critique = state.get("critique", "")
    approved = state.get("approved", False)

    compile_prompt = f"""Based on this research analysis, write a clear, 
comprehensive final answer to the user's question.

Question: {question}

Analysis:
{analysis}

Critic's evaluation:
{critique}

Write a well-structured final answer that:
- Directly addresses the question
- Is clear and readable
- Includes the most important insights
- Notes any limitations or uncertainties"""

    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a Report Writer. Compile research into clear final answers."
            },
            {"role": "user", "content": compile_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    final_answer = response.choices[0].message.content
    logger.info(f"Final answer compiled | chars={len(final_answer)}")

    return {
        "final_answer": final_answer,
        "messages": [{
            "agent": "Compiler",
            "content": "Final answer compiled",
            "approved": approved,
        }],
    }


def build_research_graph():
    """
    Builds and compiles the LangGraph workflow.

    Graph structure:
    START → researcher → analyst → critic → [conditional]
                ↑                              |
                └──────── REJECTED ────────────┘
                                               |
                                           APPROVED
                                               |
                                           compiler → END
    """
    logger.info("Building multi-agent research graph")

    # Create the graph with our state schema
    workflow = StateGraph(ResearchState)

    # Add agent nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("compiler", compile_final_answer)

    # Define the flow
    # START → Researcher (always start with research)
    workflow.set_entry_point("researcher")

    # Researcher → Analyst (always)
    workflow.add_edge("researcher", "analyst")

    # Analyst → Critic (always)
    workflow.add_edge("analyst", "critic")

    # Critic → [conditional routing]
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "researcher": "researcher",  # rejected → back to researcher
            "end": "compiler",           # approved → compile answer
        }
    )

    # Compiler → END
    workflow.add_edge("compiler", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("Multi-agent graph compiled successfully")
    return app


# Build once at module load time — reused for all requests
research_graph = build_research_graph()