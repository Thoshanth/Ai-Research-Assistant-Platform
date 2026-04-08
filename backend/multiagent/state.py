from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    """
    Shared state passed between all agents in the graph.

    TypedDict gives us type safety — every agent knows
    exactly what fields exist and what type they are.

    Annotated with operator.add means these fields are
    APPENDED to (not replaced) when agents write to them.
    This keeps a full history of each agent's contributions.
    """

    # Input
    question: str
    document_id: int | None

    # Researcher output
    research_findings: str
    sources: list[dict]

    # Analyst output
    analysis: str

    # Critic output
    critique: str
    approved: bool
    feedback_for_researcher: str

    # Control
    iterations: int
    max_iterations: int
    final_answer: str

    # Full message history for debugging
    messages: Annotated[list[dict], operator.add]