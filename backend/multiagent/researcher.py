import os
from groq import Groq
from dotenv import load_dotenv
from backend.rag.pipeline import query_pipeline
from backend.graphrag.pipeline import query_knowledge_graph
from backend.logger import get_logger

load_dotenv()
logger = get_logger("multiagent.researcher")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def researcher_node(state: dict) -> dict:
    """
    Researcher Agent Node.

    Responsibilities:
    - Search documents using RAG (Stage 3)
    - Search knowledge graph (Stage 9)
    - Compile raw findings without analysis
    - Return structured research report

    Receives: question, document_id, feedback_for_researcher
    Writes: research_findings, sources
    """
    question = state["question"]
    document_id = state.get("document_id")
    feedback = state.get("feedback_for_researcher", "")
    iteration = state.get("iterations", 1)

    logger.info(
        f"Researcher starting | iteration={iteration} | "
        f"question='{question[:50]}'"
    )

    # If there's feedback from the Critic, augment the search
    search_question = question
    if feedback:
        search_question = f"{question}\n\nAdditionally, specifically find: {feedback}"
        logger.info(f"Researcher received feedback: '{feedback[:80]}'")

    findings_parts = []

    # Search 1: Vector RAG
    logger.info("Researcher: running vector RAG search")
    try:
        rag_result = query_pipeline(
            question=search_question,
            document_id=document_id,
            top_k=5,
        )
        rag_answer = rag_result.get("answer", "")
        sources = rag_result.get("sources", [])

        if rag_answer:
            findings_parts.append(
                f"=== DOCUMENT SEARCH FINDINGS ===\n{rag_answer}"
            )
            logger.info(f"RAG findings | chars={len(rag_answer)}")
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        sources = []

    # Search 2: GraphRAG (if graph exists)
    logger.info("Researcher: running GraphRAG search")
    try:
        graph_result = query_knowledge_graph(
            question=search_question,
            document_id=document_id or 7,
            top_k=3,
        )
        graph_answer = graph_result.get("answer", "")
        if graph_result.get("graph_used") and graph_answer:
            findings_parts.append(
                f"=== KNOWLEDGE GRAPH FINDINGS ===\n{graph_answer}"
            )
            logger.info(f"GraphRAG findings | chars={len(graph_answer)}")
    except Exception as e:
        logger.warning(f"GraphRAG search failed: {e}")

    # Compile findings report
    if findings_parts:
        raw_findings = "\n\n".join(findings_parts)
    else:
        raw_findings = "No relevant information found in the documents."

    # Ask LLM to structure the findings clearly
    structure_prompt = f"""You are a Research Assistant. 
Organize these raw search findings into a clear, structured research report.

Original Question: {question}
{f'Additional focus areas: {feedback}' if feedback else ''}

Raw Findings:
{raw_findings}

Create a structured report with:
1. Key Facts Found
2. Relevant Details
3. Information Gaps (what you could NOT find)

Be factual, not analytical. Report what was found, don't interpret."""

    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a Research Assistant. Report facts clearly and objectively."
            },
            {"role": "user", "content": structure_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    research_findings = response.choices[0].message.content
    logger.info(
        f"Researcher complete | findings_chars={len(research_findings)}"
    )

    return {
        "research_findings": research_findings,
        "sources": sources,
        "messages": [{
            "agent": "Researcher",
            "iteration": iteration,
            "content": research_findings[:200] + "...",
        }],
    }