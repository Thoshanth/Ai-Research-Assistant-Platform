import os
from groq import Groq
from dotenv import load_dotenv
from backend.graphrag.graph_store import load_graph, search_graph
from backend.logger import get_logger

load_dotenv()
logger = get_logger("graphrag.retriever")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_query_entities(question: str) -> list[str]:
    """
    Extracts key entities from the user's question.
    These are used as starting points for graph traversal.

    Example:
    "What technologies does Thoshanth know?" 
    → ["Thoshanth", "technologies"]
    """
    prompt = f"""Extract the key entities (people, technologies, organizations, skills, concepts) from this question.

Question: "{question}"

Return ONLY a JSON array of entity strings.
Example: ["Python", "Thoshanth", "machine learning"]

If no specific entities found return: []"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        import json
        entities = json.loads(raw.strip())
        logger.info(f"Query entities extracted | entities={entities}")
        return entities if isinstance(entities, list) else []
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
        return []


def graphrag_retrieve(
    question: str,
    document_id: int,
    top_k: int = 3,
) -> dict:
    """
    Two-path retrieval combining graph + vector search.

    Path 1 — Graph traversal:
    Extract entities from question → traverse knowledge graph
    → collect related nodes and edges as structured context

    Path 2 — Vector search:
    Use Stage 3 RAG pipeline for semantic text chunk retrieval

    Combine both for a richer, more accurate context
    before sending to the LLM.
    """
    logger.info(
        f"GraphRAG retrieve | question='{question[:50]}' | doc_id={document_id}"
    )

    # Path 1: Graph traversal
    graph_context = ""
    try:
        G = load_graph(document_id)
        query_entities = extract_query_entities(question)

        if query_entities:
            graph_result = search_graph(G, query_entities, max_hops=2)

            if graph_result["edges"] or graph_result["nodes"]:
                graph_context = "=== KNOWLEDGE GRAPH CONTEXT ===\n"
                graph_context += "Entities found:\n"
                graph_context += "\n".join(
                    f"  • {n}" for n in graph_result["nodes"]
                )
                graph_context += "\n\nRelationships:\n"
                graph_context += "\n".join(
                    f"  • {e}" for e in graph_result["edges"]
                )
                logger.info(
                    f"Graph context built | "
                    f"nodes={len(graph_result['nodes'])} | "
                    f"edges={len(graph_result['edges'])}"
                )
    except ValueError as e:
        logger.warning(f"Graph not available: {e}")

    # Path 2: Vector search (Stage 3 RAG)
    from backend.rag.pipeline import query_pipeline
    vector_result = query_pipeline(
        question=question,
        document_id=document_id,
        top_k=top_k,
    )
    vector_context = vector_result.get("answer", "")
    sources = vector_result.get("sources", [])

    return {
        "graph_context": graph_context,
        "vector_context": vector_context,
        "sources": sources,
        "query_entities": query_entities if "query_entities" in dir() else [],
    }