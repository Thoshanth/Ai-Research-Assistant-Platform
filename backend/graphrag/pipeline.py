import os
from groq import Groq
from dotenv import load_dotenv
from backend.graphrag.extractor import extract_from_chunks
from backend.graphrag.graph_store import (
    build_graph, save_graph, load_graph, get_graph_summary
)
from backend.graphrag.graph_retriever import graphrag_retrieve
from backend.rag.chunker import chunk_text
from backend.database.db import SessionLocal, DocumentRecord
from backend.logger import get_logger

load_dotenv()
logger = get_logger("graphrag.pipeline")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_knowledge_graph(document_id: int) -> dict:
    """
    Full graph building pipeline:
    1. Load document text from SQLite (Stage 1)
    2. Chunk the text (Stage 3 chunker)
    3. Extract entities and relations from each chunk (LLM)
    4. Build NetworkX graph
    5. Save to disk
    """
    logger.info(f"Building knowledge graph | doc_id={document_id}")

    # Step 1: Load document from SQLite
    db = SessionLocal()
    try:
        record = db.query(DocumentRecord).filter(
            DocumentRecord.id == document_id
        ).first()
        if not record:
            raise ValueError(f"Document {document_id} not found")
        text = record.extracted_text
        filename = record.filename
    finally:
        db.close()

    logger.info(f"Loaded document | file='{filename}' | chars={len(text)}")

    # Step 2: Chunk the text
    chunks = chunk_text(text, strategy="recursive")
    # Limit chunks for graph building — more chunks = more API calls
    chunks = chunks[:20]
    logger.info(f"Processing {len(chunks)} chunks for entity extraction")

    # Step 3: Extract entities and relations
    extracted = extract_from_chunks(chunks)

    # Step 4: Build graph
    G = build_graph(extracted, document_id)

    # Step 5: Save graph
    save_graph(G, document_id)

    summary = get_graph_summary(G)

    logger.info(
        f"Knowledge graph complete | "
        f"nodes={summary['total_nodes']} | "
        f"edges={summary['total_edges']}"
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "graph_summary": summary,
        "status": "graph_built",
        "next_step": f"Query with POST /graphrag/query?document_id={document_id}",
    }


def query_knowledge_graph(
    question: str,
    document_id: int,
    top_k: int = 3,
) -> dict:
    """
    GraphRAG query pipeline:
    1. Two-path retrieval (graph + vector)
    2. Combine contexts
    3. LLM synthesizes final answer
    """
    logger.info(
        f"GraphRAG query | question='{question[:60]}' | doc_id={document_id}"
    )

    # Step 1: Two-path retrieval
    retrieval = graphrag_retrieve(question, document_id, top_k)

    graph_context = retrieval["graph_context"]
    vector_context = retrieval["vector_context"]
    sources = retrieval["sources"]

    # Step 2: Build combined prompt
    system_prompt = """You are a precise research assistant with access to both 
a knowledge graph and document text.

Use ALL provided context to answer the question:
- Knowledge graph shows entity relationships and connections
- Document context provides supporting text with details

Synthesize both sources for the most complete answer.
Always be specific and cite what type of context supports your answer."""

    combined_context = ""
    if graph_context:
        combined_context += graph_context + "\n\n"

    combined_context += "=== DOCUMENT TEXT CONTEXT ===\n"
    combined_context += vector_context

    user_prompt = f"""Context:
{combined_context}

Question: {question}

Answer using both the knowledge graph relationships and document text:"""

    logger.info(
        f"Calling LLM | "
        f"has_graph_context={bool(graph_context)} | "
        f"context_chars={len(combined_context)}"
    )

    # Step 3: Generate answer
    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    answer = response.choices[0].message.content
    logger.info(f"GraphRAG answer generated | chars={len(answer)}")

    return {
        "answer": answer,
        "sources": sources,
        "graph_used": bool(graph_context),
        "query_entities": retrieval.get("query_entities", []),
        "question": question,
        "retrieval_type": "graphrag",
    }


def explore_graph(document_id: int) -> dict:
    """Returns the full graph summary for exploration."""
    G = load_graph(document_id)
    summary = get_graph_summary(G)

    # Also return all nodes grouped by type
    nodes_by_type = {}
    for node, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        if t not in nodes_by_type:
            nodes_by_type[t] = []
        nodes_by_type[t].append(node)

    # Return all edges
    all_edges = []
    for source, target, data in G.edges(data=True):
        all_edges.append({
            "source": source,
            "relation": data.get("relation", "RELATED_TO"),
            "target": target,
        })

    return {
        "document_id": document_id,
        "summary": summary,
        "nodes_by_type": nodes_by_type,
        "all_relationships": all_edges,
    }