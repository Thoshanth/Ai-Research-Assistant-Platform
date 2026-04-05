import os
from anthropic import Anthropic
from dotenv import load_dotenv
from backend.rag.chunker import chunk_text
from backend.rag.vector_store import store_chunks
from backend.rag.retriever import hybrid_search, rerank
from backend.embeddings.minilm import MiniLMEmbedder
from backend.database.db import SessionLocal, DocumentRecord
from backend.logger import get_logger

load_dotenv()
logger = get_logger("rag.pipeline")

# Load embedder once — reused for all indexing and querying
embedder = MiniLMEmbedder()

# Anthropic client for answer generation
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def index_document(document_id: int, strategy: str = "recursive") -> dict:
    """
    INDEXING PIPELINE
    
    Takes a document already stored in SQLite (from Stage 1),
    chunks it, embeds the chunks, and stores them in ChromaDB.
    
    This only needs to run once per document.
    After indexing, the document is searchable forever.
    """
    logger.info(f"Indexing document | id={document_id} | strategy={strategy}")

    # Step 1: Load document text from SQLite (Stage 1 stored this)
    db = SessionLocal()
    try:
        record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        if not record:
            raise ValueError(f"Document {document_id} not found in database")
        text = record.extracted_text
        filename = record.filename
    finally:
        db.close()

    # Step 2: Chunk the text
    chunks = chunk_text(text, strategy=strategy)
    logger.info(f"Created {len(chunks)} chunks from '{filename}'")

    if not chunks:
        raise ValueError("No chunks created — document may be empty")

    # Step 3: Embed all chunks (batch for efficiency)
    logger.info(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.embed(chunks)
    embeddings_list = embeddings.tolist()  # ChromaDB needs plain lists

    # Step 4: Store in ChromaDB
    store_chunks(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings_list,
        strategy=strategy,
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_created": len(chunks),
        "strategy": strategy,
        "status": "indexed",
    }


def query_pipeline(question: str, document_id: int = None, top_k: int = 3) -> dict:
    """
    QUERYING PIPELINE
    
    Takes a user question, retrieves relevant chunks,
    and generates a cited answer using Claude.
    
    Steps:
    1. Embed the question
    2. Hybrid search (vector + BM25)
    3. Rerank results
    4. Build prompt with context
    5. Call Claude API
    6. Return answer + sources
    """
    logger.info(f"Query received | question='{question[:60]}...' | doc_filter={document_id}")

    # Step 1: Embed the question using the SAME model used for indexing
    # This is critical — query and chunks must be in the same vector space
    query_embedding = embedder.embed([question])[0].tolist()

    # Step 2: Hybrid search
    raw_results = hybrid_search(
        query=question,
        query_embedding=query_embedding,
        n_results=10,
        document_id=document_id,
    )

    if not raw_results:
        logger.warning("No results found — document may not be indexed yet")
        return {
            "answer": "No relevant documents found. Please index a document first using POST /index/{document_id}",
            "sources": [],
        }

    # Step 3: Rerank to get top_k most relevant chunks
    top_chunks = rerank(question, raw_results, top_k=top_k)

    # Step 4: Build the context string from top chunks
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        source = chunk["metadata"].get("filename", "Unknown")
        context_parts.append(f"[Source {i} — {source}]:\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    # Step 5: Build the prompt
    # This is the core prompt engineering of RAG —
    # we force the LLM to only use what we give it
    system_prompt = """You are a precise research assistant. 
Answer questions using ONLY the provided context chunks.
Always cite your sources using [Source N] notation.
If the context doesn't contain enough information, say so clearly.
Never make up information not present in the context."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer based only on the context above, with source citations:"""

    logger.info(f"Sending to Claude | context_chunks={len(top_chunks)} | context_chars={len(context)}")

    # Step 6: Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = response.content[0].text
    logger.info(f"Claude responded | answer_chars={len(answer)} | input_tokens={response.usage.input_tokens} | output_tokens={response.usage.output_tokens}")

    # Step 7: Build sources list for the API response
    sources = [
        {
            "chunk_index": chunk["metadata"].get("chunk_index"),
            "filename": chunk["metadata"].get("filename"),
            "document_id": chunk["metadata"].get("document_id"),
            "relevance_score": chunk.get("rerank_score"),
            "text_preview": chunk["text"][:150] + "...",
        }
        for chunk in top_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(top_chunks),
        "question": question,
    }