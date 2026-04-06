import os
from groq import Groq
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

# Groq client — free and fast
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model to use — llama-3.1-8b-instant is fast and free
# You can also use "llama-3.1-70b-versatile" for better quality
LLM_MODEL = "llama-3.1-8b-instant"


def index_document(document_id: int, strategy: str = "recursive") -> dict:
    """
    INDEXING PIPELINE

    Takes a document already stored in SQLite (from Stage 1),
    chunks it, embeds the chunks, and stores them in ChromaDB.
    Only needs to run once per document.
    """
    logger.info(f"Indexing document | id={document_id} | strategy={strategy}")

    # Step 1: Load document text from SQLite
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

    # Step 3: Embed all chunks
    logger.info(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.embed(chunks)
    embeddings_list = embeddings.tolist()

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
    and generates a cited answer using Groq (free LLM).
    """
    logger.info(f"Query received | question='{question[:60]}' | doc_filter={document_id}")

    # Step 1: Embed the question
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

    # Step 3: Rerank to get top_k chunks
    top_chunks = rerank(question, raw_results, top_k=top_k)

    # Step 4: Build context from top chunks
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        source = chunk["metadata"].get("filename", "Unknown")
        context_parts.append(f"[Source {i} — {source}]:\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    # Step 5: Build prompt
    system_prompt = """You are a precise research assistant.
Answer questions using ONLY the provided context chunks.
Always cite your sources using [Source N] notation.
If the context does not contain enough information, say so clearly.
Never make up information not present in the context."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer based only on the context above, with source citations:"""

    logger.info(f"Sending to Groq | model={LLM_MODEL} | chunks={len(top_chunks)} | context_chars={len(context)}")

    # Step 6: Call Groq API
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,  # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    logger.info(f"Groq responded | answer_chars={len(answer)} | input_tokens={input_tokens} | output_tokens={output_tokens}")

    # Step 7: Build sources list
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
        "model_used": LLM_MODEL,
        "question": question,
    }