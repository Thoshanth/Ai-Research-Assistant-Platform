import chromadb
from chromadb.config import Settings
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("rag.vector_store")

# ChromaDB persists to disk in this folder
CHROMA_PATH = Path("chroma_db")
CHROMA_PATH.mkdir(exist_ok=True)

# Single shared client — created once, reused everywhere
_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(name: str = "documents"):
    """
    Gets or creates a ChromaDB collection.
    A collection is like a table — it holds vectors + metadata.
    get_or_create means safe to call multiple times.
    """
    collection = _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )
    logger.debug(f"Collection '{name}' ready | count={collection.count()}")
    return collection


def store_chunks(
    document_id: int,
    filename: str,
    chunks: list[str],
    embeddings: list[list[float]],
    strategy: str,
):
    """
    Stores chunks and their embeddings in ChromaDB.
    
    Each chunk gets:
    - A unique ID (doc_id + chunk_index)
    - Its embedding vector
    - The raw text (ChromaDB stores this as 'document')
    - Metadata (source file, page, strategy used)
    """
    collection = get_collection()

    ids = [f"doc{document_id}_chunk{i}_{strategy}" for i in range(len(chunks))]

    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": i,
            "strategy": strategy,
            "chunk_length": len(chunk.split()),
        }
        for i, chunk in enumerate(chunks)
    ]

    # Upsert = insert or update if ID already exists
    # Safe to re-index the same document multiple times
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(f"Stored {len(chunks)} chunks | doc_id={document_id} | file='{filename}' | strategy={strategy}")


def search_vectors(query_embedding: list[float], n_results: int = 10, document_id: int = None) -> dict:
    """
    Searches ChromaDB for the most similar chunks to the query embedding.
    
    Optional document_id filter — search only within one document.
    Returns chunks with their similarity distances and metadata.
    """
    collection = get_collection()

    where_filter = {"document_id": document_id} if document_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    logger.debug(f"Vector search returned {len(results['documents'][0])} results")
    return results


def get_all_chunks_for_bm25(document_id: int = None) -> tuple[list[str], list[dict]]:
    """
    Retrieves all stored chunks for BM25 keyword search.
    BM25 needs to see all documents to compute term frequencies.
    """
    collection = get_collection()

    where_filter = {"document_id": document_id} if document_id else None

    results = collection.get(
        where=where_filter,
        include=["documents", "metadatas"],
    )

    chunks = results["documents"]
    metadatas = results["metadatas"]
    logger.debug(f"Loaded {len(chunks)} chunks for BM25")
    return chunks, metadatas