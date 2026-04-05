import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from backend.rag.vector_store import search_vectors, get_all_chunks_for_bm25
from backend.logger import get_logger

logger = get_logger("rag.retriever")


def vector_search(query_embedding: list[float], n_results: int = 10, document_id: int = None) -> list[dict]:
    """
    Pure semantic search using ChromaDB.
    Returns list of dicts with text, metadata, and similarity score.
    """
    raw = search_vectors(query_embedding, n_results, document_id)

    results = []
    for text, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append({
            "text": text,
            "metadata": meta,
            "score": round(1 - dist, 4),  # convert distance to similarity
            "source": "vector",
        })

    logger.debug(f"Vector search: {len(results)} results")
    return results


def bm25_search(query: str, n_results: int = 10, document_id: int = None) -> list[dict]:
    """
    BM25 keyword search across all stored chunks.
    Great for exact term matching — finds 'EBITDA' when vector search might miss it.
    """
    all_chunks, all_metadatas = get_all_chunks_for_bm25(document_id)

    if not all_chunks:
        logger.warning("No chunks found for BM25 search")
        return []

    # Tokenize all chunks and the query
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    tokenized_query = query.lower().split()

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenized_query)

    # Get top N results
    top_indices = np.argsort(scores)[::-1][:n_results]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # only include results with actual matches
            results.append({
                "text": all_chunks[idx],
                "metadata": all_metadatas[idx],
                "score": round(float(scores[idx]), 4),
                "source": "bm25",
            })

    logger.debug(f"BM25 search: {len(results)} results")
    return results


def hybrid_search(
    query: str,
    query_embedding: list[float],
    n_results: int = 10,
    document_id: int = None,
) -> list[dict]:
    """
    Combines vector + BM25 results using Reciprocal Rank Fusion (RRF).
    
    RRF works by giving each result a score based on its rank
    in each search method, then combining those rank scores.
    This is better than averaging raw scores because BM25 and
    vector scores are on completely different scales.
    
    RRF formula: score = 1 / (rank + 60)
    The 60 is a constant that reduces the impact of very high ranks.
    """
    vector_results = vector_search(query_embedding, n_results, document_id)
    bm25_results = bm25_search(query, n_results, document_id)

    # Build a unified score map using chunk text as key
    rrf_scores = {}
    chunk_data = {}

    # Score vector results by rank
    for rank, result in enumerate(vector_results):
        key = result["text"][:100]  # use first 100 chars as key
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
        chunk_data[key] = result

    # Score BM25 results by rank and add to existing scores
    for rank, result in enumerate(bm25_results):
        key = result["text"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
        if key not in chunk_data:
            chunk_data[key] = result

    # Sort by combined RRF score
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    merged = []
    for key in sorted_keys[:n_results]:
        item = chunk_data[key].copy()
        item["rrf_score"] = round(rrf_scores[key], 6)
        item["source"] = "hybrid"
        merged.append(item)

    logger.info(f"Hybrid search: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(merged)} merged")
    return merged


def rerank(query: str, results: list[dict], top_k: int = 3) -> list[dict]:
    """
    Simple but effective reranker using keyword overlap scoring.
    
    In production you'd use a CrossEncoder model here (e.g. ms-marco-MiniLM).
    For now we score by how many query words appear in each chunk —
    this already significantly improves result quality over raw retrieval.
    
    A CrossEncoder reads query + chunk together and gives a relevance score,
    which is much more accurate but slower.
    """
    query_words = set(query.lower().split())

    for result in results:
        chunk_words = set(result["text"].lower().split())
        overlap = len(query_words & chunk_words)
        overlap_ratio = overlap / max(len(query_words), 1)

        # Combine original score with keyword overlap
        result["rerank_score"] = round(
            result.get("rrf_score", result.get("score", 0)) + overlap_ratio * 0.3,
            4
        )

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    logger.info(f"Reranking: {len(results)} → top {len(reranked)} results")
    return reranked