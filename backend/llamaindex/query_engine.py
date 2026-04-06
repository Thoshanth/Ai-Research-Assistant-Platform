from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from backend.logger import get_logger

logger = get_logger("llamaindex.query_engine")


def build_query_engine(index: VectorStoreIndex, top_k: int = 5, similarity_cutoff: float = 0.3):
    """
    Builds a query engine on top of the index.

    Pipeline inside this engine:
    1. Retriever — fetches top_k most similar nodes from ChromaDB
    2. SimilarityPostprocessor — filters out nodes below similarity_cutoff
       (removes irrelevant chunks before they reach the LLM)
    3. ResponseSynthesizer — takes remaining nodes, builds prompt,
       calls the LLM, and returns the answer

    This replaces our entire manual query_pipeline() from Stage 3
    with a clean, configurable object.
    """
    logger.info(f"Building query engine | top_k={top_k} | similarity_cutoff={similarity_cutoff}")

    # Step 1: Retriever — how many chunks to fetch
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    # Step 2: Postprocessor — filter low quality chunks
    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=similarity_cutoff
    )

    # Step 3: Assemble the query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[postprocessor],
    )

    logger.info("Query engine ready")
    return query_engine