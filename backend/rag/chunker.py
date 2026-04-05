import re
import numpy as np
from backend.logger import get_logger

logger = get_logger("rag.chunker")


def fixed_chunker(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    Splits text into fixed-size word chunks with overlap.
    
    overlap means the last 20 words of chunk N become
    the first 20 words of chunk N+1. This prevents context
    being lost at chunk boundaries.
    
    Simple but can cut mid-sentence.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap  # move forward with overlap

    logger.info(f"Fixed chunking | input_words={len(words)} | chunks={len(chunks)} | size={chunk_size} | overlap={overlap}")
    return chunks


def recursive_chunker(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    """
    Tries to split on natural boundaries in this order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences (. ! ?)
    4. Words (fallback)
    
    Produces much more natural chunks that don't cut mid-sentence.
    """
    # Step 1: split on paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    # Step 2: if any paragraph is too long, split it into sentences
    sentences = []
    for para in paragraphs:
        if len(para.split()) <= chunk_size:
            sentences.append(para)
        else:
            # Split into sentences
            sent_list = re.split(r"(?<=[.!?])\s+", para)
            sentences.extend([s.strip() for s in sent_list if s.strip()])

    # Step 3: group sentences into chunks up to chunk_size words
    chunks = []
    current_chunk_words = []

    for sentence in sentences:
        sentence_words = sentence.split()

        if len(current_chunk_words) + len(sentence_words) <= chunk_size:
            current_chunk_words.extend(sentence_words)
        else:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            # Start new chunk with overlap from previous
            current_chunk_words = current_chunk_words[-overlap:] + sentence_words

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    logger.info(f"Recursive chunking | chunks={len(chunks)} | strategy=paragraph→sentence→word")
    return chunks


def semantic_chunker(text: str, threshold: float = 0.75) -> list[str]:
    """
    Groups sentences by semantic similarity.
    
    Algorithm:
    1. Split into individual sentences
    2. Embed each sentence
    3. Compute cosine similarity between consecutive sentences
    4. When similarity drops below threshold, start a new chunk
    
    Most intelligent strategy — keeps related ideas together.
    Slower because it needs to embed every sentence.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]

    if len(sentences) < 2:
        logger.warning("Too few sentences for semantic chunking — falling back to recursive")
        return recursive_chunker(text)

    logger.debug(f"Semantic chunking | sentences={len(sentences)}")

    # Embed all sentences at once (efficient batch)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Group sentences into chunks
    chunks = []
    current_group = [sentences[0]]

    for i in range(1, len(sentences)):
        # Compare current sentence to previous sentence
        similarity = cos_sim(
            embeddings[i - 1].reshape(1, -1),
            embeddings[i].reshape(1, -1)
        )[0][0]

        if similarity >= threshold:
            # Same topic — add to current chunk
            current_group.append(sentences[i])
        else:
            # Topic shift — save current chunk, start new one
            chunks.append(" ".join(current_group))
            current_group = [sentences[i]]

    if current_group:
        chunks.append(" ".join(current_group))

    logger.info(f"Semantic chunking | threshold={threshold} | chunks={len(chunks)}")
    return chunks


def chunk_text(text: str, strategy: str = "recursive") -> list[str]:
    """
    Main entry point. Pick strategy: 'fixed', 'recursive', 'semantic'
    """
    logger.info(f"Chunking with strategy='{strategy}'")

    if strategy == "fixed":
        return fixed_chunker(text)
    elif strategy == "recursive":
        return recursive_chunker(text)
    elif strategy == "semantic":
        return semantic_chunker(text)
    else:
        logger.warning(f"Unknown strategy '{strategy}' — defaulting to recursive")
        return recursive_chunker(text)