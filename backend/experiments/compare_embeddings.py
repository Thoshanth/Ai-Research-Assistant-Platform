import numpy as np
import mlflow
import time
from sklearn.metrics.pairwise import cosine_similarity
from backend.database.db import SessionLocal, DocumentRecord
from backend.embeddings.minilm import MiniLMEmbedder
from backend.embeddings.mpnet import MPNetEmbedder
from backend.logger import get_logger

logger = get_logger("experiments.compare")

# ── Test sentence pairs ──────────────────────────────────────────
# We define pairs we KNOW should be similar and pairs we know
# should be different. This gives us ground truth to evaluate against.

SIMILAR_PAIRS = [
    ("The cat sat on the mat", "A cat was resting on a rug"),
    ("Machine learning is a subset of AI", "AI includes machine learning techniques"),
    ("The stock market crashed today", "Equity markets fell sharply"),
]

DIFFERENT_PAIRS = [
    ("The cat sat on the mat", "The stock market crashed today"),
    ("Machine learning is a subset of AI", "I enjoy eating pizza on Fridays"),
    ("The sun rises in the east", "Quantum computing uses qubits"),
]


def compute_similarity_score(embedder, pairs: list[tuple]) -> float:
    """
    Embeds both sentences in each pair and computes cosine similarity.
    Returns the average similarity across all pairs.
    """
    scores = []
    for text_a, text_b in pairs:
        vec_a = embedder.embed([text_a])
        vec_b = embedder.embed([text_b])
        score = cosine_similarity(vec_a, vec_b)[0][0]
        scores.append(float(score))
        logger.debug(f"Pair similarity: '{text_a[:30]}...' | score={round(score, 4)}")
    return round(float(np.mean(scores)), 4)


def compute_speed(embedder, texts: list[str]) -> float:
    """
    Measures how long it takes to embed all texts.
    Returns seconds taken.
    """
    start = time.time()
    embedder.embed(texts)
    return round(time.time() - start, 3)


def load_texts_from_db(limit: int = 50) -> list[str]:
    """
    Loads extracted text from documents we processed in Stage 1.
    Splits into chunks of ~200 words for realistic speed testing.
    """
    db = SessionLocal()
    try:
        records = db.query(DocumentRecord).limit(limit).all()
        chunks = []
        for record in records:
            words = record.extracted_text.split()
            # Split text into 200-word chunks
            for i in range(0, min(len(words), 1000), 200):
                chunk = " ".join(words[i:i+200])
                if chunk.strip():
                    chunks.append(chunk)
        logger.info(f"Loaded {len(chunks)} text chunks from {len(records)} documents")
        return chunks if chunks else ["Sample text for speed testing"] * 20
    finally:
        db.close()


def run_comparison() -> dict:
    """
    Main function: runs all embedding models, logs to MLflow,
    and returns the winner with evidence.
    """
    logger.info("=" * 60)
    logger.info("Starting embedding model comparison experiment")
    logger.info("=" * 60)

    # Load real text from Stage 1 database
    texts = load_texts_from_db()

    # Set up MLflow experiment
    # All runs under this experiment will appear together in the dashboard
    mlflow.set_experiment("embedding_model_comparison")

    results = {}

    # ── Define which models to test ──────────────────────────────
    embedders_to_test = [
        MiniLMEmbedder,
        MPNetEmbedder,
    ]

    # Try OpenAI too if API key is available
    try:
        from backend.embeddings.openai_embed import OpenAIEmbedder
        embedders_to_test.append(OpenAIEmbedder)
        logger.info("OpenAI API key found — including OpenAI embedder in comparison")
    except EnvironmentError as e:
        logger.warning(f"Skipping OpenAI: {e}")

    # ── Run each model ────────────────────────────────────────────
    for EmbedderClass in embedders_to_test:
        try:
            embedder = EmbedderClass()
            name = embedder.model_name()
            logger.info(f"Testing model: {name}")

            # Each model gets its own MLflow run
            with mlflow.start_run(run_name=name):

                # Log what model we're using
                mlflow.log_param("model_name", name)
                mlflow.log_param("num_test_texts", len(texts))

                # Metric 1: Similar pair score (higher = better)
                similar_score = compute_similarity_score(embedder, SIMILAR_PAIRS)
                mlflow.log_metric("similar_pair_avg_cosine", similar_score)
                logger.info(f"{name} | similar_score={similar_score}")

                # Metric 2: Different pair score (lower = better)
                different_score = compute_similarity_score(embedder, DIFFERENT_PAIRS)
                mlflow.log_metric("different_pair_avg_cosine", different_score)
                logger.info(f"{name} | different_score={different_score}")

                # Metric 3: Separation (similar - different, higher = better)
                # This is the key metric — how well does the model separate
                # similar from different texts?
                separation = round(similar_score - different_score, 4)
                mlflow.log_metric("separation_score", separation)
                logger.info(f"{name} | separation={separation}")

                # Metric 4: Embedding speed
                speed = compute_speed(embedder, texts)
                mlflow.log_metric("embedding_speed_seconds", speed)
                logger.info(f"{name} | speed={speed}s for {len(texts)} chunks")

                # Metric 5: Vector dimension
                sample_vec = embedder.embed(["test"])
                dim = sample_vec.shape[1]
                mlflow.log_metric("vector_dimension", dim)
                logger.info(f"{name} | dimensions={dim}")

                results[name] = {
                    "similar_score": similar_score,
                    "different_score": different_score,
                    "separation_score": separation,
                    "speed_seconds": speed,
                    "dimensions": dim,
                }

        except Exception as e:
            logger.error(f"Model {EmbedderClass.__name__} failed: {e}", exc_info=True)

    # ── Pick the winner ───────────────────────────────────────────
    if results:
        winner = max(results, key=lambda m: results[m]["separation_score"])
        logger.info("=" * 60)
        logger.info(f"WINNER: {winner} with separation={results[winner]['separation_score']}")
        logger.info("=" * 60)
        results["winner"] = winner
    else:
        logger.error("No models completed successfully")
        results["winner"] = None

    return results