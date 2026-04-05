import numpy as np
import mlflow
import time
from sklearn.metrics.pairwise import cosine_similarity
from backend.database.db import SessionLocal, DocumentRecord
from backend.embeddings.minilm import MiniLMEmbedder
from backend.embeddings.mpnet import MPNetEmbedder
from backend.logger import get_logger

logger = get_logger("experiments.compare")

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
    scores = []
    for text_a, text_b in pairs:
        vec_a = embedder.embed([text_a])
        vec_b = embedder.embed([text_b])
        score = cosine_similarity(vec_a, vec_b)[0][0]
        scores.append(float(score))
        logger.debug(f"Pair similarity score={round(score, 4)}")
    return round(float(np.mean(scores)), 4)


def compute_speed(embedder, texts: list[str]) -> float:
    start = time.time()
    embedder.embed(texts)
    return round(time.time() - start, 3)


def load_texts_from_db(limit: int = 50) -> list[str]:
    db = SessionLocal()
    try:
        records = db.query(DocumentRecord).limit(limit).all()
        chunks = []
        for record in records:
            words = record.extracted_text.split()
            for i in range(0, min(len(words), 1000), 200):
                chunk = " ".join(words[i:i+200])
                if chunk.strip():
                    chunks.append(chunk)
        logger.info(f"Loaded {len(chunks)} chunks from {len(records)} documents")
        return chunks if chunks else ["Sample text for speed testing"] * 20
    finally:
        db.close()


def run_comparison() -> dict:
    logger.info("Starting embedding model comparison experiment")

    texts = load_texts_from_db()
    mlflow.set_experiment("embedding_model_comparison")

    results = {}

    embedders_to_test = [MiniLMEmbedder, MPNetEmbedder]

    try:
        from backend.embeddings.openai_embed import OpenAIEmbedder
        embedders_to_test.append(OpenAIEmbedder)
        logger.info("OpenAI API key found — including OpenAI in comparison")
    except EnvironmentError as e:
        logger.warning(f"Skipping OpenAI: {e}")

    for EmbedderClass in embedders_to_test:
        try:
            embedder = EmbedderClass()
            name = embedder.model_name()
            logger.info(f"Testing model: {name}")

            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_name", name)
                mlflow.log_param("num_test_texts", len(texts))

                similar_score = compute_similarity_score(embedder, SIMILAR_PAIRS)
                mlflow.log_metric("similar_pair_avg_cosine", similar_score)

                different_score = compute_similarity_score(embedder, DIFFERENT_PAIRS)
                mlflow.log_metric("different_pair_avg_cosine", different_score)

                separation = round(similar_score - different_score, 4)
                mlflow.log_metric("separation_score", separation)

                speed = compute_speed(embedder, texts)
                mlflow.log_metric("embedding_speed_seconds", speed)

                sample_vec = embedder.embed(["test"])
                dim = sample_vec.shape[1]
                mlflow.log_metric("vector_dimension", dim)

                logger.info(f"{name} | similar={similar_score} | different={different_score} | separation={separation} | speed={speed}s | dim={dim}")

                results[name] = {
                    "similar_score": similar_score,
                    "different_score": different_score,
                    "separation_score": separation,
                    "speed_seconds": speed,
                    "dimensions": dim,
                }

        except Exception as e:
            logger.error(f"Model {EmbedderClass.__name__} failed: {e}", exc_info=True)

    if results:
        winner = max(results, key=lambda m: results[m]["separation_score"])
        logger.info(f"WINNER: {winner} with separation={results[winner]['separation_score']}")
        results["winner"] = winner
    else:
        logger.error("No models completed successfully")
        results["winner"] = None

    return results