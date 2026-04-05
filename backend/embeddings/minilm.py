import numpy as np
import time
from sentence_transformers import SentenceTransformer
from backend.embeddings.base import BaseEmbedder
from backend.logger import get_logger

logger = get_logger("embeddings.minilm")

class MiniLMEmbedder(BaseEmbedder):

    def __init__(self):
        logger.info("Loading MiniLM model...")
        start = time.time()
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"MiniLM loaded in {round(time.time() - start, 2)}s")

    def embed(self, texts: list[str]) -> np.ndarray:
        logger.debug(f"Embedding {len(texts)} texts with MiniLM")
        start = time.time()
        vectors = self._model.encode(texts, convert_to_numpy=True)
        elapsed = round(time.time() - start, 3)
        logger.info(f"MiniLM embedded {len(texts)} texts | dim={vectors.shape[1]} | time={elapsed}s")
        return vectors

    def model_name(self) -> str:
        return "all-MiniLM-L6-v2"