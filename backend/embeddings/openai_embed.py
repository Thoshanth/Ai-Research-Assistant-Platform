import numpy as np
import time
import os
from backend.embeddings.base import BaseEmbedder
from backend.logger import get_logger

logger = get_logger("embeddings.openai")

class OpenAIEmbedder(BaseEmbedder):

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set — skipping OpenAI embedder")
        self._client = OpenAI(api_key=api_key)
        logger.info("OpenAI embedder initialized")

    def embed(self, texts: list[str]) -> np.ndarray:
        logger.debug(f"Embedding {len(texts)} texts with OpenAI")
        start = time.time()
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        vectors = np.array([item.embedding for item in response.data])
        elapsed = round(time.time() - start, 3)
        logger.info(f"OpenAI embedded {len(texts)} texts | dim={vectors.shape[1]} | time={elapsed}s")
        return vectors

    def model_name(self) -> str:
        return "text-embedding-3-small"