from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):
    """
    Every embedding model must implement these two methods.
    This ensures all models are swappable without changing
    any other part of the codebase.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Takes a list of strings.
        Returns a 2D numpy array of shape (num_texts, embedding_dim).
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the human-readable name of this model.
        Used for logging and MLflow tracking.
        """
        pass