from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        pass

    @abstractmethod
    def model_name(self) -> str:
        pass