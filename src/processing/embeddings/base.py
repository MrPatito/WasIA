from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    """Base class for all embedding models."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text into an embedding vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts into embedding vectors.
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        pass
