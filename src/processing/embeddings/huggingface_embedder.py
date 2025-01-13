from typing import List
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder

class HuggingFaceEmbedder(BaseEmbedder):
    """Hugging Face's Sentence Transformers embedding model wrapper."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize HuggingFace embedder.
        
        Args:
            model_name: Name of the model to use from HuggingFace hub
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.encode(texts).tolist()
