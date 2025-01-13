from typing import List
import os
from langchain.embeddings import OpenAIEmbeddings
from .base import BaseEmbedder

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI's text embedding model wrapper."""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-ada-002"):
        """
        Initialize OpenAI embedder.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var
            model: Model name to use for embeddings
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
            
        self.model = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=model
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.embed_query(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.model.embed_documents(texts)
