from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
from PIL import Image

class BaseEmbedder(ABC):
    """Base class for all embedders."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
        
    @abstractmethod
    def embed_image(self, image: Union[str, Image.Image]) -> List[float]:
        """Generate embedding for a single image."""
        pass
        
    @abstractmethod
    def embed_batch(
        self,
        items: List[Union[str, Image.Image, Dict[str, Any]]],
        modality: str = "text"
    ) -> List[List[float]]:
        """Generate embeddings for multiple items."""
        pass
        
    @abstractmethod
    def embed_multimodal(
        self,
        text: str,
        image: Union[str, Image.Image]
    ) -> List[float]:
        """Generate a combined embedding for text and image pair."""
        pass
