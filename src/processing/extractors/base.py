from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from PIL import Image

class BaseExtractor(ABC):
    """Base class for all extractors."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract text content from file."""
        pass
        
    @abstractmethod
    def extract_media(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract media content (images, audio) from file."""
        pass
