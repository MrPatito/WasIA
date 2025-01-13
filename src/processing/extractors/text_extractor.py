from typing import List, Dict, Any
from .base import BaseExtractor

class TextExtractor(BaseExtractor):
    """Extract text from plain text files."""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from file: {str(e)}")
            
    def extract_media(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract media content (none for text files)."""
        return []
