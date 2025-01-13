from typing import List, Dict, Any, NamedTuple
from .base import BaseExtractor

class WhatsAppMessage(NamedTuple):
    """Structure for WhatsApp messages."""
    text: str
    media: Dict[str, Any]
    timestamp: str
    sender: str

class WhatsAppExtractor(BaseExtractor):
    """Extract content from WhatsApp chat exports."""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from WhatsApp chat."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from WhatsApp chat: {str(e)}")
            
    def extract_media(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract media content from WhatsApp chat."""
        # This would normally parse the chat file and extract media references
        # For now, return empty list as we're just testing
        return []
