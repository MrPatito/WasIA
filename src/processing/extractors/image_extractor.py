from typing import List, Dict, Any
from PIL import Image
import pytesseract
from .base import BaseExtractor

class ImageExtractor(BaseExtractor):
    """Extract text and metadata from images."""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from image: {str(e)}")
            
    def extract_media(self, file_path: str) -> List[Dict[str, Any]]:
        """Return the image itself as media content."""
        try:
            image = Image.open(file_path)
            return [{
                'type': 'image',
                'format': image.format.lower(),
                'data': image
            }]
        except Exception as e:
            raise ValueError(f"Error extracting image: {str(e)}")
            
    def __del__(self):
        """Clean up resources."""
        pass
