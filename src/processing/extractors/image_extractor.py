import pytesseract
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageExtractor:
    def extract(self, file_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            str: Extracted text content
        """
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image {file_path}: {str(e)}")
            raise
