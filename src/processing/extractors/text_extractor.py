import logging

logger = logging.getLogger(__name__)

class TextExtractor:
    def extract(self, file_path: str) -> str:
        """
        Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            str: Extracted text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        except Exception as e:
            logger.error(f"Error extracting text from file {file_path}: {str(e)}")
            raise
