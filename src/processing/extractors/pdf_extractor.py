from typing import Optional
import PyPDF2
import logging

logger = logging.getLogger(__name__)

class PDFExtractor:
    def extract(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
                    
            return "\n".join(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise
