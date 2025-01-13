import fitz  # PyMuPDF
from typing import List, Dict, Any
from PIL import Image
import io
from .base import BaseExtractor

class PDFExtractor(BaseExtractor):
    """Extract text and images from PDF files."""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text content from PDF."""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
            
    def extract_media(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF."""
        try:
            doc = fitz.open(file_path)
            images = []
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images()
                
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    images.append({
                        'type': 'image',
                        'format': base_image["ext"],
                        'page': page_num + 1,
                        'index': img_idx + 1,
                        'data': image
                    })
                    
            return images
            
        except Exception as e:
            raise ValueError(f"Error extracting images from PDF: {str(e)}")
            
    def __del__(self):
        """Clean up resources."""
        pass
