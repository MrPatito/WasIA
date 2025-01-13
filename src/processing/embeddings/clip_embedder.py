from typing import List, Union, Dict, Any
import torch
from PIL import Image
import logging
import os
from transformers import CLIPProcessor, CLIPModel
from .base import BaseEmbedder

logger = logging.getLogger(__name__)

class CLIPEmbedder(BaseEmbedder):
    """
    Unified multimodal embedder using CLIP.
    Handles both text and images, producing embeddings in the same vector space.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CLIP embedder.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Text input must be a non-empty string")
                
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max token length
            ).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
            # Normalize and convert to list
            embedding = text_features.cpu().numpy()[0].tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise
            
    def embed_image(self, image: Union[str, Image.Image]) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: Path to image file or PIL Image object
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a file path or PIL Image object")
                
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Normalize and convert to list
            embedding = image_features.cpu().numpy()[0].tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            raise
            
    def embed_batch(
        self,
        items: List[Union[str, Image.Image, Dict[str, Any]]],
        modality: str = "text"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple items.
        
        Args:
            items: List of items to embed (texts, images, or mixed content)
            modality: Default modality for string inputs ('text' or 'image' paths)
        """
        try:
            if not items:
                raise ValueError("Batch cannot be empty")
                
            embeddings = []
            
            # Process each item
            for i, item in enumerate(items):
                if item is None:
                    raise ValueError(f"Item {i} in batch is None")
                    
                if isinstance(item, dict):
                    # Handle mixed content dictionary
                    if 'text' in item and 'image' in item:
                        # Mixed content - use multimodal embedding
                        embedding = self.embed_multimodal(item['text'], item['image'])
                        embeddings.append(embedding)
                    elif 'text' in item:
                        # Text only
                        embedding = self.embed_text(item['text'])
                        embeddings.append(embedding)
                    elif 'image' in item:
                        # Image only
                        embedding = self.embed_image(item['image'])
                        embeddings.append(embedding)
                    else:
                        raise ValueError(f"Item {i} in batch is a dictionary but has no 'text' or 'image' key")
                elif isinstance(item, Image.Image):
                    # Image object
                    embedding = self.embed_image(item)
                    embeddings.append(embedding)
                elif isinstance(item, str):
                    if modality == "text":
                        # Text string
                        embedding = self.embed_text(item)
                        embeddings.append(embedding)
                    else:
                        # Image path
                        embedding = self.embed_image(item)
                        embeddings.append(embedding)
                else:
                    raise ValueError(f"Item {i} in batch has unsupported type: {type(item)}")
                        
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
            
    def embed_multimodal(self, text: str, image: Union[str, Image.Image]) -> List[float]:
        """
        Generate a combined embedding for text and image pair.
        
        Args:
            text: Text to embed
            image: Image to embed (path or PIL Image)
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
                
            # Process inputs separately
            text_inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            image_inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get features separately
                text_features = self.model.get_text_features(**text_inputs)
                image_features = self.model.get_image_features(**image_inputs)
                
                # Average the features
                combined_features = (text_features + image_features) / 2
                
            # Normalize and convert to list
            embedding = combined_features.cpu().numpy()[0].tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating multimodal embedding: {str(e)}")
            raise
