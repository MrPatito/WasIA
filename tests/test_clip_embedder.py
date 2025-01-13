import unittest
import os
import torch
from PIL import Image
import numpy as np
from src.processing.embeddings.clip_embedder import CLIPEmbedder

class TestCLIPEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used for all tests."""
        cls.embedder = CLIPEmbedder()
        
        # Test data paths
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cls.test_image_path = os.path.join(cls.test_data_dir, 'WhatsApp Image 2021-12-03 at 2.07.05 PM(2).jpeg')
        
        # Load test files
        cls.test_image = Image.open(cls.test_image_path)
        cls.test_text = "A test text for embedding generation"
        
    def test_text_embedding(self):
        """Test text embedding generation."""
        embedding = self.embedder.embed_text(self.test_text)
        
        # Verify embedding properties
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 512)  # CLIP base model dimension
        self.assertTrue(all(isinstance(x, float) for x in embedding))
        
        # Test consistency
        embedding2 = self.embedder.embed_text(self.test_text)
        self.assertTrue(np.allclose(embedding, embedding2, atol=1e-5))
        
    def test_image_embedding(self):
        """Test image embedding generation."""
        # Test with image path
        embedding_path = self.embedder.embed_image(self.test_image_path)
        
        # Test with PIL Image
        embedding_pil = self.embedder.embed_image(self.test_image)
        
        # Verify embedding properties
        self.assertIsInstance(embedding_path, list)
        self.assertEqual(len(embedding_path), 512)
        self.assertTrue(all(isinstance(x, float) for x in embedding_path))
        
        # Verify consistency between path and PIL inputs
        self.assertTrue(np.allclose(embedding_path, embedding_pil, atol=1e-5))
        
    def test_batch_processing(self):
        """Test batch processing of multiple items."""
        # Create test batch
        batch_items = [
            self.test_text,  # Text only
            self.test_image,  # Image only
            "Another test text",  # Text only
            {"text": "Mixed content", "image": self.test_image_path}  # Mixed content
        ]
        
        # Get batch embeddings
        embeddings = self.embedder.embed_batch(batch_items)
        
        # Verify batch results
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 4)  # Should match number of inputs
        for emb in embeddings:
            self.assertEqual(len(emb), 512)  # CLIP base model dimension
            self.assertTrue(all(isinstance(x, float) for x in emb))
            
    def test_multimodal_embedding(self):
        """Test combined text-image embedding."""
        embedding = self.embedder.embed_multimodal(
            self.test_text,
            self.test_image
        )
        
        # Verify multimodal embedding
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 512)
        self.assertTrue(all(isinstance(x, float) for x in embedding))
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid text
        with self.assertRaises(ValueError):
            self.embedder.embed_text("")
            
        with self.assertRaises(ValueError):
            self.embedder.embed_text(None)
            
        # Test invalid image path
        with self.assertRaises(FileNotFoundError):
            self.embedder.embed_image("nonexistent_image.jpg")
            
        # Test invalid image object
        with self.assertRaises(ValueError):
            self.embedder.embed_image(None)
            
        # Test invalid batch items
        with self.assertRaises(ValueError):
            self.embedder.embed_batch([])  # Empty batch
            
        with self.assertRaises(ValueError):
            self.embedder.embed_batch([None])  # Invalid item
            
    def test_embedding_similarity(self):
        """Test that similar inputs produce similar embeddings."""
        # Similar texts
        text1 = "A red square"
        text2 = "A crimson square"
        
        emb1 = self.embedder.embed_text(text1)
        emb2 = self.embedder.embed_text(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        
        # Similar texts should have high similarity
        self.assertGreater(similarity, 0.7)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        if hasattr(cls, 'test_image'):
            cls.test_image.close()

if __name__ == '__main__':
    unittest.main()
