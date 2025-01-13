import unittest
from datetime import datetime, timedelta
from src.database.neo4j import Neo4jConnection
import numpy as np
import time
import os
import json

class TestNeo4jConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test database connection."""
        # Ensure test database environment variables are set
        os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
        os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "testpassword")
        
        cls.db = Neo4jConnection.get_instance()
        cls.db.connect()
        
    def setUp(self):
        """Clear database before each test."""
        with self.db.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            
    def test_connection(self):
        """Test database connection and setup."""
        with self.db.get_session() as session:
            # Test constraints
            result = session.run("""
                SHOW CONSTRAINTS
                YIELD name, labelsOrTypes
                RETURN count(*) as count
            """)
            self.assertEqual(result.single()["count"], 3)  # Document, Chunk, Message
            
            # Test indexes
            result = session.run("""
                SHOW INDEXES
                YIELD name, type
                RETURN count(*) as count
            """)
            self.assertGreaterEqual(result.single()["count"], 3)
            
    def test_document_creation(self):
        """Test document node creation and retrieval."""
        # Create document
        doc_id = self.db.create_document_node(
            source="test.pdf",
            metadata={"type": "pdf", "pages": 10}
        )
        
        # Verify document
        with self.db.get_session() as session:
            result = session.run("""
                MATCH (d:Document {id: $id})
                RETURN d.source as source, d.metadata as metadata_str
            """, id=doc_id)
            record = result.single()
            self.assertEqual(record["source"], "test.pdf")
            self.assertEqual(json.loads(record["metadata_str"]), {"type": "pdf", "pages": 10})
            
    def test_chunk_creation(self):
        """Test chunk node creation and document relationship."""
        # Create document and chunk
        doc_id = self.db.create_document_node("test.pdf", {})
        embedding = [0.1] * 512  # CLIP embedding dimension
        
        chunk_id = self.db.create_chunk_node(
            document_id=doc_id,
            content="Test content",
            embedding=embedding,
            metadata={"page": 1}
        )
        
        # Verify chunk and relationship
        with self.db.get_session() as session:
            result = session.run("""
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
                RETURN c.content as content, c.metadata as metadata_str
            """, doc_id=doc_id, chunk_id=chunk_id)
            
            record = result.single()
            self.assertEqual(record["content"], "Test content")
            self.assertEqual(json.loads(record["metadata_str"]), {"page": 1})
            
    def test_message_creation(self):
        """Test message node creation with embedding."""
        embedding = [0.1] * 512
        message_id = self.db.create_message_node(
            content="Hello world",
            embedding=embedding,
            metadata={"source": "whatsapp"}
        )
        
        # Verify message
        with self.db.get_session() as session:
            result = session.run("""
                MATCH (m:Message {id: $id})
                RETURN m.content as content, m.metadata as metadata_str
            """, id=message_id)
            message = result.single()
            self.assertEqual(message["content"], "Hello world")
            self.assertEqual(json.loads(message["metadata_str"]), {"source": "whatsapp"})
            
    def test_vector_similarity_search(self):
        """Test vector similarity search with temporal boost."""
        # Create test vectors
        base_vector = np.random.rand(512)
        similar_vector = base_vector + np.random.normal(0, 0.1, 512)
        different_vector = np.random.rand(512)
        
        # Normalize vectors
        base_vector = base_vector / np.linalg.norm(base_vector)
        similar_vector = similar_vector / np.linalg.norm(similar_vector)
        different_vector = different_vector / np.linalg.norm(different_vector)
        
        # Create test nodes
        self.db.create_message_node(
            content="Similar message",
            embedding=similar_vector.tolist(),
            metadata={"test": True}
        )
        
        time.sleep(1)  # Ensure different timestamps
        
        self.db.create_message_node(
            content="Different message",
            embedding=different_vector.tolist(),
            metadata={"test": True}
        )
        
        # Search similar vectors
        results = self.db.find_similar_vectors(
            embedding=base_vector.tolist(),
            limit=10,
            min_similarity=0.5
        )
        
        # Verify results
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "Similar message")
        self.assertGreater(results[0]["similarity"], 0.8)
        
    def test_performance_monitoring(self):
        """Test query performance monitoring."""
        # Create test data with embeddings
        for _ in range(5):
            self.db.create_message_node(
                content="test",
                embedding=[0.1] * 512,
                metadata={}
            )
            
        # Get performance stats
        stats = self.db.get_performance_stats(
            query_type="create_message",
            time_window=timedelta(minutes=5)
        )
        
        # Verify stats
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["avg_embedding_size"], 512)
        
    def test_caching(self):
        """Test query result caching."""
        # Create test data
        embedding = np.random.rand(512).tolist()
        self.db.create_message_node(
            content="Test message",
            embedding=embedding,
            metadata={}
        )
        
        # First query - should cache
        start_time = time.time()
        first_results = self.db.find_similar_vectors(
            embedding=embedding,
            limit=10
        )
        first_query_time = time.time() - start_time
        
        # Second query - should use cache
        start_time = time.time()
        second_results = self.db.find_similar_vectors(
            embedding=embedding,
            limit=10
        )
        second_query_time = time.time() - start_time
        
        # Verify cache effectiveness
        self.assertEqual(first_results, second_results)
        self.assertLess(second_query_time, first_query_time)
        
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test invalid document ID
        with self.assertRaises(Exception):
            self.db.create_chunk_node(
                document_id="invalid_id",
                content="Test",
                embedding=[0.1] * 512,
                metadata={}
            )
            
        # Test invalid embedding dimension
        with self.assertRaises(ValueError):
            self.db.create_message_node(
                content="Test",
                embedding=[0.1] * 10,  # Wrong dimension
                metadata={}
            )
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        with cls.db.get_session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        cls.db.close()

if __name__ == '__main__':
    unittest.main()
