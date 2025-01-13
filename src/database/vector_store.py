from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from neo4j import GraphDatabase
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]
    type: str

class VectorStore:
    def __init__(self, neo4j_connection):
        """
        Initialize vector store for similarity search.
        
        Args:
            neo4j_connection: Neo4j database connection
        """
        self.neo4j = neo4j_connection
        
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def similar_items(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        item_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Find similar items using cosine similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            item_type: Filter by item type
            
        Returns:
            List[SearchResult]: Sorted list of similar items
        """
        # Query to get all relevant nodes
        type_filter = f"AND n.type = '{item_type}'" if item_type else ""
        query = f"""
        MATCH (n)
        WHERE (n:Chunk OR n:Message)
        {type_filter}
        RETURN n.id as id,
               n.content as content,
               n.embedding as embedding,
               n.metadata as metadata,
               n.type as type
        """
        
        try:
            # Get candidates
            results = []
            with self.neo4j.driver.session() as session:
                records = session.run(query).records()
                
                # Calculate similarities
                for record in records:
                    similarity = self._cosine_similarity(
                        query_embedding,
                        record['embedding']
                    )
                    
                    if similarity >= min_similarity:
                        results.append(SearchResult(
                            id=record['id'],
                            content=record['content'],
                            similarity=similarity,
                            metadata=record['metadata'],
                            type=record['type']
                        ))
                        
            # Sort by similarity
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
            
    def cross_modal_search(
        self,
        text_query: Optional[str] = None,
        image_query: Optional[str] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search across different modalities.
        
        Args:
            text_query: Text query
            image_query: Path to image query
            limit: Maximum number of results
            
        Returns:
            List[SearchResult]: Cross-modal search results
        """
        from ..processing.embeddings.clip_embedder import CLIPEmbedder
        
        try:
            embedder = CLIPEmbedder()
            
            # Generate query embedding
            if text_query and image_query:
                # Multimodal query
                query_embedding = embedder.embed_multimodal(
                    text_query,
                    image_query
                )
            elif text_query:
                # Text-only query
                query_embedding = embedder.embed_text(text_query)
            elif image_query:
                # Image-only query
                query_embedding = embedder.embed_image(image_query)
            else:
                raise ValueError("Must provide either text or image query")
                
            # Search using query embedding
            return self.similar_items(
                query_embedding,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Error in cross-modal search: {str(e)}")
            raise
            
    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        item_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search using semantic text understanding.
        
        Args:
            query: Text query
            limit: Maximum number of results
            item_type: Filter by item type
            
        Returns:
            List[SearchResult]: Semantic search results
        """
        from ..processing.embeddings.clip_embedder import CLIPEmbedder
        
        try:
            # Generate text embedding
            embedder = CLIPEmbedder()
            query_embedding = embedder.embed_text(query)
            
            # Search using text embedding
            return self.similar_items(
                query_embedding,
                limit=limit,
                item_type=item_type
            )
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise
            
    def image_search(
        self,
        image_path: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search using image similarity.
        
        Args:
            image_path: Path to query image
            limit: Maximum number of results
            
        Returns:
            List[SearchResult]: Image search results
        """
        from ..processing.embeddings.clip_embedder import CLIPEmbedder
        
        try:
            # Generate image embedding
            embedder = CLIPEmbedder()
            query_embedding = embedder.embed_image(image_path)
            
            # Search using image embedding
            return self.similar_items(
                query_embedding,
                limit=limit,
                item_type='image'
            )
            
        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            raise
