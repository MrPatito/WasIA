from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List, Union
import os
import logging
import time
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class QueryStats:
    query_type: str
    execution_time: float
    timestamp: datetime
    parameters: Dict[str, Any]

class Neo4jConnection:
    _instance: Optional['Neo4jConnection'] = None
    
    def __init__(self):
        self._driver = None
        self._stats: List[QueryStats] = []
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=30)
        
    @classmethod
    def get_instance(cls) -> 'Neo4jConnection':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def connect(self):
        if not self._driver:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            max_pool_size = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))
            
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=max_pool_size
            )
            
            # Verify connection
            try:
                self._driver.verify_connectivity()
                self._setup_database()
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise
                
    def _setup_database(self):
        """Setup database indexes and constraints."""
        with self.get_session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document)
                REQUIRE d.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk)
                REQUIRE c.id IS UNIQUE
            """)
            
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message)
                REQUIRE m.id IS UNIQUE
            """)
            
    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
            
    def get_session(self):
        if not self._driver:
            self.connect()
        return self._driver.session()
        
    def _log_query(self, query_type: str, start_time: float, params: Dict[str, Any]):
        """Log query statistics."""
        execution_time = time.time() - start_time
        self._stats.append(QueryStats(
            query_type=query_type,
            execution_time=execution_time,
            timestamp=datetime.now(),
            parameters=params
        ))
        
    def _cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{query}:{param_str}"
        
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._cache_ttl:
                return result
            del self._cache[key]
        return None
        
    def _convert_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Convert metadata string back to dictionary."""
        if not metadata_str:
            return {}
        return json.loads(metadata_str)

    def create_document_node(
        self,
        source: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Create a document node with metadata."""
        try:
            with self.get_session() as session:
                # Convert metadata to string to store as property
                metadata_str = json.dumps(metadata)
                result = session.run("""
                    CREATE (d:Document {
                        id: randomUUID(),
                        source: $source,
                        metadata: $metadata,
                        created_at: datetime()
                    })
                    RETURN d.id as id, d.metadata as metadata
                """, source=source, metadata=metadata_str)
                record = result.single()
                doc_id = str(record['id'])
                return doc_id
        except Exception as e:
            logger.error(f"Error creating document node: {str(e)}")
            raise
            
    def create_chunk_node(
        self,
        document_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """Create a chunk node and link to document."""
        try:
            with self.get_session() as session:
                # Convert metadata to string
                metadata_str = json.dumps(metadata)
                result = session.run("""
                    MATCH (d:Document {id: $doc_id})
                    CREATE (c:Chunk {
                        id: randomUUID(),
                        content: $content,
                        embedding: $embedding,
                        metadata: $metadata,
                        created_at: datetime()
                    })
                    CREATE (d)-[:HAS_CHUNK]->(c)
                    RETURN c.id as id, c.metadata as metadata
                """, doc_id=document_id, content=content, 
                     embedding=embedding, metadata=metadata_str)
                record = result.single()
                chunk_id = str(record['id'])
                return chunk_id
        except Exception as e:
            logger.error(f"Error creating chunk node: {str(e)}")
            raise
            
    def create_message_node(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """Create a message node with embedding."""
        try:
            # Validate embedding dimension
            if len(embedding) != 512:
                raise ValueError(f"Expected embedding dimension of 512, got {len(embedding)}")
                
            with self.get_session() as session:
                # Convert metadata to string
                metadata_str = json.dumps(metadata)
                result = session.run("""
                    CREATE (m:Message {
                        id: randomUUID(),
                        content: $content,
                        embedding: $embedding,
                        metadata: $metadata,
                        created_at: datetime()
                    })
                    RETURN m.id as id, m.metadata as metadata
                """, content=content, embedding=embedding, 
                     metadata=metadata_str)
                record = result.single()
                message_id = str(record['id'])
                return message_id
        except Exception as e:
            logger.error(f"Error creating message node: {str(e)}")
            raise
            
    def find_similar_vectors(
        self,
        embedding: List[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        node_type: Optional[str] = None,
        temporal_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find similar vectors using cosine similarity and temporal boost.
        
        Args:
            embedding: Query vector
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            node_type: Filter by node type
            temporal_weight: Weight for temporal boost (0-1)
        """
        cache_key = self._cache_key('find_similar', {
            'embedding': embedding,
            'limit': limit,
            'min_similarity': min_similarity,
            'node_type': node_type
        })
        
        cached = self._get_cached(cache_key)
        if cached:
            return cached
            
        start_time = time.time()
        
        try:
            # Convert query vector to numpy array
            query_vector = np.array(embedding)
            query_norm = np.linalg.norm(query_vector)
            
            # Build type filter
            type_filter = ""
            if node_type:
                type_filter = f"WHERE n:{node_type}"
                
            query = f"""
            MATCH (n)
            {type_filter}
            RETURN n.id as id,
                   n.content as content,
                   n.metadata as metadata,
                   n.embedding as embedding,
                   n.created_at.epochSeconds as created_at,
                   labels(n)[0] as type
            """
            
            with self.get_session() as session:
                result = session.run(query)
                records = list(result)
                
                # Calculate similarities and scores
                similarities = []
                for record in records:
                    vector = np.array(record['embedding'])
                    similarity = np.dot(query_vector, vector) / (query_norm * np.linalg.norm(vector))
                    
                    if similarity >= min_similarity:
                        # Convert epoch seconds to hours ago
                        created_at = datetime.fromtimestamp(record['created_at'])
                        hours_ago = (datetime.now() - created_at).total_seconds() / 3600
                        
                        # Calculate temporal decay
                        temporal_decay = 1.0 / (1.0 + temporal_weight * hours_ago)
                        
                        # Combine similarity and temporal boost
                        score = similarity * temporal_decay
                        
                        similarities.append({
                            'id': record['id'],
                            'content': record['content'],
                            'metadata': self._convert_metadata(record['metadata']),
                            'type': record['type'],
                            'similarity': similarity,
                            'score': score
                        })
                        
                # Sort by score and limit results
                results = sorted(similarities, key=lambda x: x['score'], reverse=True)[:limit]
                
                # Cache results
                self._cache[cache_key] = (results, datetime.now())
                
                return results
                
        except Exception as e:
            logger.error(f"Error finding similar vectors: {str(e)}")
            raise
            
    def get_performance_stats(
        self,
        query_type: str,
        time_window: timedelta
    ) -> Dict[str, Any]:
        """Get performance statistics for a query type."""
        try:
            with self.get_session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.created_at >= datetime() - duration({seconds: $seconds})
                    RETURN count(n) as count,
                           avg(size(n.embedding)) as avg_embedding_size
                """, seconds=int(time_window.total_seconds()))
                stats = result.single()
                return {
                    'count': stats['count'],
                    'avg_embedding_size': stats['avg_embedding_size']
                }
        except Exception as e:
            logger.error(f"Error getting performance stats: {str(e)}")
            raise
            
    def cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._cache_ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
