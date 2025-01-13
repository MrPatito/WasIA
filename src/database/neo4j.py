from neo4j import GraphDatabase
from typing import Optional, Dict, List, Any
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class Neo4jConnection:
    _instance: Optional['Neo4jConnection'] = None

    def __init__(self):
        self._driver = None
        self._max_connection_pool_size = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))

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
            
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=self._max_connection_pool_size
            )
            
            # Verify connection and create constraints
            try:
                self._driver.verify_connectivity()
                self._create_constraints()
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def get_session(self):
        if not self._driver:
            self.connect()
        return self._driver.session()

    def _create_constraints(self):
        """Create necessary constraints for the database."""
        with self.get_session() as session:
            # Create constraints if they don't exist
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) REQUIRE m.id IS UNIQUE"
            ]
            for constraint in constraints:
                session.run(constraint)

    def create_document_node(self, source: str, metadata: Dict[str, Any]) -> str:
        """Create a document node in Neo4j."""
        try:
            doc_id = str(uuid.uuid4())
            query = """
            CREATE (d:Document {
                id: $id,
                source: $source,
                metadata: $metadata,
                created_at: datetime()
            })
            RETURN d.id as id
            """
            
            with self.get_session() as session:
                result = session.run(query, id=doc_id, source=source, metadata=metadata)
                return result.single()["id"]
                
        except Exception as e:
            logger.error(f"Error creating document node: {e}")
            raise

    def create_chunk_node(
        self,
        document_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """Create a chunk node and link it to its document."""
        try:
            chunk_id = str(uuid.uuid4())
            query = """
            MATCH (d:Document {id: $doc_id})
            CREATE (c:Chunk {
                id: $id,
                content: $content,
                embedding: $embedding,
                metadata: $metadata,
                created_at: datetime()
            })
            CREATE (d)-[:HAS_CHUNK]->(c)
            RETURN c.id as id
            """
            
            with self.get_session() as session:
                result = session.run(
                    query,
                    id=chunk_id,
                    doc_id=document_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata
                )
                return result.single()["id"]
                
        except Exception as e:
            logger.error(f"Error creating chunk node: {e}")
            raise

    def create_message_node(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> str:
        """Create a message node with its embedding."""
        try:
            message_id = str(uuid.uuid4())
            query = """
            CREATE (m:Message {
                id: $id,
                content: $content,
                embedding: $embedding,
                metadata: $metadata,
                created_at: datetime()
            })
            RETURN m.id as id
            """
            
            with self.get_session() as session:
                result = session.run(
                    query,
                    id=message_id,
                    content=content,
                    embedding=embedding,
                    metadata=metadata
                )
                return result.single()["id"]
                
        except Exception as e:
            logger.error(f"Error creating message node: {e}")
            raise

    def find_similar_chunks(
        self,
        embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find chunks similar to the given embedding."""
        try:
            query = """
            MATCH (c:Chunk)
            WITH c, gds.similarity.cosine(c.embedding, $embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN c.content as content, c.metadata as metadata, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            with self.get_session() as session:
                result = session.run(
                    query,
                    embedding=embedding,
                    threshold=similarity_threshold,
                    limit=limit
                )
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Error finding similar chunks: {e}")
            raise

    def find_similar_messages(
        self,
        embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find messages similar to the given embedding."""
        try:
            query = """
            MATCH (m:Message)
            WITH m, gds.similarity.cosine(m.embedding, $embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN m.content as content, m.metadata as metadata, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            with self.get_session() as session:
                result = session.run(
                    query,
                    embedding=embedding,
                    threshold=similarity_threshold,
                    limit=limit
                )
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Error finding similar messages: {e}")
            raise
