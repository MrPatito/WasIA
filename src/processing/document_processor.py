from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes
import logging
import tempfile
import shutil
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .extractors.pdf_extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.image_extractor import ImageExtractor
from .extractors.whatsapp_extractor import WhatsAppExtractor, WhatsAppMessage
from .embeddings import BaseEmbedder, HuggingFaceEmbedder
from ..database.neo4j import Neo4jConnection
from ..config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self,
        neo4j_connection: Neo4jConnection,
        embedder: Optional[BaseEmbedder] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.neo4j = neo4j_connection
        # Default to HuggingFace embedder if none provided
        self.embedder = embedder or HuggingFaceEmbedder(
            model_name=config.embedding['model']
        )
        
        # Initialize tokenizer for length validation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.max_tokens_per_chunk = 8192  # OpenAI's limit, adjust based on model
        
        # Calculate chunk size in tokens
        target_tokens = min(self.max_tokens_per_chunk, 
                          chunk_size or config.embedding['max_chunk_size'])
        
        # Use configuration values for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_tokens,
            chunk_overlap=chunk_overlap or config.embedding['chunk_overlap'],
            length_function=lambda x: len(self.tokenizer.encode(x)),
            is_separator_regex=False
        )
        
        # Initialize extractors
        self.extractors = {
            'application/pdf': PDFExtractor(),
            'text/plain': TextExtractor(),
            'image': ImageExtractor(),
            'whatsapp': WhatsAppExtractor()
        }
        
        # Create temp directory
        self.temp_dir = config.get_temp_dir()

    def __del__(self):
        """Cleanup temporary files on deletion."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    def _validate_and_split_text(self, text: str) -> List[str]:
        """
        Validate text length and split into appropriate chunks.
        
        Args:
            text: Input text to validate and split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text.strip():
            raise ValueError("Empty text content")
            
        # Count tokens in full text
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        
        logger.info(f"Text length: {len(text)} chars, {token_count} tokens")
        
        if token_count > self.max_tokens_per_chunk * 1000:  # Reasonable limit for processing
            logger.warning(
                f"Text is very long ({token_count} tokens). "
                "Processing may take significant time and resources."
            )
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Validate each chunk
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(self.tokenizer.encode(chunk))
            if chunk_tokens > self.max_tokens_per_chunk:
                logger.warning(
                    f"Chunk {i} exceeds token limit "
                    f"({chunk_tokens} > {self.max_tokens_per_chunk}). "
                    "Splitting further."
                )
                # Recursively split large chunks
                subchunks = self.text_splitter.split_text(chunk)
                valid_chunks.extend(subchunks)
            else:
                valid_chunks.append(chunk)
        
        return valid_chunks

    def process_document(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a document and store it in the knowledge base.
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata about the document
            
        Returns:
            document_id: Unique identifier for the processed document
        """
        temp_file = None
        try:
            # Validate and get file type
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Copy file to temp directory for processing
            temp_file = self.temp_dir / path.name
            shutil.copy2(path, temp_file)
            
            mime_type = mimetypes.guess_type(temp_file)[0]
            if not mime_type:
                mime_type = 'text/plain'  # Default to text
                
            # Get appropriate extractor
            extractor = self._get_extractor(mime_type)
            if not extractor:
                raise ValueError(f"Unsupported file type: {mime_type}")
            
            # Extract text
            text = extractor.extract(str(temp_file))
            
            # Validate and split text into appropriate chunks
            chunks = self._validate_and_split_text(text)
            
            # Create documents with metadata
            docs = []
            for i, chunk in enumerate(chunks):
                # Validate chunk token count
                chunk_tokens = len(self.tokenizer.encode(chunk))
                doc_metadata = {
                    'source': file_path,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'token_count': chunk_tokens,
                    **metadata or {}
                }
                docs.append(Document(page_content=chunk, metadata=doc_metadata))
            
            # Generate embeddings with error handling
            try:
                embeddings = self.embedder.embed_batch([doc.page_content for doc in docs])
            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                raise
            
            # Store in Neo4j
            document_id = self._store_in_neo4j(docs, embeddings)
            
            logger.info(f"Successfully processed document: {file_path}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
            
        finally:
            # Cleanup temporary file
            if temp_file and temp_file.exists():
                temp_file.unlink()

    def process_whatsapp_message(self, message_data: Dict) -> str:
        """
        Process a WhatsApp message and store it in the knowledge base.
        
        Args:
            message_data: Dictionary containing WhatsApp message data
            
        Returns:
            str: Message ID in the knowledge base
        """
        try:
            # Validate message data
            required_fields = ['text', 'from', 'timestamp']
            missing_fields = [f for f in required_fields if f not in message_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Extract message using WhatsApp extractor
            extractor = self.extractors['whatsapp']
            message = extractor.extract(message_data)
            
            # Format for storage
            formatted_data = extractor.format_for_storage(message)
            
            # Validate content length
            if len(formatted_data['content']) > config.embedding['max_chunk_size']:
                logger.warning("Message content exceeds recommended length")
            
            # Create document with metadata
            doc = Document(
                page_content=formatted_data['content'],
                metadata=formatted_data['metadata']
            )
            
            # Generate embedding with error handling
            try:
                embedding = self.embedder.embed_text(doc.page_content)
            except Exception as e:
                logger.error(f"Error generating message embedding: {e}")
                raise
            
            # Store in Neo4j
            message_id = self._store_whatsapp_message(doc, embedding)
            
            logger.info(f"Successfully processed WhatsApp message from {message.sender}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {str(e)}")
            raise

    def process_whatsapp_chat(self, messages: List[Dict]) -> List[str]:
        """
        Process multiple WhatsApp messages in batch.
        
        Args:
            messages: List of WhatsApp message dictionaries
            
        Returns:
            List[str]: List of message IDs
        """
        if not messages:
            logger.warning("Empty message list provided")
            return []
            
        return [self.process_whatsapp_message(msg) for msg in messages]

    def _get_extractor(self, mime_type: str):
        """Get the appropriate extractor for the file type."""
        if mime_type.startswith('image/'):
            return self.extractors['image']
        return self.extractors.get(mime_type)

    def _store_in_neo4j(self, docs: List[Document], embeddings: List[List[float]]) -> str:
        """Store documents and their embeddings in Neo4j."""
        if len(docs) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
            
        # Create document node
        document_id = self.neo4j.create_document_node(
            source=docs[0].metadata['source'],
            metadata=docs[0].metadata
        )
        
        # Store chunks with embeddings
        for doc, embedding in zip(docs, embeddings):
            self.neo4j.create_chunk_node(
                document_id=document_id,
                content=doc.page_content,
                embedding=embedding,
                metadata=doc.metadata
            )
            
        return document_id

    def _store_whatsapp_message(self, doc: Document, embedding: List[float]) -> str:
        """Store WhatsApp message and its embedding in Neo4j."""
        # Create message node with metadata
        message_id = self.neo4j.create_message_node(
            content=doc.page_content,
            embedding=embedding,
            metadata=doc.metadata
        )
        
        return message_id
