from typing import List, Dict, Any, Optional, Union
import logging
import mimetypes
import tempfile
import shutil
from pathlib import Path
import tiktoken
from PIL import Image

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .extractors.pdf_extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.image_extractor import ImageExtractor
from .extractors.audio_extractor import AudioExtractor
from .extractors.whatsapp_extractor import WhatsAppExtractor, WhatsAppMessage
from .embeddings.clip_embedder import CLIPEmbedder
from ..database.neo4j import Neo4jConnection
from ..config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(
        self,
        neo4j_connection: Neo4jConnection,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document processor with CLIP-based multimodal embeddings.
        """
        self.neo4j = neo4j_connection
        
        # Initialize CLIP embedder
        self.embedder = CLIPEmbedder(
            model_name=config.embedding['model']
        )
        
        # Initialize tokenizer for length validation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens_per_chunk = 77  # CLIP's max token length
        
        # Calculate chunk size in tokens
        target_tokens = min(
            self.max_tokens_per_chunk,
            chunk_size or config.embedding['max_chunk_size']
        )
        
        # Text splitter for chunking
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
            'audio': AudioExtractor(),
            'whatsapp': WhatsAppExtractor()
        }
        
        # MIME type mappings
        self.mime_mappings = {
            'audio/mpeg': 'audio',
            'audio/mp3': 'audio',
            'audio/wav': 'audio',
            'audio/ogg': 'audio',
            'audio/x-m4a': 'audio',
            'image/jpeg': 'image',
            'image/png': 'image',
            'image/gif': 'image'
        }
        
        # Create temp directory
        self.temp_dir = config.get_temp_dir()
        
    def _get_extractor(self, mime_type: str):
        """Get appropriate extractor for file type."""
        if mime_type.startswith('image/'):
            return self.extractors['image']
        elif mime_type in self.mime_mappings:
            return self.extractors[self.mime_mappings[mime_type]]
        return self.extractors.get(mime_type)
        
    def _validate_and_split_text(self, text: str) -> List[str]:
        """Validate and split text into appropriate chunks."""
        if not text.strip():
            raise ValueError("Empty text content")
            
        # Count tokens
        tokens = self.tokenizer.encode(text)
        token_count = len(tokens)
        
        logger.info(f"Text length: {len(text)} chars, {token_count} tokens")
        
        if token_count > self.max_tokens_per_chunk * 100:
            logger.warning(
                f"Text is very long ({token_count} tokens). "
                "Processing may take significant time."
            )
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Validate chunks
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(self.tokenizer.encode(chunk))
            if chunk_tokens > self.max_tokens_per_chunk:
                logger.warning(
                    f"Chunk {i} exceeds token limit. Splitting further."
                )
                subchunks = self.text_splitter.split_text(chunk)
                valid_chunks.extend(subchunks)
            else:
                valid_chunks.append(chunk)
                
        return valid_chunks
        
    def process_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a document and store in knowledge base."""
        temp_file = None
        try:
            # Validate file
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Copy to temp directory
            temp_file = self.temp_dir / path.name
            shutil.copy2(path, temp_file)
            
            # Get MIME type
            mime_type = mimetypes.guess_type(temp_file)[0]
            if not mime_type:
                mime_type = 'text/plain'
                
            # Get extractor
            extractor = self._get_extractor(mime_type)
            if not extractor:
                raise ValueError(f"Unsupported file type: {mime_type}")
                
            # Process based on type
            if mime_type.startswith('image/'):
                # For images, get both visual and text features
                image = Image.open(temp_file).convert('RGB')
                image_embedding = self.embedder.embed_image(image)
                
                # Extract text from image
                text = extractor.extract(str(temp_file))
                if text.strip():
                    # If text found, create multimodal embedding
                    embedding = self.embedder.embed_multimodal(text, image)
                else:
                    # If no text, use image embedding
                    embedding = image_embedding
                    
                # Create single document
                metadata_dict = metadata or {}
                doc = Document(
                    page_content=text if text.strip() else "[IMAGE]",
                    metadata={
                        'source': file_path,
                        'type': 'image',
                        'has_text': bool(text.strip()),
                        **metadata_dict
                    }
                )
                
                # Store in Neo4j
                return self._store_in_neo4j([doc], [embedding])
                
            else:
                # For text-based documents
                text = extractor.extract(str(temp_file))
                chunks = self._validate_and_split_text(text)
                
                # Create documents
                docs = []
                for i, chunk in enumerate(chunks):
                    chunk_tokens = len(self.tokenizer.encode(chunk))
                    metadata_dict = metadata or {}
                    doc_metadata = {
                        'source': file_path,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'token_count': chunk_tokens,
                        'type': mime_type,
                        **metadata_dict
                    }
                    docs.append(Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    ))
                
                # Generate embeddings
                embeddings = self.embedder.embed_batch(
                    [doc.page_content for doc in docs]
                )
                
                # Store in Neo4j
                return self._store_in_neo4j(docs, embeddings)
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
            
        finally:
            # Cleanup
            if temp_file and temp_file.exists():
                temp_file.unlink()
                
    def process_message(
        self,
        message: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a message (text, image, or voice)."""
        try:
            if isinstance(message, str):
                # Simple text message
                metadata_dict = metadata or {}
                embedding = self.embedder.embed_text(message)
                doc = Document(
                    page_content=message,
                    metadata={
                        'type': 'text_message',
                        **metadata_dict
                    }
                )
                return self._store_whatsapp_message(doc, embedding)
                
            elif isinstance(message, dict):
                # Mixed content message
                if 'image' in message:
                    # Image message
                    image = Image.open(message['image']).convert('RGB')
                    if 'caption' in message:
                        # Image with caption
                        embedding = self.embedder.embed_multimodal(
                            message['caption'],
                            image
                        )
                        content = f"{message['caption']} [IMAGE]"
                    else:
                        # Image only
                        embedding = self.embedder.embed_image(image)
                        content = "[IMAGE]"
                        
                    metadata_dict = metadata or {}
                    doc = Document(
                        page_content=content,
                        metadata={
                            'type': 'image_message',
                            **metadata_dict
                        }
                    )
                    return self._store_whatsapp_message(doc, embedding)
                    
                elif 'voice' in message:
                    # Voice message
                    return self.process_voice_message(
                        message['voice'],
                        metadata
                    )
                    
            raise ValueError("Unsupported message format")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise
            
    def _store_in_neo4j(
        self,
        docs: List[Document],
        embeddings: List[List[float]]
    ) -> str:
        """Store documents and embeddings in Neo4j."""
        try:
            # Create document node
            doc_id = self.neo4j.create_document_node(
                docs[0].metadata['source'],
                docs[0].metadata
            )
            
            # Create chunk nodes
            for doc, embedding in zip(docs, embeddings):
                self.neo4j.create_chunk_node(
                    doc_id,
                    doc.page_content,
                    embedding,
                    doc.metadata
                )
                
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing in Neo4j: {str(e)}")
            raise
            
    def _store_whatsapp_message(
        self,
        doc: Document,
        embedding: List[float]
    ) -> str:
        """Store WhatsApp message in Neo4j."""
        try:
            return self.neo4j.create_message_node(
                doc.page_content,
                embedding,
                doc.metadata
            )
        except Exception as e:
            logger.error(f"Error storing message: {str(e)}")
            raise
