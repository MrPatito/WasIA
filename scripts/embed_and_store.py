#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path
from PIL import Image
import mimetypes
from src.processing.embeddings.clip_embedder import CLIPEmbedder
from src.database.neo4j import Neo4jConnection
import PyPDF2
import docx

def get_file_content(file_path):
    """Extract content from different file types."""
    file_type, _ = mimetypes.guess_type(file_path)
    
    if file_type is None:
        # Try to guess by extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif']:
            return None, "image"
        elif ext == '.pdf':
            file_type = 'application/pdf'
        elif ext == '.docx':
            file_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif ext in ['.txt', '.md']:
            file_type = 'text/plain'
        else:
            return None, "unknown"
    
    if file_type.startswith('image/'):
        return None, "image"  # Images will be processed directly by CLIP
        
    elif file_type == 'application/pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text, "pdf"
            
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text, "docx"
        
    elif file_type.startswith('text/'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(), "text"
            
    return None, "unknown"

def main():
    # Set up Neo4j connection
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "testpassword"
    
    # Initialize connections
    db = Neo4jConnection.get_instance()
    db.connect()
    embedder = CLIPEmbedder()
    
    # Directory to process
    test_data_dir = Path("/home/catilo/Documents/WasIA/tests/test_data")
    
    print("Starting embedding and storage process...")
    
    # Walk through all files in the directory
    for root, _, files in os.walk(test_data_dir):
        for file in files:
            if file == "Thumbs.db":  # Skip Windows thumbnail files
                continue
                
            file_path = Path(root) / file
            print(f"\nProcessing file: {file_path}")
            
            try:
                content, file_type = get_file_content(str(file_path))
                
                if file_type == "image":
                    # Process image directly
                    image = Image.open(file_path)
                    embedding = embedder.embed_image(image)
                    metadata = {
                        "source": str(file_path.relative_to(test_data_dir)),
                        "type": "image",
                        "size": os.path.getsize(file_path)
                    }
                else:
                    if content is None:
                        print(f"Skipping unsupported file type: {file_path}")
                        continue
                        
                    # Generate text embedding
                    embedding = embedder.embed_text(content)
                    metadata = {
                        "source": str(file_path.relative_to(test_data_dir)),
                        "type": file_type,
                        "size": os.path.getsize(file_path)
                    }
                
                # Store in Neo4j
                doc_id = db.create_document_node(
                    source=str(file_path.relative_to(test_data_dir)),
                    metadata=metadata
                )
                
                chunk_id = db.create_chunk_node(
                    document_id=doc_id,
                    content=content if content else str(file_path),
                    embedding=embedding,
                    metadata=metadata
                )
                
                print(f"Successfully processed and stored: {file}")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    print("\nAll files processed!")
    
    # Test similarity search
    print("\nTesting similarity search...")
    query = "technical documentation about equipment"
    query_embedding = embedder.embed_text(query)
    
    similar_docs = db.find_similar_vectors(
        embedding=query_embedding,
        limit=3,
        node_type="Chunk"
    )
    
    print(f"\nTop 3 documents similar to '{query}':")
    for doc in similar_docs:
        metadata = doc.get('metadata', {})
        print(f"- {metadata.get('source', 'Unknown')} ({metadata.get('type', 'Unknown')})")

if __name__ == "__main__":
    main()
