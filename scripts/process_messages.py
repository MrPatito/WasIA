#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.database.neo4j import Neo4jConnection
from src.processing.document_processor import DocumentProcessor
from src.config import Config

def main():
    # Set up Neo4j connection
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "testpassword"
    
    # Set CLIP model
    os.environ["EMBEDDING_MODEL"] = "openai/clip-vit-base-patch32"
    
    # Initialize connections
    db = Neo4jConnection.get_instance()
    db.connect()
    processor = DocumentProcessor(db)
    
    # Sample messages to process
    messages = [
        {
            "text": "WhatsApp is a great platform for communication",
            "metadata": {
                "type": "text_message",
                "source": "whatsapp",
                "timestamp": datetime.now().isoformat()
            }
        },
        {
            "text": "I love how AI can help us be more productive",
            "metadata": {
                "type": "text_message",
                "source": "whatsapp",
                "timestamp": datetime.now().isoformat()
            }
        },
        {
            "text": "Graph databases are perfect for storing relationships",
            "metadata": {
                "type": "text_message",
                "source": "whatsapp",
                "timestamp": datetime.now().isoformat()
            }
        }
    ]
    
    print("Starting to process messages...")
    
    # Process each message
    for i, msg in enumerate(messages, 1):
        print(f"\nProcessing message {i}/{len(messages)}")
        print(f"Text: {msg['text']}")
        
        try:
            # Process and store the message
            message_id = processor.process_message(
                msg["text"],
                metadata=msg["metadata"]
            )
            print(f"Successfully stored message with ID: {message_id}")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
    
    print("\nAll messages processed!")
    
    # Test similarity search
    print("\nTesting similarity search...")
    query = "communication platforms"
    similar_messages = db.find_similar_messages(
        processor.embedder.get_text_embedding(query).tolist(),
        limit=2
    )
    
    print(f"\nTop 2 messages similar to '{query}':")
    for msg in similar_messages:
        print(f"- {msg['text']}")

if __name__ == "__main__":
    main()
