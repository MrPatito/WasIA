#!/usr/bin/env python3
import os
import logging
from pathlib import Path
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.database.neo4j import Neo4jConnection
from src.processing.document_processor import DocumentProcessor
from src.processing.batch_processor import BatchProcessor
from src.monitoring import HardwareMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_connections():
    """Initialize database connections."""
    # Neo4j connection settings
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"
    
    neo4j = Neo4jConnection(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )
    
    return neo4j

async def process_directory(input_dir: str, neo4j: Neo4jConnection):
    """Process all documents in a directory."""
    # Initialize processors
    doc_processor = DocumentProcessor(
        neo4j_connection=neo4j,
        embedding_model='all-MiniLM-L6-v2',  # Fast and efficient model
        similarity_threshold=0.75,  # Adjust based on your needs
        min_cluster_size=2  # Minimum entities to form a relationship cluster
    )
    
    batch_processor = BatchProcessor(
        neo4j_connection=neo4j,
        max_workers=4  # Adjust based on your CPU cores
    )
    
    # Start hardware monitoring
    monitor = HardwareMonitor()
    monitor.start()
    
    try:
        # Process all documents
        logger.info(f"Starting document processing from {input_dir}")
        results = await batch_processor.process_directory(
            input_dir,
            recursive=True
        )
        
        # Print processing summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Successfully processed: {results['processed_count']} documents")
        logger.info(f"Failed to process: {results['failed_count']} documents")
        
        if results['failures']:
            logger.warning("\nFailed documents:")
            for failure in results['failures']:
                logger.warning(f"  - {failure['file']}: {failure['error']}")
        
        # Get relationship statistics
        stats = await get_relationship_stats(neo4j)
        logger.info("\nRelationship Statistics:")
        logger.info(f"Total relationships discovered: {stats['total_relationships']}")
        logger.info("\nTop relationship types:")
        for rel_type, count in stats['relationship_types'].items():
            logger.info(f"  - {rel_type}: {count}")
        
        logger.info("\nTop entity types:")
        for entity_type, count in stats['entity_types'].items():
            logger.info(f"  - {entity_type}: {count}")
        
    finally:
        # Stop hardware monitoring
        monitor.stop()
        monitor_stats = monitor.get_statistics()
        
        logger.info("\nHardware Usage Statistics:")
        logger.info(f"Average CPU Usage: {monitor_stats['avg_cpu']:.2f}%")
        logger.info(f"Average Memory Usage: {monitor_stats['avg_memory']:.2f}%")
        logger.info(f"Peak Memory Usage: {monitor_stats['peak_memory']:.2f}%")
        logger.info(f"Processing Duration: {monitor_stats['duration']:.2f} seconds")

async def get_relationship_stats(neo4j):
    """Get statistics about discovered relationships."""
    with neo4j.driver.session() as session:
        # Get relationship counts
        result = session.run("""
            MATCH ()-[r]->()
            WITH type(r) as rel_type, count(*) as count
            RETURN collect({type: rel_type, count: count}) as relationships
        """)
        rel_stats = result.single()['relationships']
        
        # Get entity counts
        result = session.run("""
            MATCH (e:Entity)
            WITH e.type as entity_type, count(*) as count
            RETURN collect({type: entity_type, count: count}) as entities
        """)
        entity_stats = result.single()['entities']
        
        return {
            'total_relationships': sum(r['count'] for r in rel_stats),
            'relationship_types': {r['type']: r['count'] for r in rel_stats},
            'entity_types': {e['type']: e['count'] for e in entity_stats}
        }

async def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)
    
    # Setup connections
    neo4j = setup_connections()
    
    try:
        # Process directory
        await process_directory(input_dir, neo4j)
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        # Close connections
        neo4j.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
