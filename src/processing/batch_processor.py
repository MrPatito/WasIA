from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import queue
import threading
from dataclasses import dataclass
from datetime import datetime

from .document_processor import DocumentProcessor
from ..database.neo4j import Neo4jConnection

logger = logging.getLogger(__name__)

@dataclass
class ProcessingItem:
    id: str
    path: str
    type: str
    metadata: Dict[str, Any]
    created_at: datetime
    status: str = 'pending'
    error: Optional[str] = None

class BatchProcessor:
    def __init__(
        self,
        neo4j_connection: Neo4jConnection,
        batch_size: int = 32,
        max_workers: int = 4
    ):
        """
        Initialize batch processor.
        
        Args:
            neo4j_connection: Neo4j database connection
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of parallel workers
        """
        self.doc_processor = DocumentProcessor(neo4j_connection)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Processing queues
        self.input_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.completed_queue = queue.Queue()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing status
        self.items: Dict[str, ProcessingItem] = {}
        self.is_running = False
        self.lock = threading.Lock()
        
    async def start(self):
        """Start the batch processor."""
        self.is_running = True
        
        # Start processing loop
        asyncio.create_task(self._process_loop())
        
    async def stop(self):
        """Stop the batch processor."""
        self.is_running = False
        self.executor.shutdown(wait=True)
        
    def add_item(
        self,
        file_path: str,
        item_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add an item to the processing queue.
        
        Args:
            file_path: Path to the file
            item_type: Type of item ('document', 'message', etc.)
            metadata: Additional metadata
            
        Returns:
            str: Item ID for tracking
        """
        item = ProcessingItem(
            id=f"item_{len(self.items)}_{datetime.now().timestamp()}",
            path=file_path,
            type=item_type,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        with self.lock:
            self.items[item.id] = item
            self.input_queue.put(item)
            
        return item.id
        
    def get_status(self, item_id: str) -> Dict[str, Any]:
        """Get processing status of an item."""
        item = self.items.get(item_id)
        if not item:
            raise ValueError(f"Item {item_id} not found")
            
        return {
            'id': item.id,
            'status': item.status,
            'error': item.error,
            'created_at': item.created_at.isoformat(),
            'type': item.type,
            'metadata': item.metadata
        }
        
    async def _process_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Collect items for current batch
                batch = []
                while len(batch) < self.batch_size:
                    try:
                        item = self.input_queue.get_nowait()
                        batch.append(item)
                    except queue.Empty:
                        break
                        
                if not batch:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Process batch in parallel
                futures = []
                for item in batch:
                    item.status = 'processing'
                    if item.type == 'document':
                        future = self.executor.submit(
                            self.doc_processor.process_document,
                            item.path,
                            item.metadata
                        )
                    elif item.type == 'message':
                        future = self.executor.submit(
                            self.doc_processor.process_message,
                            item.path,
                            item.metadata
                        )
                    futures.append((item, future))
                    
                # Handle results
                for item, future in futures:
                    try:
                        result = future.result()
                        item.status = 'completed'
                        self.completed_queue.put((item.id, result))
                    except Exception as e:
                        logger.error(f"Error processing {item.id}: {str(e)}")
                        item.status = 'error'
                        item.error = str(e)
                        
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(1)
                
    async def process_directory(self, directory: str) -> List[str]:
        """
        Process all files in a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            List[str]: List of item IDs
        """
        item_ids = []
        directory = Path(directory)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                item_id = self.add_item(
                    str(file_path),
                    'document',
                    {'source_dir': str(directory)}
                )
                item_ids.append(item_id)
                
        return item_ids
