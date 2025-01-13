from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class WhatsAppMessage:
    def __init__(
        self,
        content: str,
        sender: str,
        timestamp: datetime,
        message_type: str = "text",
        metadata: Optional[Dict] = None
    ):
        self.content = content
        self.sender = sender
        self.timestamp = timestamp
        self.message_type = message_type  # text, image, audio, etc.
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "metadata": self.metadata
        }

class WhatsAppExtractor:
    def extract(self, message_data: Dict) -> WhatsAppMessage:
        """
        Extract and process WhatsApp message data.
        
        Args:
            message_data: Dictionary containing WhatsApp message information
                {
                    "text": str,
                    "from": str,
                    "timestamp": str,
                    "type": str,
                    "metadata": dict
                }
            
        Returns:
            WhatsAppMessage: Processed message object
        """
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(message_data["timestamp"])
            
            # Create WhatsApp message object
            message = WhatsAppMessage(
                content=message_data["text"],
                sender=message_data["from"],
                timestamp=timestamp,
                message_type=message_data.get("type", "text"),
                metadata=message_data.get("metadata", {})
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp message: {str(e)}")
            raise

    def batch_extract(self, messages: List[Dict]) -> List[WhatsAppMessage]:
        """
        Process multiple WhatsApp messages at once.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List[WhatsAppMessage]: List of processed message objects
        """
        return [self.extract(msg) for msg in messages]

    def format_for_storage(self, message: WhatsAppMessage) -> Dict:
        """
        Format message for storage in the knowledge base.
        
        Args:
            message: WhatsApp message object
            
        Returns:
            Dict: Formatted message data
        """
        return {
            "content": message.content,
            "metadata": {
                "sender": message.sender,
                "timestamp": message.timestamp.isoformat(),
                "message_type": message.message_type,
                "source": "whatsapp",
                **message.metadata
            }
        }
