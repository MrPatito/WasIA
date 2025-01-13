from typing import Dict, Any
import os
from pathlib import Path
import logging
from logging.config import dictConfig

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration with environment variables."""
        # Neo4j Configuration
        self.neo4j = {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password'),
            'max_pool_size': int(os.getenv('NEO4J_MAX_POOL_SIZE', '50'))
        }

        # Redis Configuration
        self.redis = {
            'url': os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        }

        # RabbitMQ Configuration
        self.rabbitmq = {
            'url': os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
        }

        # API Configuration
        self.api = {
            'host': os.getenv('API_HOST', '0.0.0.0'),
            'port': int(os.getenv('API_PORT', '8000'))
        }

        # OpenAI Configuration
        self.openai = {
            'api_key': os.getenv('OPENAI_API_KEY')
        }

        # Embedding Configuration
        self.embedding = {
            'model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '1000')),
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '200'))
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on environment variables."""
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': os.getenv(
                        'LOG_FORMAT',
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                },
            },
            'handlers': {
                'default': {
                    'level': os.getenv('LOG_LEVEL', 'INFO'),
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout',
                },
            },
            'loggers': {
                '': {  # root logger
                    'handlers': ['default'],
                    'level': os.getenv('LOG_LEVEL', 'INFO'),
                    'propagate': True
                }
            }
        }
        dictConfig(log_config)

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    def get_temp_dir(self) -> Path:
        """Get the temporary directory for file processing."""
        temp_dir = self.project_root / 'temp'
        temp_dir.mkdir(exist_ok=True)
        return temp_dir

# Global configuration instance
config = Config()
