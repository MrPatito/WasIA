from typing import Optional, Dict, Any, Union
import redis
import json
import logging
from datetime import timedelta
import pickle

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 1800  # 30 minutes
    ):
        """
        Initialize Redis cache service.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default TTL in seconds
        """
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        self.default_ttl = default_ttl
        self.binary_redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False
        )
        
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ):
        """Set a value in cache with optional namespace."""
        try:
            full_key = f"{namespace}:{key}" if namespace else key
            
            # Handle different value types
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
                key_type = 'json'
            elif isinstance(value, bytes):
                self.binary_redis.set(
                    full_key,
                    value,
                    ex=ttl or self.default_ttl
                )
                return
            else:
                key_type = 'raw'
                
            # Store value with type information
            self.redis.hset(
                full_key,
                mapping={
                    'type': key_type,
                    'value': value
                }
            )
            
            if ttl or self.default_ttl:
                self.redis.expire(full_key, ttl or self.default_ttl)
                
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            
    def get(
        self,
        key: str,
        namespace: Optional[str] = None,
        default: Any = None
    ) -> Any:
        """Get a value from cache with optional namespace."""
        try:
            full_key = f"{namespace}:{key}" if namespace else key
            
            # Try binary data first
            binary_value = self.binary_redis.get(full_key)
            if binary_value is not None:
                return binary_value
                
            # Get value and type
            cached = self.redis.hgetall(full_key)
            if not cached:
                return default
                
            value = cached['value']
            key_type = cached['type']
            
            # Convert based on type
            if key_type == 'json':
                return json.loads(value)
            return value
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return default
            
    def delete(self, key: str, namespace: Optional[str] = None):
        """Delete a key from cache."""
        try:
            full_key = f"{namespace}:{key}" if namespace else key
            self.redis.delete(full_key)
            self.binary_redis.delete(full_key)
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            
    def set_vector(
        self,
        key: str,
        vector: Union[list, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ):
        """Store vector data efficiently."""
        try:
            # Store vector as binary
            if isinstance(vector, list):
                vector = pickle.dumps(vector)
            self.binary_redis.set(
                f"vector:{key}",
                vector,
                ex=ttl or self.default_ttl
            )
            
            # Store metadata if provided
            if metadata:
                self.set(
                    f"metadata:{key}",
                    metadata,
                    ttl=ttl
                )
                
        except Exception as e:
            logger.error(f"Error storing vector {key}: {str(e)}")
            
    def get_vector(
        self,
        key: str,
        with_metadata: bool = False
    ) -> Union[list, Dict[str, Any]]:
        """Retrieve vector data."""
        try:
            # Get vector
            vector_data = self.binary_redis.get(f"vector:{key}")
            if vector_data is None:
                return None
                
            vector = pickle.loads(vector_data)
            
            if not with_metadata:
                return vector
                
            # Get metadata if requested
            metadata = self.get(f"metadata:{key}", default={})
            return {
                'vector': vector,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error retrieving vector {key}: {str(e)}")
            return None
            
    def clear_namespace(self, namespace: str):
        """Clear all keys in a namespace."""
        try:
            pattern = f"{namespace}:*"
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                if keys:
                    self.redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Error clearing namespace {namespace}: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis.info()
            return {
                'used_memory': info['used_memory_human'],
                'connected_clients': info['connected_clients'],
                'total_keys': sum(
                    db['keys'] for db in info.values()
                    if isinstance(db, dict) and 'keys' in db
                ),
                'hits': info['keyspace_hits'],
                'misses': info['keyspace_misses'],
                'hit_rate': (
                    info['keyspace_hits'] /
                    (info['keyspace_hits'] + info['keyspace_misses'])
                    if (info['keyspace_hits'] + info['keyspace_misses']) > 0
                    else 0
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}
