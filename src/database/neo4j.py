from neo4j import GraphDatabase
from typing import Optional
import os

class Neo4jConnection:
    _instance: Optional['Neo4jConnection'] = None

    def __init__(self):
        self._driver = None

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
            
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # Verify connection
            try:
                self._driver.verify_connectivity()
            except Exception as e:
                print(f"Failed to connect to Neo4j: {e}")
                raise

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def get_session(self):
        if not self._driver:
            self.connect()
        return self._driver.session()
