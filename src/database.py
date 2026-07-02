# src/database.py
from qdrant_client import AsyncQdrantClient
from src.config import settings
from loguru import logger

# Initialize the shared asynchronous Qdrant client
try:
    qdrant_client = AsyncQdrantClient(
        host=settings.QDRANT_HOST, 
        port=settings.QDRANT_PORT
    )
    logger.info(f"Asynchronous Qdrant client successfully routed to {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
except Exception as e:
    logger.critical(f"Failed to establish connection payload for Qdrant client: {e}")
    raise e

# Using a distinct collection name for our advanced parent-child RAG pipeline
COLLECTION_NAME = "enterprise_parent_child_kb"