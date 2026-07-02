# src/ingestion/pipeline.py
import uuid
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import settings
from src.database import qdrant_client


logger.info(f"Initializing embedding space with model: {settings.EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

COLLECTION_NAME = "enterprise_parent_child_kb"

async def init_vector_db():
    """
    Verifies and guarantees collection structures inside Qdrant before writes.
    """
    try:
        collections_response = await qdrant_client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        
        if COLLECTION_NAME not in existing_collections:
            logger.info(f"Target collection '{COLLECTION_NAME}' absent. Initiating cluster creation sequence...")
            await qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,  
                    distance=Distance.COSINE
                )
            )
            logger.success(f"Successfully generated persistent cluster collection: '{COLLECTION_NAME}'")
        else:
            logger.info(f"Target collection '{COLLECTION_NAME}' verified online.")
    except Exception as e:
        logger.error(f"Failed to communicate with Qdrant collection layer initialization space: {e}")
        raise e

def chunk_text_hierarchical(text: str, parent_size: int = 1200, child_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Structuralizes raw text into high-density Parent-Child hierarchies.
    """
    chunks_hierarchy = []
    
    
    for i in range(0, len(text), parent_size - overlap):
        parent_chunk = text[i:i + parent_size].strip()
        if not parent_chunk:
            continue
            
        
        for j in range(0, len(parent_chunk), child_size - overlap):
            child_chunk = parent_chunk[j:j + child_size].strip()
            if len(child_chunk) < 40:  # Skips meaningless fragments
                continue
                
            chunks_hierarchy.append({
                "parent_text": parent_chunk,
                "child_text": child_chunk
            })
            
    return chunks_hierarchy

async def ingest_document(text: str, filename: str):
    """
    Parses, structuralizes, and upserts text payloads asynchronously inside the worker thread.
    """
    logger.info(f"Background worker processing text graph injection payload for target: '{filename}'")
    
    try:
        
        hierarchical_nodes = chunk_text_hierarchical(text)
        if not hierarchical_nodes:
            logger.warning("Chunking loop resulted in 0 indexable child nodes. Aborting pipeline process.")
            return

        child_texts = [node["child_text"] for node in hierarchical_nodes]
        
        
        logger.info(f"Computing matrix mathematical vector spaces for {len(child_texts)} child segments...")
        embeddings = embedding_model.encode(child_texts, show_progress_bar=False).tolist()

        
        points = []
        for idx, node in enumerate(hierarchical_nodes):
            point_id = str(uuid.uuid4())  
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[idx],
                    payload={
                        "filename": filename,
                        "child_text": node["child_text"],
                        "parent_text": node["parent_text"]  # Context retention link
                    }
                )
            )

        
        logger.info(f"Streaming and indexing database matrix payload rows into collection '{COLLECTION_NAME}'...")
        await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        logger.success(f"Asynchronous processing pipeline safely committed data entries from '{filename}' into Qdrant.")

    except Exception as e:
        logger.critical(f"Fatal background execution thread failure on ingestion pipeline matrix conversion: {e}")