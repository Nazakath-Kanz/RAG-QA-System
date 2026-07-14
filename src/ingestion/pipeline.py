# src/ingestion/pipeline.py
import uuid
import pickle
import os
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.database import qdrant_client


logger.info(f"Initializing embedding space with model: {settings.EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

COLLECTION_NAME = "enterprise_parent_child_kb"
BM25_CORPUS_PATH = "bm25_corpus.pkl"

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " ", ""],
)

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

def chunk_text_hierarchical(text: str) -> list[dict]:
    """
    Structuralizes raw text into Parent-Child hierarchies using line/paragraph-aware
    recursive splitting, so rows and facts don't get sliced across chunk boundaries.
    """
    chunks_hierarchy = []
    parent_chunks = parent_splitter.split_text(text)

    for parent_chunk in parent_chunks:
        parent_chunk = parent_chunk.strip()
        if not parent_chunk:
            continue

        child_chunks = child_splitter.split_text(parent_chunk)

        for child_chunk in child_chunks:
            child_chunk = child_chunk.strip()
            if len(child_chunk) < 20:
                continue

            chunks_hierarchy.append({
                "parent_text": parent_chunk,
                "child_text": child_chunk
            })

    return chunks_hierarchy

def save_bm25_corpus(hierarchical_nodes: list[dict], filename: str):
    """
    Persists the child chunks as LangChain Document objects to disk,
    so the BM25 sparse retriever can be rebuilt on demand at query time
    without needing a live in-memory reference across requests/restarts.
    """
    documents = [
        Document(
            page_content=node["child_text"],
            metadata={"parent_text": node["parent_text"], "filename": filename}
        )
        for node in hierarchical_nodes
    ]

    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(documents, f)

    logger.success(f"Persisted {len(documents)} documents to BM25 corpus store at '{BM25_CORPUS_PATH}'.")

async def ingest_document(text: str, filename: str):
    """
    Parses, structuralizes, and upserts text payloads asynchronously inside the worker thread.
    Also rebuilds the sparse BM25 corpus alongside the dense Qdrant index.
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
                        "parent_text": node["parent_text"]
                    }
                )
            )

        logger.info(f"Streaming and indexing database matrix payload rows into collection '{COLLECTION_NAME}'...")
        await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        # Rebuild sparse corpus so BM25 stays in sync with what's in Qdrant
        save_bm25_corpus(hierarchical_nodes, filename)

        logger.success(f"Asynchronous processing pipeline safely committed data entries from '{filename}' into Qdrant.")

    except Exception as e:
        logger.critical(f"Fatal background execution thread failure on ingestion pipeline matrix conversion: {e}")