# src/retrieval/search.py
from loguru import logger
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.database import qdrant_client

# Load local models for Vector Search generation and FlashRank Reranking
logger.info(f"Initializing embedding space with model: {settings.EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

COLLECTION_NAME = "enterprise_parent_child_kb"

async def retrieve_context(query: str, top_k: int = 15) -> list[str]:
    """
    Executes an asynchronous vector search, extracts text, and runs deep Cross-Encoder reranking.
    """
    logger.info(f"Initiating vector search space lookups for query: '{query}'")
    
    try:
        # Step 1: Compute matrix query mapping using local SentenceTransformer
        query_vector = embedding_model.encode(query).tolist()
        
        # Step 2: Query Qdrant asynchronously using query_points()
        response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        
        # Extract individual points out of the AsyncQdrantClient query wrapper
        search_results = response.points
        
        if not search_results:
            logger.warning("Qdrant cluster lookup returned an empty vector match list.")
            return []

        # Step 3: Package candidates explicitly for the FlashRank engine
        passages = []
        for index, point in enumerate(search_results):
            # Extract parent context block for context retention strategy
            parent_text = point.payload.get("parent_text", "")
            child_text = point.payload.get("child_text", "")
            
            passages.append({
                "id": index,
                "text": child_text,         # Used for ranking math matching
                "meta": {"parent": parent_text}  # Held for final contextual output
            })

        # Step 4: Run Deep Cross-Encoder Reranking
        logger.info(f"Routing {len(passages)} vectors into FlashRank Cross-Encoder pipeline...")
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)

        # Step 5: Extract the highest relevance Parent segments (Widen window to Top 6)
        final_contexts = []
        for result in reranked_results[:6]:  # <--- Increased from 3 to 6 to handle multi-project queries
            # Filter out low relevance scores to maintain strict context boundary
            if result.get("score", 0.0) > 0.05:  # <--- Gently tuned from 0.1 to 0.05 for comprehensive retrieval
                parent_context = result.get("meta", {}).get("parent", "")
                if parent_context and parent_context not in final_contexts:
                    final_contexts.append(parent_context)

        logger.success(f"Context retrieval completed. Selected {len(final_contexts)} parent documents.")
        return final_contexts

    except Exception as e:
        logger.error(f"Failed to query database vector search spaces cleanly: {e}")
        return []