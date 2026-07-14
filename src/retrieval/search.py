import os
from langsmith import traceable
import pickle
from loguru import logger
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field

from src.config import settings
from src.database import qdrant_client

logger.info(f"Initializing embedding space with model: {settings.EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

COLLECTION_NAME = "enterprise_parent_child_kb"
BM25_CORPUS_PATH = "bm25_corpus.pkl"


class QdrantDenseRetriever(BaseRetriever):
    """
    Thin LangChain-compatible wrapper around our existing async Qdrant dense search,
    so it can sit inside an EnsembleRetriever alongside BM25Retriever.
    """
    top_k: int = Field(default=15)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # Sync fallback required by BaseRetriever's interface; we use the async path in practice.
        raise NotImplementedError("Use async retrieval via aget_relevant_documents.")

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        query_vector = embedding_model.encode(query).tolist()
        response = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=self.top_k
        )
        search_results = response.points

        documents = []
        for point in search_results:
            child_text = point.payload.get("child_text", "")
            parent_text = point.payload.get("parent_text", "")
            documents.append(
                Document(page_content=child_text, metadata={"parent_text": parent_text})
            )
        return documents


def load_bm25_retriever(top_k: int = 15) -> BM25Retriever | None:
    """
    Loads the persisted sparse corpus from disk and builds a fresh BM25Retriever.
    Returns None if no corpus has been indexed yet.
    """
    if not os.path.exists(BM25_CORPUS_PATH):
        logger.warning("BM25 corpus file not found on disk. Sparse retrieval will be skipped.")
        return None

    with open(BM25_CORPUS_PATH, "rb") as f:
        documents = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k
    return bm25_retriever

@traceable(name="hybrid_retrieval", tags=["retrieval"])
async def retrieve_context(query: str, top_k: int = 15) -> list[str]:
    """
    Executes hybrid retrieval (BM25 sparse + Qdrant dense, fused via EnsembleRetriever),
    then runs deep Cross-Encoder reranking on the fused candidate set.
    """
    logger.info(f"Initiating hybrid retrieval for query: '{query}'")

    try:
        dense_retriever = QdrantDenseRetriever(top_k=top_k)
        bm25_retriever = load_bm25_retriever(top_k=top_k)

        if bm25_retriever is None:
            # No corpus indexed yet — fall back to dense-only
            fused_documents = await dense_retriever._aget_relevant_documents(query)
        else:
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, dense_retriever],
                weights=[0.5, 0.5],  # Equal trust in sparse and dense signals
            )
            fused_documents = await ensemble_retriever.ainvoke(query)

        if not fused_documents:
            logger.warning("Hybrid retrieval returned an empty candidate list.")
            return []

        # Package candidates for FlashRank
        passages = []
        for index, doc in enumerate(fused_documents):
            parent_text = doc.metadata.get("parent_text", "")
            passages.append({
                "id": index,
                "text": doc.page_content,
                "meta": {"parent": parent_text}
            })

        logger.info(f"Routing {len(passages)} fused candidates into FlashRank Cross-Encoder pipeline...")
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_request)

        final_contexts = []
        for idx, result in enumerate(reranked_results[:6]):
            # Always keep the top-ranked result regardless of absolute score —
            # FlashRank already sorts by relevance, so #1 is its best guess even
            # when the score is low (e.g. single-fact queries score lower than
            # multi-entity ones due to how cross-encoders normalize confidence).
            # Threshold only gates whether to include additional supporting results.
            if idx == 0 or result.get("score", 0.0) > 0.05:
                parent_context = result.get("meta", {}).get("parent", "")
                if parent_context and parent_context not in final_contexts:
                    final_contexts.append(parent_context)

        logger.success(f"Context retrieval completed. Selected {len(final_contexts)} parent documents.")
        return final_contexts

    except Exception as e:
        logger.error(f"Failed to execute hybrid retrieval pipeline cleanly: {e}")
        return []