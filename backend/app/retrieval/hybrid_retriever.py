"""
app/retrieval/hybrid_retriever.py
─────────────────────────────────
Combines Dense Retrieval (Vector) with Sparse Retrieval (BM25) using LangChain's EnsembleRetriever.
BM25 excels at exact keyword matches, while Dense excels at semantic meaning.
EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to merge the results.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List

from app.config import get_settings
from app.retrieval.vector_store import get_vector_store

settings = get_settings()

# We cache the BM25 index in memory for this example.
# In a true distributed system, you'd use Qdrant's native sparse vectors
# or Elasticsearch, but BM25Retriever is great for showcasing the Ensemble logic.
# NOTE: the cache is per-process and lost on restart, which is why app startup
# calls `rebuild_bm25_from_collection()` to repopulate it from Qdrant.
_bm25_retriever_cache = None
_bm25_documents: List[Document] = []


def reset_bm25_index() -> None:
    """Drop the in-memory BM25 index (used by tests and when re-seeding)."""
    global _bm25_retriever_cache, _bm25_documents
    _bm25_retriever_cache = None
    _bm25_documents = []


def is_bm25_ready() -> bool:
    """True when a BM25 index exists, i.e. retrieval is genuinely hybrid."""
    return _bm25_retriever_cache is not None


async def initialize_bm25_from_docs(documents: List[Document]) -> None:
    """
    Add documents to the BM25 index.

    Pass the *chunked* documents: BM25 hits must carry the same `chunk_id`
    metadata as dense hits, otherwise the LLM cannot cite them. Documents
    accumulate, so a second ingest does not discard the first one's index.
    """
    global _bm25_retriever_cache, _bm25_documents
    if not documents:
        return

    _bm25_documents.extend(documents)
    _bm25_retriever_cache = BM25Retriever.from_documents(_bm25_documents)
    _bm25_retriever_cache.k = settings.top_k


async def rebuild_bm25_from_collection(collection_name: str = None) -> int:
    """
    Rebuild the BM25 index from the chunks already stored in Qdrant.

    The in-memory index dies with the process, so a freshly started API server
    would silently serve dense-only results until someone re-ingested. Calling
    this at startup restores true hybrid retrieval. Returns the chunk count.
    """
    from app.retrieval.vector_store import scroll_all_documents

    documents = await scroll_all_documents(collection_name)
    if not documents:
        return 0

    reset_bm25_index()
    await initialize_bm25_from_docs(documents)
    return len(documents)


async def get_hybrid_retriever(collection_name: str = None) -> BaseRetriever:
    """
    Returns an EnsembleRetriever combining Qdrant (dense) and BM25 (sparse).
    Weights are set to 0.5 / 0.5 as requested by the user for balanced RRF.
    Falls back to the dense retriever alone when no BM25 index exists.
    """
    vector_store = await get_vector_store(collection_name)
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": settings.top_k})

    if _bm25_retriever_cache is None:
        # Fallback to pure dense if BM25 hasn't been initialized
        # (e.g., server restart without persistent BM25 storage)
        return dense_retriever

    # Reciprocal Rank Fusion formula is applied automatically by EnsembleRetriever:
    # score = 1 / (k + rank), where k=60 by default.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, _bm25_retriever_cache],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever
