"""
app/retrieval/vector_store.py
─────────────────────────────
Qdrant Cloud integration.
Initialises the Qdrant LangChain wrapper.
"""

import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.retrieval.embedder import get_embeddings

logger = logging.getLogger(__name__)
settings = get_settings()

# Page size when reading the whole collection back (BM25 rebuild).
SCROLL_BATCH_SIZE = 256

# We maintain a global client to avoid reconnect overhead
_client = None


def _get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _client


async def get_vector_store(collection_name: str = None) -> QdrantVectorStore:
    """
    Get the LangChain Qdrant wrapper (dense vectors only).

    Sparse/BM25 retrieval is handled application-side by `hybrid_retriever.py`
    via EnsembleRetriever, so the collection only needs dense vectors here.
    """
    col_name = collection_name or settings.qdrant_collection_name
    client = _get_qdrant_client()
    embeddings = get_embeddings()

    # The Qdrant client is synchronous — every call goes to a worker thread so
    # it cannot stall the event loop for concurrent requests.
    if not await run_in_threadpool(client.collection_exists, col_name):
        logger.info("Collection '%s' does not exist. Creating it now...", col_name)
        await run_in_threadpool(
            client.create_collection,
            collection_name=col_name,
            vectors_config=models.VectorParams(
                size=384,  # Default for all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            ),
        )

    # Initialize QdrantVectorStore
    return QdrantVectorStore(
        client=client,
        collection_name=col_name,
        embedding=embeddings,
        # Enable hybrid search natively in Qdrant (requires fastembed or server-side sparse).
        # We'll configure EnsembleRetriever at the app level if we want fine-grained control,
        # but LangChain Qdrant supports it out of the box if configured correctly.
        # For this setup, we'll return the base store and let `hybrid_retriever.py` manage the ensemble
        # or use native depending on what's configured.
    )


async def scroll_all_documents(collection_name: str = None) -> List[Document]:
    """
    Read every stored chunk back out of Qdrant as LangChain Documents.

    Used to rebuild the in-memory BM25 index at startup. `QdrantVectorStore`
    stores the text under the `page_content` payload key and the chunk metadata
    (including `chunk_id`) under `metadata`.
    """
    col_name = collection_name or settings.qdrant_collection_name
    client = _get_qdrant_client()

    if not await run_in_threadpool(client.collection_exists, col_name):
        logger.warning("Collection '%s' does not exist — nothing to scroll.", col_name)
        return []

    documents: List[Document] = []
    offset = None
    while True:
        points, offset = await run_in_threadpool(
            lambda off=offset: client.scroll(
                collection_name=col_name,
                limit=SCROLL_BATCH_SIZE,
                offset=off,
                with_payload=True,
                with_vectors=False,
            )
        )
        for point in points:
            payload = point.payload or {}
            content = payload.get("page_content", "")
            if not content:
                continue
            documents.append(
                Document(page_content=content, metadata=payload.get("metadata", {}) or {})
            )
        if offset is None:
            break

    logger.info("Scrolled %d chunks from collection '%s'.", len(documents), col_name)
    return documents


async def check_connection() -> bool:
    """Health check ping to Qdrant Cloud."""
    try:
        client = _get_qdrant_client()
        await run_in_threadpool(client.get_collections)
        return True
    except Exception as e:
        logger.error("Qdrant connection failed: %s", e)
        return False
