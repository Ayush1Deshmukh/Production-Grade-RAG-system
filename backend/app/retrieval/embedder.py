"""
app/retrieval/embedder.py
─────────────────────────
Embeddings wrapper.
Uses the local HuggingFace sentence-transformer 'all-MiniLM-L6-v2' (384 dims),
which runs on-device with no API calls or rate limits. The Qdrant collection is
created with size=384 to match — changing the model means recreating the collection.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached singleton of the embeddings model."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
