"""
app/ingestion/ingest.py
───────────────────────
Full ingestion pipeline: load -> chunk -> embed -> Qdrant.
Can be run as a CLI script or called via the FastAPI endpoint.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.loader import load_directory, load_urls
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import get_vector_store
from app.retrieval.hybrid_retriever import initialize_bm25_from_docs

logger = logging.getLogger(__name__)
settings = get_settings()


async def ingest_pipeline(
    documents: List[Document], collection_name: str = None
) -> int:
    """
    Core pipeline to process and store documents.
    Returns the number of chunks added.
    """
    if not documents:
        logger.warning("No documents to ingest.")
        return 0

    chunks = chunk_documents(documents)
    if not chunks:
        return 0

    # Index the same chunks in BM25 so sparse hits carry `chunk_id` metadata
    # and can be cited exactly like dense hits.
    await initialize_bm25_from_docs(chunks)

    target_collection = collection_name or settings.qdrant_collection_name
    vector_store = await get_vector_store(collection_name=target_collection)

    logger.info(
        "Upserting %d chunks to Qdrant collection '%s'...",
        len(chunks),
        target_collection,
    )

    # Upsert in small batches to prevent WriteTimeout on Qdrant Cloud
    BATCH_SIZE = 50
    all_ids: List[str] = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info("  Batch %d/%d: upserting %d chunks...", batch_num, total_batches, len(batch))
        ids = await vector_store.aadd_documents(batch)
        all_ids.extend(ids)

    logger.info("Successfully upserted %d chunks.", len(all_ids))
    return len(all_ids)


async def run_cli_ingestion(source_path: str):
    """Entry point for CLI ingestion script."""
    path = Path(source_path)
    if not path.exists():
        logger.error("Path does not exist: %s", path)
        return

    logger.info("Starting local file ingestion from %s", path)
    docs = load_directory(path)
    await ingest_pipeline(docs)


if __name__ == "__main__":
    import asyncio
    import sys

    # Quick and dirty CLI entry point: python -m app.ingestion.ingest ./data/corpus
    source = sys.argv[1] if len(sys.argv) > 1 else "./data/corpus"
    asyncio.run(run_cli_ingestion(source))
