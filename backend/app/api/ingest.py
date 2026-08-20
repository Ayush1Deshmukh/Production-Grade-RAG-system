"""
app/api/ingest.py
─────────────────
POST /ingest endpoint to load URLs into the vector store dynamically.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.models import IngestRequest, IngestResponse
from app.api.security import require_ingest_key
from app.ingestion.loader import load_urls
from app.ingestion.ingest import ingest_pipeline
from app.ingestion.url_validation import UrlNotAllowed, validate_ingest_urls

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(require_ingest_key)],
)
async def ingest_urls(request: IngestRequest) -> IngestResponse:
    """
    Scrape given URLs, chunk them, embed, and upsert to Qdrant.
    `ingest_pipeline` also updates the in-memory BM25 index with the new chunks.

    Requires the `X-API-Key` header, and every URL must pass the SSRF guard.
    """
    # Reject unsafe targets before the server issues any request.
    try:
        validate_ingest_urls(request.urls)
    except UrlNotAllowed as exc:
        logger.warning("Rejected ingest URLs: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        # Load from web
        docs = load_urls(request.urls)

        # Chunk -> BM25 index -> embed -> Qdrant
        chunks_added = await ingest_pipeline(docs, request.collection_name)

        return IngestResponse(
            status="success",
            chunks_added=chunks_added,
            collection_name=request.collection_name or settings.qdrant_collection_name,
            message=f"Successfully ingested {len(docs)} documents."
        )

    except Exception:
        # Log the full traceback server-side; never echo internals (URLs, keys) to the client.
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail="Ingestion failed. See server logs for details.")
