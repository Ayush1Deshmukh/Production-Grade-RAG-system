"""
app/api/query.py
────────────────
POST /query endpoint for the RAG system.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse
from app.rag.chain import execute_rag_pipeline
from app.observability.langfuse_handler import get_langfuse_handler

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Submit a question to the RAG pipeline.
    Uses Langfuse for tracing if configured.
    """
    logger.info("Received query: %s", request.question)

    # Optional: Setup observability callback
    callbacks = []
    langfuse_handler = get_langfuse_handler(session_id=request.session_id)
    if langfuse_handler:
        callbacks.append(langfuse_handler)

    try:
        result_dict = await execute_rag_pipeline(
            question=request.question,
            collection_name=request.collection_name,
            callbacks=callbacks,
        )

        # Flush Langfuse traces in the background so we don't delay the response
        if langfuse_handler:
            langfuse_handler.flush()

        return QueryResponse(**result_dict)

    except Exception:
        # Log the full traceback server-side; never echo internals (URLs, keys) to the client.
        logger.exception("Pipeline execution failed")
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline failed. See server logs for details.",
        )
