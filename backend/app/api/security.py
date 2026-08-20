"""
app/api/security.py
───────────────────
Shared-secret guard for the write endpoints.

/ingest causes the server to fetch external URLs and mutate the vector store,
so it must not be callable by anyone who can reach the API. The key is compared
with `secrets.compare_digest` to avoid leaking length/prefix through timing.
"""

import logging
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

API_KEY_HEADER = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


async def require_ingest_key(api_key: str = Security(_api_key_scheme)) -> None:
    """
    FastAPI dependency enforcing the ingest shared secret.

    Fails closed: with no INGEST_API_KEY configured the endpoint is disabled
    rather than left open to the internet.
    """
    if not settings.ingest_enabled:
        logger.warning("Rejected /ingest call: INGEST_API_KEY is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingest endpoint is disabled. Set INGEST_API_KEY to enable it.",
        )

    if not api_key or not secrets.compare_digest(api_key, settings.ingest_api_key):
        logger.warning("Rejected /ingest call: missing or invalid %s header", API_KEY_HEADER)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER} header.",
        )
