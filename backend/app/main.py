"""
app/main.py
───────────
FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.models import HealthResponse
from app.api import ingest as ingest_router
from app.api import query as query_router
from app.retrieval.vector_store import check_connection
from app.retrieval.hybrid_retriever import rebuild_bm25_from_collection
from app.retrieval.embedder import get_embeddings
from app.retrieval.reranker import _get_cross_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("RAG backend starting up (env=%s)", settings.environment)
    logger.info("Collection: %s", settings.qdrant_collection_name)

    # Load the embedding and cross-encoder weights now. Otherwise the first user
    # query pays ~10s of model loading, and that warm-up time lands in the
    # latency breakdown as if it were retrieval cost.
    try:
        await run_in_threadpool(get_embeddings().embed_query, "warmup")
        await run_in_threadpool(_get_cross_encoder().score, [["warmup", "warmup"]])
        logger.info("Embedding + cross-encoder models warmed up")
    except Exception:
        logger.exception("Model warm-up failed — first query will be slower")

    # The BM25 index is in-memory, so it is empty on every boot. Rebuild it from
    # the chunks already in Qdrant, otherwise retrieval silently degrades to
    # dense-only until the next ingest. A failure here must not block startup.
    try:
        indexed = await rebuild_bm25_from_collection()
        if indexed:
            logger.info("BM25 index rebuilt from %d stored chunks (hybrid retrieval active)", indexed)
        else:
            logger.warning("No stored chunks found — retrieval will be dense-only until you ingest.")
    except Exception:
        logger.exception("BM25 rebuild failed — continuing with dense-only retrieval")

    yield
    logger.info("RAG backend shutting down")


app = FastAPI(
    title="Production-Grade RAG API",
    description="Hybrid retrieval (Dense + BM25) with local cross-encoder re-ranking and Cerebras inference.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

def _cors_policy() -> tuple[list[str], bool]:
    """
    Resolve the CORS policy, refusing the unsafe wildcard-plus-credentials combo.

    `Access-Control-Allow-Origin: *` together with credentials is rejected by
    browsers anyway, and it lets any site call the API with the caller's
    cookies. Wildcard is therefore downgraded to credential-less, and is
    refused outright in production.
    """
    origins = settings.cors_origins
    if "*" not in origins:
        return origins, True

    if settings.is_production:
        raise RuntimeError(
            "ALLOWED_ORIGINS='*' is not permitted in production — "
            "set it to your frontend origin(s), e.g. https://your-app.vercel.app"
        )

    logger.warning(
        "ALLOWED_ORIGINS is '*' — allowing any origin without credentials. "
        "Set explicit origins before deploying."
    )
    return ["*"], False


_allowed_origins, _allow_credentials = _cors_policy()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(query_router.router, prefix="/api/v1", tags=["Query"])
app.include_router(ingest_router.router, prefix="/api/v1", tags=["Ingest"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    qdrant_ok = await check_connection()
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        environment=settings.environment,
        qdrant_connected=qdrant_ok,
    )


@app.get("/", tags=["Root"])
async def root():
    return JSONResponse(
        {"message": "Production-Grade RAG API", "docs": "/docs", "health": "/health"}
    )
