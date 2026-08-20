"""
app/models.py
─────────────
All Pydantic request/response schemas for the API.
Citation enforcement lives here: the LLM MUST populate `citations`
and MUST use `refusal` if the context doesn't contain an answer.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Body for POST /ingest (URL-based ingestion)."""
    urls: List[str] = Field(..., min_length=1, description="Web URLs to ingest")
    collection_name: Optional[str] = Field(None, description="Override collection name")


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    collection_name: str
    message: str


# ── Query (LLM output schema — enforced via with_structured_output) ───────────

class RAGResponse(BaseModel):
    """
    Structured output the LLM must produce.
    `citations` maps each claim to a chunk ID from the retrieved context.
    `refusal` must be populated (and `answer` left empty) when the
    retrieved context does not contain sufficient information.
    """
    answer: str = Field(
        description=(
            "The answer grounded strictly in the provided context. "
            "Leave empty if context is insufficient."
        ),
    )
    citations: List[str] = Field(
        description="List of CHUNK_IDs from the context that support the answer.",
    )
    refusal: Optional[str] = Field(
        default=None,
        description=(
            "Populated ONLY when the context does not contain the answer. "
            "Example: 'The provided documents do not contain information about X.' "
            "Must be a string or null. DO NOT output a boolean."
        ),
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence score between 0 and 1.",
    )

    @field_validator("refusal", mode="before")
    @classmethod
    def cast_boolean_refusal(cls, v: Any) -> Optional[str]:
        """
        Tolerate a boolean refusal instead of failing the whole response.

        The prompt asks for a sentence, but models occasionally answer the
        *question* ("did you refuse?") rather than filling the field. This text
        is rendered to the user verbatim, so the fallback has to say something
        useful — "Refused" on its own reads like an error page.
        """
        if isinstance(v, bool):
            return (
                "The retrieved documents do not contain enough information to "
                "answer this question."
            ) if v else None
        return v


# ── Chunk metadata returned alongside the answer ──────────────────────────────

class ChunkMetadata(BaseModel):
    chunk_id: str
    source: str
    content_preview: str = Field(..., description="First 200 chars of the chunk")
    score: float = Field(default=0.0, description="Relevance score after re-ranking")
    retrieval_modality: str = Field(default="Hybrid (BM25 + Dense)", description="Method of retrieval")


# ── Full API response ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = Field(None, description="For multi-turn trace grouping")
    collection_name: Optional[str] = Field(None)


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    sources: List[str]
    refusal: Optional[str] = None
    confidence: float
    retrieved_chunks: List[ChunkMetadata]
    latency_ms: float
    latency_breakdown: dict = Field(default_factory=dict)
    langfuse_url: Optional[str] = None
    prompt_version: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    environment: str
    qdrant_connected: bool
    version: str = "1.0.0"
