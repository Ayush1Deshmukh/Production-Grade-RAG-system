"""
app/config.py
─────────────
Centralised settings loaded from environment variables / .env file.
Pydantic-Settings validates types and raises clear errors on startup
if any required key is missing — no silent misconfigurations.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (Generative) ──────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    llm_model: str = Field("llama-3.3-70b", description="Model name for inference")

    # ── Groq (fallback inference / evaluation) ────────────────────────────────
    groq_api_key: str = Field("", description="Groq API key (optional fallback)")

    # ── Cerebras (primary fast inference) ────────────────────────────────────
    cerebras_api_key: str = Field(..., description="Cerebras API key")

    # ── Cohere (embeddings + rerank) ──────────────────────────────────────────
    cohere_api_key: str = Field(..., description="Cohere API key")
    embedding_model: str = Field("embed-english-v3.0")
    rerank_model: str = Field("rerank-english-v3.0")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(..., description="Qdrant Cloud cluster URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")
    qdrant_collection_name: str = Field("rag_documents")

    # ── Langfuse ──────────────────────────────────────────────────────────────
    langfuse_public_key: str = Field("", description="Optional: Langfuse public key")
    langfuse_secret_key: str = Field("", description="Optional: Langfuse secret key")
    langfuse_host: str = Field("https://cloud.langfuse.com")

    # ── RAG Hyperparameters ───────────────────────────────────────────────────
    top_k: int = Field(10, ge=1, le=50, description="Initial retrieval count")
    rerank_top_n: int = Field(5, ge=1, le=20, description="Chunks after re-ranking")
    chunk_size: int = Field(700, ge=100, le=2000)
    chunk_overlap: int = Field(100, ge=0, le=500)
    prompt_version: str = Field("v1", description="Prompt YAML version key")

    # ── App ───────────────────────────────────────────────────────────────────
    environment: str = Field("development")
    log_level: str = Field("INFO")
    allowed_origins: str = Field("http://localhost:3000")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 700)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
