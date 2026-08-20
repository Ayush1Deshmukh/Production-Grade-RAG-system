"""
app/config.py
─────────────
Centralised settings loaded from environment variables / .env file.
Pydantic-Settings validates types and raises clear errors on startup
if any required key is missing — no silent misconfigurations.
"""

import logging
from functools import lru_cache
from typing import Dict, List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# Generation backends, all speaking the OpenAI-compatible protocol — switching
# is a base_url and key swap, nothing more. Kept configurable on purpose: this
# project has already lost one backend to a retired free tier, and the next one
# should cost an env-var change rather than a code change.
LLM_ENDPOINTS: Dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
}

# The same open weights are published under different ids per provider: Groq
# namespaces them ("openai/gpt-oss-120b"), Cerebras does not ("gpt-oss-120b").
# An .env carried over from the other backend would otherwise 404 at the first
# question, so the id is translated to the selected provider's spelling.
MODEL_ALIASES: Dict[str, Dict[str, str]] = {
    "groq": {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
    },
    "cerebras": {
        "openai/gpt-oss-120b": "gpt-oss-120b",
        "openai/gpt-oss-20b": "gpt-oss-20b",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM (generation) ──────────────────────────────────────────────────────
    # Only the selected provider's key is required; see `provider_key_present`.
    llm_provider: str = Field("groq", description=f"Generation backend: one of {sorted(LLM_ENDPOINTS)}")
    llm_model: str = Field(
        "openai/gpt-oss-120b",
        description=(
            "Model id exactly as the chosen provider names it. The same model is "
            "'openai/gpt-oss-120b' on Groq and 'gpt-oss-120b' on Cerebras."
        ),
    )

    groq_api_key: str = Field("", description="Required when llm_provider='groq'")
    cerebras_api_key: str = Field("", description="Required when llm_provider='cerebras'")

    # ── Unused provider credentials (kept so existing .env files still load) ──
    # Nothing in the codebase reads these: embeddings and re-ranking run locally
    # on HuggingFace models. They are optional so no deploy or CI job has to
    # invent dummy values.
    gemini_api_key: str = Field("", description="Unused; kept for backwards compatibility")
    cohere_api_key: str = Field("", description="Unused; kept for backwards compatibility")
    embedding_model: str = Field("embed-english-v3.0", description="Unused; see retrieval/embedder.py")
    rerank_model: str = Field("rerank-english-v3.0", description="Unused; see retrieval/reranker.py")

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

    # ── Ingest protection ─────────────────────────────────────────────────────
    # /ingest makes the server fetch arbitrary URLs and write them into the
    # vector store, so it is disabled until a key is configured.
    ingest_api_key: str = Field(
        "", description="Shared secret required in the X-API-Key header on /ingest"
    )
    ingest_allowed_domains: str = Field(
        "",
        description=(
            "Comma-separated hostname suffixes /ingest may fetch "
            "(e.g. 'python.langchain.com,docs.qdrant.tech'). Empty = any public host."
        ),
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 700)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("llm_provider")
    @classmethod
    def known_provider(cls, v: str) -> str:
        provider = v.strip().lower()
        if provider not in LLM_ENDPOINTS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(LLM_ENDPOINTS)}, got {v!r}"
            )
        return provider

    @model_validator(mode="after")
    def normalise_model_for_provider(self) -> "Settings":
        """Translate a known model id into the selected provider's spelling."""
        renamed = MODEL_ALIASES.get(self.llm_provider, {}).get(self.llm_model)
        if renamed:
            logger.warning(
                "LLM_MODEL=%r is the id used by another provider; using %r for %s.",
                self.llm_model, renamed, self.llm_provider,
            )
            self.llm_model = renamed
        return self

    @model_validator(mode="after")
    def provider_key_present(self) -> "Settings":
        """
        Require a key for the *selected* provider only.

        Demanding all of them would force every deploy and CI job to invent
        dummy values for backends it does not use; demanding none would defer
        the failure to the first user question instead of to startup.
        """
        if not self.llm_api_key:
            raise ValueError(
                f"{self.llm_provider.upper()}_API_KEY is required when "
                f"LLM_PROVIDER={self.llm_provider!r}"
            )
        return self

    @property
    def llm_base_url(self) -> str:
        return LLM_ENDPOINTS[self.llm_provider]

    @property
    def llm_api_key(self) -> str:
        return {
            "groq": self.groq_api_key,
            "cerebras": self.cerebras_api_key,
        }[self.llm_provider]

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def ingest_enabled(self) -> bool:
        """/ingest stays closed until a shared secret is configured."""
        return bool(self.ingest_api_key)

    @property
    def ingest_domain_allowlist(self) -> List[str]:
        return [d.strip().lower() for d in self.ingest_allowed_domains.split(",") if d.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
