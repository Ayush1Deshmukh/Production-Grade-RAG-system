# Stage 1: Builder — dependencies and model weights
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# A venv keeps the full dependency tree in one directory that the runtime
# stage can copy wholesale, without dragging along build-essential.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# torch comes from the CPU wheel index: the default PyPI wheel bundles ~2GB of
# CUDA libraries that never get used here. torchvision and torchaudio are not
# installed at all — only text models are loaded.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Bake the embedding and re-ranker weights into the image (~180MB). Downloading
# them on first use instead means every cold start waits on huggingface.co, and
# a hub outage turns into a broken boot rather than a slow one.
ENV HF_HOME=/opt/hf-cache
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

RUN useradd --create-home appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf-cache /opt/hf-cache

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/ ./app/
COPY prompts/ ./prompts/
# Needed to seed the vector store from inside the container/Space.
COPY scripts/ ./scripts/

# The cache must stay writable: sentence-transformers takes lock files inside
# HF_HOME even when every weight it needs is already present.
RUN mkdir -p /app/data/corpus && \
    chown -R appuser:appuser /app /opt/hf-cache

USER appuser

EXPOSE 7860

# Generous start period: boot loads both models from disk and rebuilds the BM25
# index by scrolling the Qdrant collection before the app reports healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
