"""
scripts/seed.py
───────────────
Standalone script to seed the Vector DB.
Loads comprehensive LangChain + system documentation to cover all
20 golden dataset evaluation questions.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path so we can import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.ingestion.loader import load_urls, load_directory
from app.ingestion.ingest import ingest_pipeline


URLS = [
    # ── LangChain Core Concepts ──────────────────────────────────────────────
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/lcel/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/document_loaders/",
    "https://python.langchain.com/docs/concepts/structured_outputs/",
    # ── How-to Guides ────────────────────────────────────────────────────────
    "https://python.langchain.com/docs/how_to/ensemble_retriever/",
    "https://python.langchain.com/docs/how_to/contextual_compression/",
    "https://python.langchain.com/docs/how_to/vectorstore_retriever/",
    "https://python.langchain.com/docs/how_to/recursive_text_splitter/",
    "https://python.langchain.com/docs/how_to/structured_output/",
    "https://python.langchain.com/docs/how_to/chat_model_caching/",
    # ── RAG Tutorials ────────────────────────────────────────────────────────
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/qa_chat_history/",
    # ── Integrations ─────────────────────────────────────────────────────────
    "https://python.langchain.com/docs/integrations/vectorstores/qdrant/",
    "https://python.langchain.com/docs/integrations/retrievers/cohere-reranker/",
]

# System-specific text to inject as synthetic documents
SYSTEM_DOCS_TEXT = """
# RAG System Architecture Documentation

## Embedding Model
This RAG system uses the all-MiniLM-L6-v2 sentence-transformer model from HuggingFace
for generating 384-dimensional dense vector embeddings. This is a local, open-source
model that runs entirely on-device without any API calls. Previously the system used
Cohere's embed-english-v3.0 model which produced 1024-dimensional embeddings.

## Reranking
A local CrossEncoder model (ms-marco-MiniLM-L-6-v2) is used for reranking retrieved
documents. This is wrapped in LangChain's ContextualCompressionRetriever. The cross-encoder
scores each (query, document) pair together for higher-precision relevance scoring.

## LLM (Generative Model)
The RAG system uses gpt-oss-120b as the primary LLM for answer generation, served by
Groq and called through its OpenAI-compatible API. The generation backend is selected
by the LLM_PROVIDER environment variable and both supported backends (Groq and
Cerebras) speak the same OpenAI-compatible protocol, so switching providers requires
no code change. It uses LangChain's with_structured_output in json_mode to bind the
LLM to a Pydantic schema, forcing structured JSON responses and preventing
hallucinated or malformed output.

## Vector Database
Qdrant Cloud is used as the vector database. It provides a free 1GB permanent cluster
and supports both dense semantic vector search and sparse vectors for hybrid search.
In this system the sparse side is handled by LangChain's EnsembleRetriever, which
merges dense and BM25 results using Reciprocal Rank Fusion (RRF).

## Hybrid Search
The system uses LangChain's EnsembleRetriever combining:
- Dense vector search (Qdrant with MiniLM embeddings)
- Sparse BM25 keyword search (rank-bm25)
EnsembleRetriever uses Reciprocal Rank Fusion (RRF): score = 1/(k + rank) where k=60.

## Evaluation with Ragas
The pipeline is evaluated using Ragas with the faithfulness metric. Faithfulness measures
whether every claim in the generated answer is supported by the retrieved context chunks.
A score of 1.0 means all claims are grounded in the context. The LLM judge for evaluation
is gpt-oss-120b, the same model used for inference, chosen for reliable JSON-structured
grading.

## CI/CD Quality Gate
GitHub Actions runs Ragas evaluation on every pull request. The run_evaluation.py script
exits with code 1 if faithfulness < 0.90. GitHub Actions treats a non-zero exit code as
a build failure, blocking the PR from being merged.

## Observability
Langfuse is used as the LLM observability platform. It captures traces via LangChain
callbacks for each RAG request, tracking latency, token usage, retrieval steps, and
LLM responses. Traces include the input query, retrieval results, prompt version, and
LLM response with token counts.

## Text Splitting
The RecursiveCharacterTextSplitter is used with chunk_size=700 and chunk_overlap=100.
It splits text using a hierarchy of separators (paragraphs, sentences, words) recursively.
Chunk overlap of 100 tokens ensures consecutive chunks share content to prevent loss of
context at chunk boundaries.

## Prompt Versioning
Prompts are stored as YAML files in the prompts/ directory and loaded dynamically via
the prompt_registry module. This enables version control and A/B testing of prompts.
"""


async def main():
    print("🌱 Starting Comprehensive Seed Script...")

    all_docs = []

    # 1. Load web URLs
    print(f"📡 Loading {len(URLS)} documentation URLs...")
    try:
        web_docs = load_urls(URLS)
        all_docs.extend(web_docs)
        print(f"   ✅ Loaded {len(web_docs)} web documents")
    except Exception as e:
        print(f"   ⚠️  Web loading partially failed: {e}")

    # 2. Load any local corpus files
    corpus_dir = Path(__file__).parent.parent / "data" / "corpus"
    if corpus_dir.exists() and any(corpus_dir.rglob("*")):
        print(f"📁 Loading local corpus from {corpus_dir}...")
        local_docs = load_directory(corpus_dir)
        all_docs.extend(local_docs)
        print(f"   ✅ Loaded {len(local_docs)} local documents")

    # 3. Inject system-specific synthetic document
    from langchain_core.documents import Document
    system_doc = Document(
        page_content=SYSTEM_DOCS_TEXT,
        metadata={"source": "system_architecture.md", "file_type": ".md"}
    )
    all_docs.append(system_doc)
    print("   ✅ Injected system architecture document")

    if not all_docs:
        print("❌ No documents loaded. Aborting.")
        return

    print(f"\n📊 Total documents loaded: {len(all_docs)}")

    # 4. Chunk, index in BM25, and upsert to Qdrant.
    #    ingest_pipeline builds the BM25 index from the *chunks*, so sparse hits
    #    carry the same chunk_id metadata the citations rely on.
    print("⬆️  Chunking and upserting to Qdrant with 384-dim MiniLM embeddings...")
    chunks_added = await ingest_pipeline(all_docs)

    print(f"\n✅ Seeding Complete! {chunks_added} chunks added to Qdrant.")
    print("   Collection is now ready for evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
