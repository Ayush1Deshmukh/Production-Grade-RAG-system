import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.retrieval import hybrid_retriever
from app.retrieval.hybrid_retriever import (
    get_hybrid_retriever,
    initialize_bm25_from_docs,
    reset_bm25_index,
)


@pytest.fixture(autouse=True)
def _clean_bm25_index():
    """BM25 lives in a module-level cache — reset it around every test."""
    reset_bm25_index()
    yield
    reset_bm25_index()


def _mock_vector_store():
    """A vector store whose retriever passes EnsembleRetriever's Runnable check."""
    mock_store = MagicMock()
    # spec=BaseRetriever makes isinstance() succeed, which Pydantic requires.
    mock_store.as_retriever.return_value = MagicMock(spec=BaseRetriever)
    return mock_store


@pytest.mark.asyncio
@patch("app.retrieval.hybrid_retriever.get_vector_store")
async def test_hybrid_retriever_initialization(mock_get_vector_store):
    mock_get_vector_store.return_value = _mock_vector_store()

    await initialize_bm25_from_docs([Document(page_content="test")])

    retriever = await get_hybrid_retriever()

    assert retriever is not None
    assert len(retriever.retrievers) == 2
    assert retriever.weights == [0.5, 0.5]


@pytest.mark.asyncio
@patch("app.retrieval.hybrid_retriever.get_vector_store")
async def test_falls_back_to_dense_when_bm25_missing(mock_get_vector_store):
    """Without an ingest, BM25 is empty and retrieval must still work (dense only)."""
    mock_get_vector_store.return_value = _mock_vector_store()

    retriever = await get_hybrid_retriever()

    assert retriever is not None
    assert not hasattr(retriever, "retrievers")
    assert hybrid_retriever.is_bm25_ready() is False


@pytest.mark.asyncio
async def test_bm25_accumulates_across_ingests():
    """A second ingest must not discard documents from the first one."""
    await initialize_bm25_from_docs([Document(page_content="alpha document")])
    await initialize_bm25_from_docs([Document(page_content="beta document")])

    assert hybrid_retriever.is_bm25_ready() is True
    assert len(hybrid_retriever._bm25_documents) == 2
