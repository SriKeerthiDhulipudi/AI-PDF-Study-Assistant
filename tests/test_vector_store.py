"""Unit tests for FAISS vector indexing and retrieval."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from src.vector_store import build_faiss_index, retrieve_relevant_chunks


@pytest.fixture
def fake_embeddings():
    """Provides deterministic fake embeddings with dimension 10 for fast testing."""
    return FakeEmbeddings(size=10)


@pytest.fixture
def sample_chunks():
    """Provides sample document chunks."""
    return [
        Document(
            page_content="Neural networks are inspired by the biological nervous systems.",
            metadata={"page": 1, "source": "ai_book.pdf", "chunk_id": 1}
        ),
        Document(
            page_content="Convolutional neural networks excel at computer vision tasks.",
            metadata={"page": 2, "source": "ai_book.pdf", "chunk_id": 2}
        ),
        Document(
            page_content="Recurrent neural networks process sequential data and time series.",
            metadata={"page": 3, "source": "ai_book.pdf", "chunk_id": 3}
        ),
    ]


def test_build_faiss_index_and_retrieve(fake_embeddings, sample_chunks):
    """Tests indexing chunks in FAISS and querying for nearest neighbors."""
    vector_store = build_faiss_index(sample_chunks, fake_embeddings)
    assert vector_store is not None

    # Retrieve top 2
    results = retrieve_relevant_chunks(vector_store, "convolutional", top_k=2)
    assert len(results) == 2
    
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert "page" in doc.metadata
    assert score is not None


def test_empty_query_retrieval(fake_embeddings, sample_chunks):
    """Tests retrieval with empty string query returns empty list."""
    vector_store = build_faiss_index(sample_chunks, fake_embeddings)
    results = retrieve_relevant_chunks(vector_store, "   ")
    assert results == []


def test_empty_chunks_error(fake_embeddings):
    """Tests that building vector store with no chunks raises ValueError."""
    with pytest.raises(ValueError):
        build_faiss_index([], fake_embeddings)
