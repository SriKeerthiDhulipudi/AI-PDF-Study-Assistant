"""Unit tests for document loader and text chunker."""

import fitz
import pytest
from langchain_core.documents import Document

from src.document_loader import chunk_documents, extract_text_from_pdf


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Creates an in-memory sample 2-page PDF document for testing."""
    doc = fitz.open()
    
    # Page 1
    p1 = doc.new_page()
    p1.insert_text(
        (50, 72),
        "Artificial Intelligence in Healthcare.\n"
        "AI applications in medicine include medical image analysis, "
        "predictive diagnostics, and robotic surgery.",
        fontsize=12
    )
    
    # Page 2
    p2 = doc.new_page()
    p2.insert_text(
        (50, 72),
        "Challenges and Ethical Considerations.\n"
        "Data privacy, algorithmic bias, and clinical validation remain "
        "critical obstacles in medical AI deployment.",
        fontsize=12
    )
    
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_text_from_pdf(sample_pdf_bytes):
    """Verifies that PDF pages and metadata are correctly extracted."""
    docs, stats = extract_text_from_pdf(sample_pdf_bytes, filename="test_health.pdf")
    
    assert stats["total_pages"] == 2
    assert stats["pages_with_text"] == 2
    assert stats["filename"] == "test_health.pdf"
    assert len(docs) == 2
    
    # Check page 1 metadata and content
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["source"] == "test_health.pdf"
    assert "Healthcare" in docs[0].page_content
    
    # Check page 2 metadata and content
    assert docs[1].metadata["page"] == 2
    assert "Ethical Considerations" in docs[1].page_content


def test_chunk_documents(sample_pdf_bytes):
    """Verifies document chunking logic and metadata propagation."""
    docs, _ = extract_text_from_pdf(sample_pdf_bytes, filename="test_health.pdf")
    
    # Test with small chunk size to force splitting
    chunks = chunk_documents(docs, chunk_size=60, chunk_overlap=15)
    
    assert len(chunks) > len(docs)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == i + 1
        assert "page" in chunk.metadata
        assert "source" in chunk.metadata
        assert len(chunk.page_content) <= 100  # allowing leeway for word boundaries


def test_chunk_empty_documents():
    """Verifies graceful handling of empty document lists."""
    assert chunk_documents([]) == []
