"""Unit tests for RAG chain orchestration and prompt formatting."""

import pytest
from langchain_core.documents import Document

from src.rag_chain import format_context_for_prompt, generate_rag_answer, get_llm


def test_format_context_for_prompt():
    """Verifies that chunks are formatted with proper citations."""
    chunks = [
        (
            Document(
                page_content="Photosynthesis converts sunlight into chemical energy.",
                metadata={"page": 5, "source": "biology.pdf", "chunk_id": 1}
            ),
            0.12
        ),
        (
            Document(
                page_content="Chlorophyll is the primary pigment involved in photosynthesis.",
                metadata={"page": 6, "source": "biology.pdf", "chunk_id": 2}
            ),
            0.25
        ),
    ]

    formatted = format_context_for_prompt(chunks)
    assert "[Excerpt 1 | Source: biology.pdf | Page: 5]" in formatted
    assert "Photosynthesis converts sunlight" in formatted
    assert "[Excerpt 2 | Source: biology.pdf | Page: 6]" in formatted
    assert "Chlorophyll is the primary pigment" in formatted


def test_format_context_empty():
    """Verifies fallback when no chunks are passed."""
    formatted = format_context_for_prompt([])
    assert "No relevant context found" in formatted


def test_generate_rag_answer_mock():
    """Verifies answer generation with Mock LLM."""
    mock_llm = get_llm("Mock / Offline", model_name="mock-model")
    chunks = [
        (
            Document(
                page_content="Water freezes at 0 degrees Celsius.",
                metadata={"page": 1, "source": "physics.pdf", "chunk_id": 1}
            ),
            0.05
        )
    ]

    result = generate_rag_answer(
        llm=mock_llm,
        question="At what temperature does water freeze?",
        retrieved_chunks=chunks
    )

    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["source"] == "physics.pdf"
    assert result["sources"][0]["score"] == 0.05


def test_get_llm_missing_api_key():
    """Verifies that missing API keys raise ValueError for cloud providers."""
    with pytest.raises(ValueError):
        get_llm("Google Gemini", model_name="gemini-2.0-flash", api_key="")

    with pytest.raises(ValueError):
        get_llm("Groq", model_name="llama-3.3-70b-versatile", api_key=None)

    with pytest.raises(ValueError):
        get_llm("OpenAI", model_name="gpt-4o-mini", api_key="")
