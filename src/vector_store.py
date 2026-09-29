"""Vector store management and embedding indexing using FAISS and LangChain."""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K


def get_embedding_model(
    provider: str = "HuggingFace (Local)",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> Any:
    """
    Factory to instantiate embedding models. Defaults to local zero-cost HuggingFace embeddings.
    
    Args:
        provider: Embedding provider ("HuggingFace (Local)", "Google Gemini", or "OpenAI").
        model_name: Optional custom embedding model identifier.
        api_key: API key for cloud providers.
        
    Returns:
        Embeddings object conforming to LangChain Embeddings interface.
    """
    if provider == "Google Gemini" or provider == "Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model_name or "models/text-embedding-004",
            google_api_key=api_key
        )
    elif provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model_name or "text-embedding-3-small",
            openai_api_key=api_key
        )
    else:
        # Default: HuggingFace sentence-transformers running locally on CPU
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model_name or DEFAULT_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model_name or DEFAULT_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )


def build_faiss_index(
    chunks: List[Document],
    embeddings: Any
) -> FAISS:
    """
    Creates an in-memory FAISS vector index from document chunks.
    
    Args:
        chunks: List of chunked Document objects.
        embeddings: Embeddings generator instance.
        
    Returns:
        FAISS vector store populated with the chunks.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunk list.")
    
    return FAISS.from_documents(documents=chunks, embedding=embeddings)


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = DEFAULT_TOP_K
) -> List[Tuple[Document, float]]:
    """
    Retrieves the top-k most semantically relevant chunks for a user query,
    along with their similarity scores.
    
    Args:
        vector_store: Instantiated FAISS vector store.
        query: User question or search query.
        top_k: Number of chunks to retrieve.
        
    Returns:
        List of (Document, score) tuples.
    """
    if not query.strip():
        return []

    # similarity_search_with_score returns (doc, L2 distance score)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results
