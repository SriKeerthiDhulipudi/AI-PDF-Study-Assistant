"""RAG pipeline orchestration, LLM initialization, and prompt formatting."""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config import RAG_SYSTEM_PROMPT, SUPPORTED_PROVIDERS


def get_llm(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.2
) -> Any:
    """
    Instantiates an LLM client based on the chosen provider and API key.
    
    Args:
        provider: Provider name ('Google Gemini', 'Groq', 'OpenAI', 'Mock / Offline').
        model_name: Target model identifier.
        api_key: User-provided API key.
        temperature: Generation temperature (0.0 = deterministic, 1.0 = creative).
        
    Returns:
        Chat model instance conforming to LangChain BaseChatModel interface.
    """
    if provider == "Google Gemini":
        if not api_key:
            raise ValueError("Google Gemini requires a valid Google API key.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )
    elif provider == "Groq":
        if not api_key:
            raise ValueError("Groq requires a valid Groq API key.")
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=model_name,
            groq_api_key=api_key,
            temperature=temperature,
        )
    elif provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI requires a valid OpenAI API key.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature,
        )
    elif provider == "Mock / Offline":
        # Useful for offline testing and verification without API keys
        class MockChatModel:
            def invoke(self, messages):
                class Response:
                    content = (
                        "This is a simulated offline answer based on the retrieved context. "
                        "To generate live AI responses, select Google Gemini, Groq, or OpenAI "
                        "and enter a valid API key."
                    )
                return Response()
        return MockChatModel()
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def format_context_for_prompt(chunks: List[Tuple[Document, float]]) -> str:
    """
    Formats retrieved document chunks into a structured context string with page citations.
    
    Args:
        chunks: List of (Document, score) tuples.
        
    Returns:
        Formatted multi-chunk string with source and page markers.
    """
    if not chunks:
        return "No relevant context found in the document."

    formatted_sections = []
    for i, (doc, score) in enumerate(chunks, 1):
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Document")
        header = f"[Excerpt {i} | Source: {source} | Page: {page}]"
        content = doc.page_content.strip()
        formatted_sections.append(f"{header}\n{content}")

    return "\n\n".join(formatted_sections)


def generate_rag_answer(
    llm: Any,
    question: str,
    retrieved_chunks: List[Tuple[Document, float]],
    custom_system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the RAG generation step by combining context into the prompt and querying the LLM.
    
    Args:
        llm: Chat model instance.
        question: User query string.
        retrieved_chunks: List of (Document, score) tuples from vector search.
        custom_system_prompt: Optional custom system instruction.
        
    Returns:
        Dictionary containing:
        - 'answer': generated text response
        - 'sources': list of chunk metadata and text for citations
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant content was found in the document to answer your question.",
            "sources": []
        }

    formatted_context = format_context_for_prompt(retrieved_chunks)
    system_template = custom_system_prompt or RAG_SYSTEM_PROMPT
    system_text = system_template.format(context=formatted_context)

    messages = [
        SystemMessage(content=system_text),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)

    sources = []
    for doc, score in retrieved_chunks:
        sources.append({
            "page": doc.metadata.get("page", "Unknown"),
            "source": doc.metadata.get("source", "Document"),
            "chunk_id": doc.metadata.get("chunk_id", None),
            "score": round(float(score), 4) if score is not None else None,
            "text": doc.page_content
        })

    return {
        "answer": answer_text,
        "sources": sources
    }
