"""Configuration settings and defaults for the RAG-based PDF Question Answering System."""

import os
from typing import Dict, List

# Text Splitting Defaults
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 150
MIN_CHUNK_SIZE: int = 200
MAX_CHUNK_SIZE: int = 3000

# Retrieval Defaults
DEFAULT_TOP_K: int = 4
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10

# Embedding Models
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# Supported LLM Providers and their model options
SUPPORTED_PROVIDERS: Dict[str, Dict[str, any]] = {
    "Google Gemini": {
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "default_model": "gemini-2.0-flash",
        "env_key": "GOOGLE_API_KEY",
        "api_key_help": "Enter your Google AI Studio API key.",
    },
    "Groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ],
        "default_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "api_key_help": "Enter your Groq Console API key (free tier available at console.groq.com).",
    },
    "OpenAI": {
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "api_key_help": "Enter your OpenAI API key.",
    },
}

# Default prompt templates
RAG_SYSTEM_PROMPT = """You are an intelligent, precise document question-answering assistant.
Your task is to answer the user's question using ONLY the provided document excerpts (context).

Guidelines:
1. Grounding: Answer strictly using facts and information stated directly in the retrieved context. Do not speculate or invent details.
2. Transparency: Explicitly cite the page numbers where your answers are derived from (e.g., "[Page X]").
3. Honesty: If the retrieved context does not contain enough information to answer the question, clearly state:
   "I am sorry, but the provided document does not contain sufficient information to answer this question."
4. Conciseness: Provide structured, well-explained, and concise responses. Use bullet points or numbered lists where suitable.

Retrieved Context:
-----------------
{context}
-----------------
"""
