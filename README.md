# AI-Powered PDF Question Answering System (RAG)

An intelligent, production-ready Document Question Answering application built with **LangChain**, **FAISS**, and **Streamlit**. Upload any PDF (research papers, academic notes, manuals, reports), extract and index its contents into a vector database, and ask natural language questions with context-grounded answers and source page citations.

---

## 🌟 Key Features

- **Document Ingestion & Parsing**: Fast and robust page-by-page PDF text extraction using PyMuPDF (`fitz`), preserving page numbers and source metadata.
- **Semantic Text Splitting**: Chunking with LangChain's `RecursiveCharacterTextSplitter` with configurable chunk size and overlap.
- **Dense Vector Search**: Powered by FAISS (Facebook AI Similarity Search) and local `sentence-transformers/all-MiniLM-L6-v2` embeddings for zero-cost, offline semantic indexing.
- **Multi-Provider LLM Integration**:
  - **Google Gemini** (`gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`)
  - **Groq** (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
  - **OpenAI** (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`)
  - **Mock / Offline Mode** for local testing without API keys.
- **Strict Grounding & Anti-Hallucination**: Custom RAG prompt template instructing the model to rely solely on retrieved document excerpts.
- **Source Transparency**: Expandable source citation drawers displaying exact page numbers, similarity distance scores, and text previews.
- **Interactive Streamlit Web UI**: Metrics dashboard (page count, chunk count, file size), quick starter prompt chips, real-time RAG tuning sliders, and chat management.

---

## 🏗️ Architecture

```
User Query ──► Embed Query ──► FAISS Vector Store ──► Top-K Relevant Chunks
                                                              │
Uploaded PDF ──► PyMuPDF ──► Chunker ──► Embedding Model ──┘
                                                              │
                                                        Prompt Template
                                                              │
                                                              ▼
                                                        LLM Provider
                                                (Gemini / Groq / OpenAI)
                                                              │
                                                              ▼
                                                    Answer + Page Citations
```

---

## 📁 Project Structure

```
AI-Powered PDF Question Answering System/
├── app.py                      # Streamlit web application
├── config.py                   # Central settings, default constants, model providers
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment variables
├── README.md                   # Documentation
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # PDF text extraction and chunk splitting
│   ├── vector_store.py         # Embedding model loading and FAISS index
│   ├── rag_chain.py            # RAG orchestration, prompt formatting, LLM dispatch
│   └── utils.py                # Formatting, citations display, and helper functions
└── tests/
    ├── __init__.py
    ├── test_document_loader.py # Tests for PDF parsing and text chunking
    ├── test_vector_store.py    # Tests for FAISS indexing and retrieval
    └── test_rag_chain.py       # Tests for prompt generation and RAG flow
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Clone or Navigate to Project Directory
```powershell
cd "c:\Users\srike\OneDrive\Desktop\AI-Powered PDF Question Answering System"
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. (Optional) Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API key:
```powershell
cp .env.example .env
```
Or directly enter your API key in the Streamlit web sidebar at runtime.

### 5. Run the Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```powershell
pytest tests/ -v
```

---

## 🔮 Future Enhancements
- Multi-PDF document collection querying.
- Persistent conversation memory across multi-turn follow-up questions.
- Multilingual document translation and querying.
- Direct PDF viewer with highlighted passages.
- Cloud deployment (Docker, Streamlit Cloud, AWS/GCP).
