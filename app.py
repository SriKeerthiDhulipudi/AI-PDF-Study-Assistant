"""Streamlit Web Application: AI-Powered PDF Question Answering System (RAG)."""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    MAX_CHUNK_SIZE,
    MAX_TOP_K,
    MIN_CHUNK_SIZE,
    MIN_TOP_K,
    SUPPORTED_PROVIDERS,
)
from src.document_loader import chunk_documents, extract_text_from_pdf
from src.rag_chain import generate_rag_answer, get_llm
from src.utils import format_file_size, format_source_display, get_suggested_questions
from src.vector_store import build_faiss_index, get_embedding_model, retrieve_relevant_chunks


# Streamlit Page Configuration
st.set_page_config(
    page_title="AI-Powered PDF QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #DBEAFE;
        color: #1D4ED8;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
    }
    .citation-card {
        background-color: #F1F5F9;
        border-left: 3px solid #3B82F6;
        padding: 8px 12px;
        margin-top: 6px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ System Control")
    st.caption("PDF Ingestion & RAG Configuration")

    # Document Upload Section
    st.subheader("1. Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a searchable PDF document (academic notes, manuals, reports)."
    )

    # Ingestion Parameters
    with st.expander("🛠️ Text Splitting Settings", expanded=False):
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=MIN_CHUNK_SIZE,
            max_value=MAX_CHUNK_SIZE,
            value=DEFAULT_CHUNK_SIZE,
            step=100,
            help="Length of each text chunk indexed in vector store."
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=DEFAULT_CHUNK_OVERLAP,
            step=25,
            help="Overlap between adjacent chunks to maintain semantic continuity."
        )

    # Process Document if uploaded and not yet processed
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        # Re-index if a new file is uploaded or parameters changed
        if (st.session_state.current_file_name != file_name or 
            st.session_state.vector_store is None):
            with st.spinner("Extracting text and indexing document..."):
                try:
                    # 1. Extract text per page
                    docs, stats = extract_text_from_pdf(file_bytes, filename=file_name)
                    
                    if not docs:
                        st.error("No extractable text found in this PDF. It may be a scanned image or empty.")
                    else:
                        # 2. Chunk documents
                        chunks = chunk_documents(
                            documents=docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        
                        # 3. Load embeddings model
                        if st.session_state.embedding_model is None:
                            st.session_state.embedding_model = get_embedding_model()
                        
                        # 4. Build FAISS index
                        vector_store = build_faiss_index(chunks, st.session_state.embedding_model)
                        
                        # Save state
                        st.session_state.vector_store = vector_store
                        stats["total_chunks"] = len(chunks)
                        stats["file_size"] = len(file_bytes)
                        st.session_state.doc_stats = stats
                        st.session_state.current_file_name = file_name
                        st.session_state.messages = []  # reset chat on new doc
                        st.success(f"Successfully indexed {file_name}!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    # Display Document Metrics if loaded
    if st.session_state.doc_stats:
        st.write("---")
        st.subheader("📊 Document Statistics")
        stats = st.session_state.doc_stats
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats.get('total_pages', 0)}</div>
                <div class="metric-label">Pages</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats.get('total_chunks', 0)}</div>
                <div class="metric-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption(f"📁 **File:** {stats.get('filename')}")
        st.caption(f"💾 **Size:** {format_file_size(stats.get('file_size', 0))}")

    st.write("---")

    # LLM Provider Configuration
    st.subheader("2. AI Model Selection")
    
    provider_options = list(SUPPORTED_PROVIDERS.keys()) + ["Mock / Offline"]
    selected_provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=0,
        help="Select the AI provider to generate answers."
    )

    api_key = ""
    selected_model = ""

    if selected_provider != "Mock / Offline":
        provider_cfg = SUPPORTED_PROVIDERS[selected_provider]
        env_var_name = provider_cfg["env_key"]
        default_env_key = os.getenv(env_var_name, "")
        
        api_key = st.text_input(
            f"{selected_provider} API Key",
            value=default_env_key,
            type="password",
            help=provider_cfg["api_key_help"]
        )

        selected_model = st.selectbox(
            "Model",
            options=provider_cfg["models"],
            index=0
        )
    else:
        st.info("Mock / Offline mode allows testing the RAG pipeline without requiring an external API key.")

    # Retrieval Tuning
    with st.expander("🔍 Retrieval Hyperparameters", expanded=False):
        top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            help="Number of document excerpts passed to the LLM as context."
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Lower is more factual and grounded; higher is more creative."
        )

    st.write("---")

    # Actions
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_btn2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.doc_stats = None
            st.session_state.current_file_name = None
            st.rerun()


# ---------------- MAIN CONTENT ----------------
st.markdown('<div class="badge">RAG Architecture: FAISS + LangChain</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">AI-Powered PDF Question Answering System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload academic notes, technical manuals, or reports and ask natural language questions with grounded source citations.</p>', unsafe_allow_html=True)

# Check if document is ready
if not st.session_state.vector_store:
    # Landing / Onboarding State
    st.info("👈 Please upload a PDF document in the sidebar to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 1. Ingest & Split")
        st.write("Upload any PDF document. The system uses PyMuPDF to extract text and splits it into overlapping semantic chunks, preserving exact page metadata.")
    with col2:
        st.markdown("### 🧠 2. Vector Indexing")
        st.write("Chunks are converted to dense vector embeddings using local sentence-transformers and indexed in an in-memory FAISS database for instant similarity search.")
    with col3:
        st.markdown("### 🎯 3. Grounded Q&A")
        st.write("When you ask a question, the top relevant chunks are retrieved and provided to the LLM to generate precise answers citing page numbers.")

    st.write("---")
    st.markdown("#### 💡 Key Capabilities")
    st.markdown("""
    - **Hallucination Reduction**: Answers are strictly constrained to retrieved context excerpts.
    - **Transparent Source Citations**: Every answer provides an expandable view of source page numbers and matching excerpts.
    - **Provider Flexibility**: Supports Google Gemini, Groq (Llama 3.3), OpenAI (GPT-4o), and local offline embeddings.
    """)

else:
    # Document is indexed and active
    st.success(f"📄 Active Document: **{st.session_state.doc_stats.get('filename')}** ({st.session_state.doc_stats.get('total_pages')} pages, {st.session_state.doc_stats.get('total_chunks')} chunks indexed)")

    # Starter question chips if chat is empty
    if not st.session_state.messages:
        st.write("##### 💡 Suggested Questions to get started:")
        suggested = get_suggested_questions()
        cols = st.columns(len(suggested))
        for i, q in enumerate(suggested):
            if cols[i].button(q, key=f"sugg_{i}", use_container_width=True):
                # Submit question directly
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()

    # Display chat conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display source citations if available for assistant messages
            if msg.get("sources"):
                with st.expander(f"📚 View Retrieved Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for idx, src in enumerate(msg["sources"], 1):
                        display_info = format_source_display(src)
                        st.markdown(f"**{display_info['title']}**")
                        st.markdown(f'<div class="citation-card">{display_info["preview"]}</div>', unsafe_allow_html=True)

    # Chat Input Box
    user_query = st.chat_input("Ask a question about the uploaded document...")

    # Handle query submission (either from input or last appended suggestion)
    pending_query = None
    if user_query:
        pending_query = user_query
        st.session_state.messages.append({"role": "user", "content": pending_query})
        with st.chat_message("user"):
            st.markdown(pending_query)
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # Handle the case where a suggested question was clicked
        if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant":
            pending_query = st.session_state.messages[-1]["content"]

    if pending_query:
        # Check API key validation
        if selected_provider != "Mock / Offline" and not api_key:
            st.warning(f"⚠️ Please enter your {selected_provider} API key in the sidebar to generate AI answers.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching document and generating answer..."):
                    try:
                        # 1. Retrieve top-k context chunks from FAISS
                        retrieved = retrieve_relevant_chunks(
                            vector_store=st.session_state.vector_store,
                            query=pending_query,
                            top_k=top_k
                        )

                        # 2. Instantiate LLM
                        llm = get_llm(
                            provider=selected_provider,
                            model_name=selected_model,
                            api_key=api_key if selected_provider != "Mock / Offline" else None,
                            temperature=temperature
                        )

                        # 3. Generate grounded answer
                        result = generate_rag_answer(
                            llm=llm,
                            question=pending_query,
                            retrieved_chunks=retrieved
                        )

                        # Render answer
                        st.markdown(result["answer"])

                        # Render sources expander
                        if result["sources"]:
                            with st.expander(f"📚 View Retrieved Sources ({len(result['sources'])} chunks)", expanded=False):
                                for src in result["sources"]:
                                    display_info = format_source_display(src)
                                    st.markdown(f"**{display_info['title']}**")
                                    st.markdown(f'<div class="citation-card">{display_info["preview"]}</div>', unsafe_allow_html=True)

                        # Store in session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"]
                        })

                    except Exception as e:
                        st.error(f"Error during RAG execution: {str(e)}")
