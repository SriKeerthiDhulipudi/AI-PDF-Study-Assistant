"""Helper script to generate a sample multi-page PDF for testing and demonstration."""

import fitz  # PyMuPDF


def generate_sample_pdf(output_path: str = "sample_document.pdf"):
    doc = fitz.open()

    # --- Page 1: Introduction ---
    p1 = doc.new_page()
    text_p1 = """
Artificial Intelligence and Retrieval-Augmented Generation (RAG)
Technical Report & Overview

1. Executive Summary
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances the output of Large
Language Models (LLMs) by referencing an authoritative knowledge base outside of its training data.
Traditional LLMs, while capable of wide-ranging reasoning, are susceptible to hallucinations and
outdated knowledge. RAG addresses this by introducing a retrieval step before generation.

2. Core Components of RAG Systems
A standard RAG pipeline consists of three foundational stages:
- Document Ingestion & Chunking: Dividing long documents into manageable semantic units.
- Vector Embedding & Storage: Transforming text chunks into high-dimensional vectors and storing
  them in indices such as FAISS, Chroma, or Milvus.
- Semantic Retrieval & Synthesis: Matching user queries against vector indices via cosine similarity
  or L2 distance, then injecting retrieved passages into the LLM prompt.
"""
    p1.insert_text((50, 70), text_p1.strip(), fontsize=11)

    # --- Page 2: Vector Search & Chunking ---
    p2 = doc.new_page()
    text_p2 = """
3. Vector Databases and Similarity Search
Dense vector embeddings map words, sentences, or paragraphs into a continuous vector space where
semantically similar passages reside close to one another.

- FAISS (Facebook AI Similarity Search):
  FAISS is an open-source library optimized for fast nearest-neighbor search in dense vector spaces.
  It supports exact L2 distance search (IndexFlatL2) and approximate nearest neighbor (ANN) methods
  such as Inverted File Indexing (IVF) and Product Quantization (PQ).

- Chunking Strategies:
  Fixed-size chunking with overlap (e.g., 1000 characters with 150-character overlap) ensures that
  sentence boundaries and conceptual continuity are preserved across chunk transitions.
"""
    p2.insert_text((50, 70), text_p2.strip(), fontsize=11)

    # --- Page 3: Evaluation & Challenges ---
    p3 = doc.new_page()
    text_p3 = """
4. Key Challenges and Evaluation Metrics
Deploying RAG in mission-critical applications requires rigorous evaluation across three main axes:

- Faithfulness: Does the generated answer strictly adhere to the retrieved context without hallucinating?
- Answer Relevance: Does the synthesized answer directly address the user's specific query?
- Context Precision: Are the retrieved document chunks pertinent, or do they include distracting noise?

5. Conclusion and Future Directions
Hybrid retrieval combining dense semantic embeddings with sparse BM25 keyword matching (reciprocal rank fusion)
and agentic self-correcting RAG represent the cutting-edge frontiers in document intelligence.
"""
    p3.insert_text((50, 70), text_p3.strip(), fontsize=11)

    doc.save(output_path)
    doc.close()
    print(f"Sample PDF successfully generated at: {output_path}")


if __name__ == "__main__":
    generate_sample_pdf()
