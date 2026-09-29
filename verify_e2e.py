"""End-to-end verification script testing the RAG pipeline with real PDF and embeddings."""

import sys
from src.document_loader import extract_text_from_pdf, chunk_documents
from src.vector_store import get_embedding_model, build_faiss_index, retrieve_relevant_chunks
from src.rag_chain import get_llm, generate_rag_answer


def main():
    print("=== 1. Loading and Chunking PDF ===")
    pdf_path = "sample_document.pdf"
    docs, stats = extract_text_from_pdf(pdf_path, filename=pdf_path)
    print(f"Loaded {len(docs)} pages, total characters: {stats['total_characters']}")
    assert len(docs) == 3, f"Expected 3 pages, got {len(docs)}"

    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")
    assert len(chunks) >= 3, "Expected at least 3 chunks"

    print("\n=== 2. Generating Local Embeddings & FAISS Index ===")
    embedding_model = get_embedding_model()
    vector_store = build_faiss_index(chunks, embedding_model)
    print("FAISS index built successfully.")

    print("\n=== 3. Testing Semantic Retrieval ===")
    query_1 = "What are the core components or stages of a standard RAG system?"
    results_1 = retrieve_relevant_chunks(vector_store, query_1, top_k=2)
    print(f"Query 1: '{query_1}'")
    print(f"Retrieved {len(results_1)} chunks.")
    top_doc_1, score_1 = results_1[0]
    print(f"Top result from Page {top_doc_1.metadata['page']}, Score: {score_1:.4f}")
    assert "Document Ingestion & Chunking" in top_doc_1.page_content or "Core Components" in top_doc_1.page_content, \
        "Failed to retrieve core RAG components chunk!"
    print("[OK] Retrieval 1 verified successfully.")

    query_2 = "How does FAISS perform fast nearest-neighbor search?"
    results_2 = retrieve_relevant_chunks(vector_store, query_2, top_k=2)
    print(f"\nQuery 2: '{query_2}'")
    top_doc_2, score_2 = results_2[0]
    print(f"Top result from Page {top_doc_2.metadata['page']}, Score: {score_2:.4f}")
    assert "FAISS" in top_doc_2.page_content, "Failed to retrieve FAISS chunk!"
    assert top_doc_2.metadata["page"] == 2, f"Expected Page 2, got {top_doc_2.metadata['page']}"
    print("[OK] Retrieval 2 verified successfully.")

    print("\n=== 4. Testing RAG Answer Generation (Mock / Grounded) ===")
    mock_llm = get_llm("Mock / Offline", model_name="mock-model")
    ans_result = generate_rag_answer(mock_llm, query_1, results_1)
    print(f"Answer: {ans_result['answer'][:120]}...")
    print(f"Sources attached: {len(ans_result['sources'])}")
    assert len(ans_result["sources"]) == 2
    assert ans_result["sources"][0]["page"] == top_doc_1.metadata["page"]
    print("[OK] RAG synthesis verified successfully.")

    print("\n[SUCCESS] ALL END-TO-END PIPELINE CHECKS PASSED!")


if __name__ == "__main__":
    main()
