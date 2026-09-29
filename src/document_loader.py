"""Document loader and text chunking utilities using PyMuPDF and LangChain."""

import io
from typing import Dict, List, Optional, Tuple, Union
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def extract_text_from_pdf(
    pdf_source: Union[str, bytes, io.BytesIO],
    filename: str = "document.pdf"
) -> Tuple[List[Document], Dict[str, Union[int, str]]]:
    """
    Extracts text page-by-page from a PDF file path or byte stream.
    
    Args:
        pdf_source: File path (str), raw bytes (bytes), or BytesIO stream.
        filename: Name of the uploaded PDF file.
        
    Returns:
        Tuple containing:
        - List of LangChain Document objects (one per page with text).
        - Dictionary of extraction statistics (page_count, total_characters, filename).
    """
    if isinstance(pdf_source, str):
        doc = fitz.open(pdf_source)
    elif isinstance(pdf_source, bytes):
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    elif hasattr(pdf_source, "read"):
        doc = fitz.open(stream=pdf_source.read(), filetype="pdf")
    else:
        raise ValueError("Unsupported PDF source type. Provide a file path, bytes, or BytesIO.")

    documents: List[Document] = []
    total_chars = 0
    total_pages = len(doc)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text("text")
        
        # Clean text
        cleaned_text = text.strip()
        if cleaned_text:
            total_chars += len(cleaned_text)
            doc_obj = Document(
                page_content=cleaned_text,
                metadata={
                    "source": filename,
                    "page": page_idx + 1,
                    "total_pages": total_pages,
                }
            )
            documents.append(doc_obj)

    doc.close()

    stats = {
        "filename": filename,
        "total_pages": total_pages,
        "pages_with_text": len(documents),
        "total_characters": total_chars,
    }

    return documents, stats


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits page-level documents into smaller semantic chunks with overlapping boundaries.
    
    Args:
        documents: List of page-level Document objects.
        chunk_size: Maximum character length for each chunk.
        chunk_overlap: Overlapping character count between consecutive chunks.
        
    Returns:
        List of chunked Document objects with preserved and enriched metadata.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # Enrich metadata with unique chunk identifier
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx + 1

    return chunks
