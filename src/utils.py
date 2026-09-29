"""Helper utility functions for UI formatting, citations, and validation."""

from typing import Any, Dict, List


def format_file_size(size_bytes: int) -> str:
    """Formats file size from bytes to human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def get_suggested_questions() -> List[str]:
    """Returns sample starter questions for the user to try."""
    return [
        "What is the main topic or objective of this document?",
        "Can you summarize the key findings or takeaways?",
        "What are the major conclusions or recommendations?",
        "Explain any methodologies or processes described here.",
    ]


def format_source_display(source_meta: Dict[str, Any], max_len: int = 300) -> Dict[str, str]:
    """
    Prepares a clean snippet and summary dictionary for UI citation presentation.
    
    Args:
        source_meta: Dictionary containing page, score, text, and chunk_id.
        max_len: Max character length for the preview excerpt.
        
    Returns:
        Formatted dictionary ready for Streamlit expanders.
    """
    raw_text = source_meta.get("text", "").strip()
    preview = (raw_text[:max_len] + "...") if len(raw_text) > max_len else raw_text
    
    page = source_meta.get("page", "Unknown")
    score = source_meta.get("score")
    score_str = f"L2 Distance: {score}" if score is not None else ""
    
    return {
        "title": f"📄 Page {page}" + (f" ({score_str})" if score_str else ""),
        "preview": preview,
        "full_text": raw_text,
        "page": str(page),
    }
