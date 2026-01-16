"""Content extraction functions for JSON output."""

from .extractor import extract_document_content
from .metadata import extract_document_metadata

__all__ = [
    "extract_document_content",
    "extract_document_metadata",
]
