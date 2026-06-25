# pipelines/models.py
from pydantic import BaseModel, HttpUrl
from typing import Optional

class RawDocument(BaseModel):
    """One document unit — either a Notion page or a crawled resource."""
    id: str
    source: str                  # "notion" | "crawled"
    url: Optional[str] = None
    title: str
    content: str                 # Raw markdown text
    metadata: dict = {}
    image_urls: list[str] = []
    pdf_urls: list[str] = []
    # Optional graph-RAG fields (skip for now, keep door open)
    parent_id: Optional[str] = None
    child_urls: list[str] = []

class Chunk(BaseModel):
    """Processed, embedded-ready chunk."""
    id: str
    document_id: str
    content: str
    context_prefix: str = ""     # Contextual retrieval summary
    embedding: list[float] = []
    metadata: dict = {}
    quality_score: float = 0.0