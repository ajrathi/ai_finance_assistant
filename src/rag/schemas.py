"""Data schemas for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ArticleMetadata:
    title: str
    category: str
    tags: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"   # beginner | intermediate | advanced
    source: str = "Finnie Educational Content"
    file_path: str = ""


@dataclass
class Chunk:
    text: str
    chunk_id: str                        # "{article_slug}_{chunk_index}"
    article_metadata: ArticleMetadata
    token_count: int = 0
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float                        # cosine similarity (higher = more relevant)

    @property
    def title(self) -> str:
        return self.chunk.article_metadata.title

    @property
    def category(self) -> str:
        return self.chunk.article_metadata.category


@dataclass
class RAGContext:
    """Output of the RAG pipeline passed to agents."""
    context_text: str                   # Formatted string for LLM prompt injection
    sources: List[Any]                  # List[SourceAttribution] (avoid circular import)
    results: List[RetrievalResult] = field(default_factory=list)
