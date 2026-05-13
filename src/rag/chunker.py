"""Document chunking for the Finnie knowledge base."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Iterator, List, Optional

import yaml

from src.rag.schemas import ArticleMetadata, Chunk
from src.core.config import settings


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from article body. Returns (meta_dict, body)."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                return meta, parts[2].strip()
            except yaml.YAMLError:
                pass
    return {}, content.strip()


def _count_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (no tiktoken dependency at runtime)."""
    return max(1, len(text) // 4)


def _split_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks, trying to break on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last sentences that fit within overlap budget
            overlap_tokens = 0
            overlap_sents: List[str] = []
            for s in reversed(current):
                t = _count_tokens(s)
                if overlap_tokens + t <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                else:
                    break
            current = overlap_sents
            current_tokens = overlap_tokens
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


class ArticleChunker:
    """Chunks markdown articles with metadata preservation."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self._chunk_size = chunk_size or settings.rag.chunk_size
        self._overlap = chunk_overlap or settings.rag.chunk_overlap

    def chunk_file(self, path: Path) -> List[Chunk]:
        """Load a markdown article file and return its chunks."""
        content = path.read_text(encoding="utf-8")
        meta_dict, body = _parse_frontmatter(content)

        metadata = ArticleMetadata(
            title=meta_dict.get("title", path.stem.replace("_", " ").title()),
            category=meta_dict.get("category", path.parent.name),
            tags=meta_dict.get("tags", []),
            difficulty=meta_dict.get("difficulty", "intermediate"),
            source=meta_dict.get("source", "Finnie Educational Content"),
            file_path=str(path),
        )
        return self._chunk_text(body, metadata)

    def chunk_text(self, text: str, metadata: ArticleMetadata) -> List[Chunk]:
        return self._chunk_text(text, metadata)

    def _chunk_text(self, text: str, metadata: ArticleMetadata) -> List[Chunk]:
        # Split on markdown headers first, then by sentence within sections
        sections = re.split(r"\n(?=#{1,3} )", text)
        all_chunks: List[str] = []
        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue
            if _count_tokens(stripped) <= self._chunk_size:
                all_chunks.append(stripped)
            else:
                all_chunks.extend(_split_by_sentences(stripped, self._chunk_size, self._overlap))

        slug = Path(metadata.file_path).stem if metadata.file_path else metadata.title.lower().replace(" ", "_")
        chunks: List[Chunk] = []
        for i, chunk_text in enumerate(all_chunks):
            if not chunk_text.strip():
                continue
            chunks.append(Chunk(
                text=chunk_text.strip(),
                chunk_id=f"{slug}_{i}",
                article_metadata=metadata,
                token_count=_count_tokens(chunk_text),
            ))
        return chunks

    def chunk_directory(self, directory: Path) -> List[Chunk]:
        """Chunk all .md files in a directory (recursive)."""
        all_chunks: List[Chunk] = []
        for md_file in sorted(directory.rglob("*.md")):
            try:
                all_chunks.extend(self.chunk_file(md_file))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Failed to chunk %s: %s", md_file, exc)
        return all_chunks
