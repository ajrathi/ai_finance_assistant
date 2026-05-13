"""FAISS index builder, updater, and loader for Finnie's knowledge base."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.rag.schemas import ArticleMetadata, Chunk
from src.core.config import settings

logger = logging.getLogger(__name__)

_INDEX_FILE = "index.faiss"
_META_FILE = "metadata.json"


def _to_numpy(vectors: List[List[float]]) -> np.ndarray:
    arr = np.array(vectors, dtype=np.float32)
    # L2-normalise for cosine similarity via inner product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class FAISSIndexer:
    """Builds and persists a FAISS index from document chunks."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding_model=None,
    ):
        self._index_path = index_path or (Path(settings.project_root) / settings.rag.index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._embedding_model = embedding_model
        self._index = None          # faiss.IndexFlatIP
        self._chunks: List[Chunk] = []

    def _get_embedder(self):
        if self._embedding_model is None:
            from src.core.embeddings import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def _embed_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> np.ndarray:
        embedder = self._get_embedder()
        all_vectors: List[List[float]] = []
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs = embedder.embed_documents(batch)
            all_vectors.extend(vecs)
            logger.debug("Embedded batch %d–%d", i, i + len(batch))
        return _to_numpy(all_vectors)

    def build_index(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and build a new FAISS index from scratch."""
        import faiss
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")
        logger.info("Building FAISS index from %d chunks…", len(chunks))
        vectors = self._embed_chunks(chunks)
        dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._chunks = chunks
        logger.info("Index built: %d vectors, dim=%d", self._index.ntotal, dim)

    def save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss
        if self._index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        faiss.write_index(self._index, str(self._index_path / _INDEX_FILE))
        meta = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "title": c.article_metadata.title,
                "category": c.article_metadata.category,
                "tags": c.article_metadata.tags,
                "difficulty": c.article_metadata.difficulty,
                "source": c.article_metadata.source,
                "file_path": c.article_metadata.file_path,
                "token_count": c.token_count,
            }
            for c in self._chunks
        ]
        with open(self._index_path / _META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved index to %s", self._index_path)

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True on success."""
        import faiss
        idx_file = self._index_path / _INDEX_FILE
        meta_file = self._index_path / _META_FILE
        if not idx_file.exists() or not meta_file.exists():
            logger.info("No existing FAISS index found at %s", self._index_path)
            return False
        self._index = faiss.read_index(str(idx_file))
        with open(meta_file, encoding="utf-8") as f:
            raw_meta = json.load(f)
        self._chunks = [
            Chunk(
                text=m["text"],
                chunk_id=m["chunk_id"],
                token_count=m.get("token_count", 0),
                article_metadata=ArticleMetadata(
                    title=m["title"],
                    category=m["category"],
                    tags=m.get("tags", []),
                    difficulty=m.get("difficulty", "intermediate"),
                    source=m.get("source", "Finnie Educational Content"),
                    file_path=m.get("file_path", ""),
                ),
            )
            for m in raw_meta
        ]
        logger.info("Loaded FAISS index: %d chunks", len(self._chunks))
        return True

    def build_from_directory(self, articles_dir: Path) -> None:
        """Convenience: chunk all articles in a directory, then build index."""
        from src.rag.chunker import ArticleChunker
        chunker = ArticleChunker()
        chunks = chunker.chunk_directory(articles_dir)
        if not chunks:
            raise ValueError(f"No .md files found in {articles_dir}")
        self.build_index(chunks)
        self.save()

    @property
    def index(self):
        return self._index

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks
