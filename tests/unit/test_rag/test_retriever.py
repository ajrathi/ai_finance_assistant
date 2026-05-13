"""Unit tests for FAISSRetriever — vector search, filtering, and MMR re-ranking.

The FAISS index and embedding model are both mocked so these tests run
offline with no GPU/model-download requirement.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.retriever import FAISSRetriever
from src.rag.schemas import ArticleMetadata, Chunk, RetrievalResult


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(text: str, category: str, title: str = "Article") -> Chunk:
    return Chunk(
        text=text,
        chunk_id=f"{title.lower().replace(' ', '_')}_0",
        article_metadata=ArticleMetadata(title=title, category=category),
    )


def _unit_vec(dim: int = 4) -> list:
    """Return a normalised unit vector."""
    v = np.ones(dim, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _mock_embedder(dim: int = 4) -> MagicMock:
    """Embedding model that always returns the same unit vector."""
    emb = MagicMock()
    emb.embed_query.return_value = _unit_vec(dim)
    emb.embed_documents.side_effect = lambda texts: [_unit_vec(dim) for _ in texts]
    return emb


def _mock_indexer(chunks: list, scores: list, idxs: list) -> MagicMock:
    """FAISS indexer mock with a fake inner index that returns preset search results."""
    inner = MagicMock()
    inner.ntotal = len(chunks)
    inner.search.return_value = (
        np.array([scores], dtype=np.float32),
        np.array([idxs], dtype=np.int64),
    )
    indexer = MagicMock()
    indexer.index = inner
    indexer.chunks = chunks
    return indexer


# ─── retrieve(): basic behaviour ──────────────────────────────────────────────

class TestFAISSRetrieverRetrieve:

    def test_returns_results_above_threshold(self):
        chunks = [_make_chunk("Compound interest basics.", "basics")]
        indexer = _mock_indexer(chunks, scores=[0.9], idxs=[0])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("compound interest")
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.9)

    def test_filters_results_below_similarity_threshold(self):
        # Default threshold is 0.3; score 0.1 must be excluded
        chunks = [_make_chunk("Low relevance content.", "basics")]
        indexer = _mock_indexer(chunks, scores=[0.1], idxs=[0])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("compound interest")
        assert results == []

    def test_empty_index_returns_empty_list(self):
        indexer = MagicMock()
        indexer.index = None
        indexer.chunks = []
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("anything")
        assert results == []

    def test_negative_faiss_idx_is_skipped(self):
        # FAISS uses -1 for padding when fewer results than requested
        chunks = [_make_chunk("Real chunk.", "basics")]
        indexer = _mock_indexer(chunks, scores=[0.9, 0.8], idxs=[0, -1])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("test")
        assert len(results) == 1

    def test_results_are_retrieval_result_instances(self):
        chunks = [_make_chunk("Some text.", "investing")]
        indexer = _mock_indexer(chunks, scores=[0.85], idxs=[0])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("investing")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_result_chunk_text_matches_source(self):
        chunks = [_make_chunk("Index funds track a market index.", "investing")]
        indexer = _mock_indexer(chunks, scores=[0.88], idxs=[0])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("index funds")
        assert results[0].chunk.text == "Index funds track a market index."


# ─── Category filtering ────────────────────────────────────────────────────────

class TestCategoryFiltering:

    def test_category_filter_excludes_non_matching(self):
        chunks = [
            _make_chunk("Roth IRA basics.", "retirement", "Roth IRA"),
            _make_chunk("Tax bracket overview.", "taxes", "Tax Brackets"),
        ]
        indexer = _mock_indexer(chunks, scores=[0.9, 0.85], idxs=[0, 1])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("retirement accounts", categories=["retirement"])
        assert len(results) == 1
        assert results[0].category == "retirement"

    def test_no_category_filter_returns_all_above_threshold(self):
        chunks = [
            _make_chunk("Basics article.", "basics"),
            _make_chunk("Tax article.", "taxes"),
        ]
        indexer = _mock_indexer(chunks, scores=[0.9, 0.8], idxs=[0, 1])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("finance")
        assert len(results) == 2

    def test_category_filter_with_no_matches_returns_empty(self):
        chunks = [_make_chunk("Market data.", "market_concepts")]
        indexer = _mock_indexer(chunks, scores=[0.9], idxs=[0])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        results = retriever.retrieve("query", categories=["taxes"])
        assert results == []


# ─── retrieve_for_agent() ─────────────────────────────────────────────────────

class TestRetrieveForAgent:

    def test_known_agent_uses_configured_categories(self):
        chunks = [
            _make_chunk("Market data chunk.", "market_concepts"),
            _make_chunk("Tax chunk.", "taxes"),
        ]
        indexer = _mock_indexer(chunks, scores=[0.9, 0.85], idxs=[0, 1])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        # Market Analysis Agent is configured for ["market_concepts"] only
        results = retriever.retrieve_for_agent("What is the S&P 500?", "Market Analysis Agent")
        assert all(r.category == "market_concepts" for r in results)

    def test_unknown_agent_retrieves_without_category_filter(self):
        chunks = [_make_chunk("General content.", "basics")]
        indexer = _mock_indexer(chunks, scores=[0.9], idxs=[0])
        retriever = FAISSRetriever(indexer=indexer, embedding_model=_mock_embedder())
        # Unknown agent name → no category restriction
        results = retriever.retrieve_for_agent("test query", "Unknown Agent")
        assert len(results) == 1
