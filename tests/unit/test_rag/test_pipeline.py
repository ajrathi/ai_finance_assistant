"""Unit tests for RAG pipeline."""
from unittest.mock import MagicMock, patch

import pytest

from src.rag.pipeline import RAGPipeline
from src.rag.schemas import RAGContext, RetrievalResult
from src.agents.base_agent import SourceAttribution
from src.rag.schemas import Chunk, ArticleMetadata


def _make_result(title: str, category: str, score: float = 0.8) -> RetrievalResult:
    chunk = Chunk(
        text=f"Content from {title}.",
        chunk_id=f"{title}_0",
        article_metadata=ArticleMetadata(title=title, category=category),
    )
    return RetrievalResult(chunk=chunk, score=score)


class TestRAGPipeline:
    def setup_method(self):
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve.return_value = [
            _make_result("Compound Interest", "basics", 0.9),
            _make_result("Time Value of Money", "basics", 0.75),
        ]
        self.mock_retriever.retrieve_for_agent.return_value = [
            _make_result("Index Funds", "investing", 0.85),
        ]
        self.pipeline = RAGPipeline(retriever=self.mock_retriever)

    def test_retrieve_and_format_returns_context(self):
        result = self.pipeline.retrieve_and_format("compound interest")
        assert isinstance(result, RAGContext)
        assert len(result.context_text) > 0
        assert len(result.sources) > 0

    def test_sources_have_correct_fields(self):
        result = self.pipeline.retrieve_and_format("compound interest")
        for src in result.sources:
            assert hasattr(src, "title")
            assert hasattr(src, "category")
            assert hasattr(src, "relevance_score")

    def test_sources_deduplicated(self):
        # Return duplicate results
        dup_result = _make_result("Compound Interest", "basics", 0.9)
        self.mock_retriever.retrieve.return_value = [dup_result, dup_result]
        result = self.pipeline.retrieve_and_format("test")
        # Should deduplicate
        titles = [s.title for s in result.sources]
        assert len(titles) == len(set(titles))

    def test_caches_results(self):
        # First call
        self.pipeline.retrieve_and_format("same query")
        # Second call — retriever should only be called once
        self.pipeline.retrieve_and_format("same query")
        assert self.mock_retriever.retrieve.call_count == 1

    def test_agent_name_uses_retrieve_for_agent(self):
        result = self.pipeline.retrieve_and_format(
            "index funds", agent_name="Finance Q&A Agent"
        )
        assert self.mock_retriever.retrieve_for_agent.called

    def test_empty_retrieval_returns_empty_context(self):
        self.mock_retriever.retrieve.return_value = []
        result = self.pipeline.retrieve_and_format("obscure query")
        assert result.context_text == ""
        assert result.sources == []
