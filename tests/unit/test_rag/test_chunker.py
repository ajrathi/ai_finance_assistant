"""Unit tests for the document chunker."""
from pathlib import Path
import tempfile

import pytest

from src.rag.chunker import ArticleChunker, _count_tokens, _parse_frontmatter
from src.rag.schemas import ArticleMetadata


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = "---\ntitle: Test\ncategory: basics\n---\nBody text here."
        meta, body = _parse_frontmatter(content)
        assert meta["title"] == "Test"
        assert meta["category"] == "basics"
        assert "Body text" in body

    def test_no_frontmatter(self):
        content = "Just plain text, no frontmatter."
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert "plain text" in body

    def test_empty_frontmatter(self):
        content = "---\n---\nBody"
        meta, body = _parse_frontmatter(content)
        assert meta == {}


class TestCountTokens:
    def test_approximate_token_count(self):
        # ~4 chars per token
        text = "a" * 400
        assert _count_tokens(text) == pytest.approx(100, abs=5)

    def test_empty_text(self):
        assert _count_tokens("") == 1  # minimum 1


class TestArticleChunker:
    def setup_method(self):
        self.chunker = ArticleChunker(chunk_size=200, chunk_overlap=20)

    def test_chunk_short_text(self):
        meta = ArticleMetadata(title="Test", category="basics")
        text = "This is a short text that fits in one chunk."
        chunks = self.chunker.chunk_text(text, meta)
        assert len(chunks) >= 1

    def test_chunks_preserve_metadata(self):
        meta = ArticleMetadata(title="Test Article", category="investing")
        text = "Short text."
        chunks = self.chunker.chunk_text(text, meta)
        assert all(c.article_metadata.category == "investing" for c in chunks)

    def test_long_text_produces_multiple_chunks(self):
        meta = ArticleMetadata(title="Long Article", category="taxes")
        # Generate text longer than chunk_size
        text = " ".join(["This is sentence number %d." % i for i in range(100)])
        chunks = self.chunker.chunk_text(text, meta)
        assert len(chunks) > 1

    def test_chunk_file(self, tmp_path):
        article = tmp_path / "test_article.md"
        article.write_text(
            "---\ntitle: Test Article\ncategory: basics\ndifficulty: beginner\n---\n\n"
            "## Introduction\n\nThis is the introduction section.\n\n"
            "## Main Content\n\nThis is the main content section with more details."
        )
        chunks = self.chunker.chunk_file(article)
        assert len(chunks) >= 1
        assert all(c.article_metadata.title == "Test Article" for c in chunks)

    def test_chunk_directory(self, tmp_path):
        for i in range(3):
            (tmp_path / f"article_{i}.md").write_text(
                f"---\ntitle: Article {i}\ncategory: basics\n---\nContent {i}."
            )
        chunks = self.chunker.chunk_directory(tmp_path)
        assert len(chunks) >= 3

    def test_chunk_ids_unique(self):
        meta = ArticleMetadata(title="Test", category="basics", file_path="test.md")
        text = " ".join(["Sentence %d." % i for i in range(100)])
        chunks = self.chunker.chunk_text(text, meta)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
