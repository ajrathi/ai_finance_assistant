"""End-to-end RAG pipeline: retrieve → format → return context for LLM injection."""
from __future__ import annotations

import logging
from typing import List, Optional

from src.agents.base_agent import SourceAttribution
from src.rag.retriever import FAISSRetriever, get_retriever
from src.rag.schemas import RAGContext, RetrievalResult
from src.utils.cache import get_rag_cache, make_cache_key

logger = logging.getLogger(__name__)


def _format_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results into a context block for LLM prompt injection."""
    if not results:
        return ""
    parts: List[str] = []
    for i, result in enumerate(results, 1):
        parts.append(
            f"[Source {i}: {result.title} ({result.category})]\n{result.chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _to_attributions(results: List[RetrievalResult]) -> List[SourceAttribution]:
    seen: set = set()
    attributions: List[SourceAttribution] = []
    for r in results:
        key = r.title
        if key not in seen:
            seen.add(key)
            attributions.append(SourceAttribution(
                title=r.title,
                category=r.category,
                relevance_score=r.score,
                excerpt=r.chunk.text[:200] + "…" if len(r.chunk.text) > 200 else r.chunk.text,
            ))
    return attributions


class RAGPipeline:
    """Orchestrates retrieval and formatting for agent consumption."""

    def __init__(self, retriever: Optional[FAISSRetriever] = None):
        self._retriever = retriever or get_retriever()
        self._cache = get_rag_cache()

    def retrieve_and_format(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        agent_name: Optional[str] = None,
    ) -> RAGContext:
        """Retrieve relevant chunks and return a formatted RAGContext.

        Caches results by (query, categories) with 1-hour TTL.
        """
        cache_key = make_cache_key(query, sorted(categories or []), agent_name or "")
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("RAG cache hit for query: %.50s…", query)
            return cached

        if agent_name:
            results = self._retriever.retrieve_for_agent(query, agent_name)
        else:
            results = self._retriever.retrieve(query, categories=categories)

        context_text = _format_context(results)
        sources = _to_attributions(results)

        ctx = RAGContext(context_text=context_text, sources=sources, results=results)
        self._cache.set(cache_key, ctx)
        return ctx


# Module-level singleton
_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
