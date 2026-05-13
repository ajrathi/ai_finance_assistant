"""FAISS retriever with MMR re-ranking and category filtering."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.rag.indexer import FAISSIndexer, _to_numpy
from src.rag.schemas import RetrievalResult
from src.core.config import settings

logger = logging.getLogger(__name__)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (already L2-normalised → dot product)."""
    return float(np.dot(a, b))


def _mmr(
    query_vec: np.ndarray,
    candidate_results: List[RetrievalResult],
    candidate_vecs: np.ndarray,
    top_k: int,
    lambda_param: float = 0.5,
) -> List[RetrievalResult]:
    """Maximal Marginal Relevance re-ranking to balance relevance and diversity."""
    selected_indices: List[int] = []
    remaining = list(range(len(candidate_results)))

    while len(selected_indices) < top_k and remaining:
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = candidate_results[idx].score
            if selected_indices:
                redundancy = max(
                    _cosine_sim(candidate_vecs[idx], candidate_vecs[sel])
                    for sel in selected_indices
                )
            else:
                redundancy = 0.0
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_results[i] for i in selected_indices]


class FAISSRetriever:
    """Retrieves relevant knowledge base chunks using FAISS with MMR re-ranking."""

    def __init__(
        self,
        indexer: Optional[FAISSIndexer] = None,
        embedding_model=None,
    ):
        self._indexer = indexer
        self._embedding_model = embedding_model

    def _ensure_index(self) -> FAISSIndexer:
        if self._indexer is None:
            self._indexer = FAISSIndexer(embedding_model=self._embedding_model)
            if not self._indexer.load():
                logger.warning("FAISS index not found — retrieval will return empty results.")
        return self._indexer

    def _get_embedder(self):
        if self._embedding_model is None:
            from src.core.embeddings import get_embedding_model
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        categories: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """Search FAISS, optionally filter by category, then MMR re-rank."""
        indexer = self._ensure_index()
        if indexer.index is None or not indexer.chunks:
            return []

        top_k_final = top_k or settings.rag.top_k_final
        top_k_initial = settings.rag.top_k_initial

        embedder = self._get_embedder()
        query_vec_raw = np.array(embedder.embed_query(query), dtype=np.float32).reshape(1, -1)
        # Normalise
        norm = np.linalg.norm(query_vec_raw)
        query_vec = query_vec_raw / norm if norm > 0 else query_vec_raw

        scores, idxs = indexer.index.search(query_vec, min(top_k_initial * 2, indexer.index.ntotal))
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        candidates: List[RetrievalResult] = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(indexer.chunks):
                continue
            chunk = indexer.chunks[idx]
            if categories and chunk.article_metadata.category not in categories:
                continue
            if score < settings.rag.similarity_threshold:
                continue
            candidates.append(RetrievalResult(chunk=chunk, score=score))

        if not candidates:
            return []

        if len(candidates) <= top_k_final:
            return candidates

        # Re-embed candidates for MMR (we already have their FAISS vectors via inner product)
        cand_texts = [r.chunk.text for r in candidates]
        cand_vecs_raw = np.array(embedder.embed_documents(cand_texts), dtype=np.float32)
        norms = np.linalg.norm(cand_vecs_raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        cand_vecs = cand_vecs_raw / norms

        return _mmr(query_vec[0], candidates, cand_vecs, top_k=top_k_final)

    def retrieve_for_agent(self, query: str, agent_name: str) -> List[RetrievalResult]:
        """Retrieve using the agent-specific category filter from config."""
        from src.core.config import settings
        agent_map = {
            "Finance Q&A Agent": settings.agents.finance_qa.rag_categories,
            "Portfolio Analysis Agent": settings.agents.portfolio.rag_categories,
            "Market Analysis Agent": settings.agents.market.rag_categories,
            "Goal Planning Agent": settings.agents.goal_planning.rag_categories,
            "News Synthesizer Agent": settings.agents.news.rag_categories,
            "Tax Education Agent": settings.agents.tax.rag_categories,
        }
        categories = agent_map.get(agent_name, [])
        return self.retrieve(query, categories=categories or None)


# Module-level singleton
_retriever: Optional[FAISSRetriever] = None


def get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever
