"""Embedding model setup for Finnie's RAG pipeline.

Primary: Google text-embedding-004 via google-genai SDK (free tier)
Fallback: sentence-transformers/all-MiniLM-L6-v2 (local, no API needed)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Protocol

logger = logging.getLogger(__name__)


class EmbeddingModel(Protocol):
    """Protocol that any embedding backend must satisfy."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class _GoogleEmbeddings:
    """Thin wrapper around google-genai embed_content for RAG use."""

    def __init__(self, client, model_name: str):
        self._client = client
        self._model = model_name

    def embed_query(self, text: str) -> List[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        return list(response.embeddings[0].values)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


def _load_google_embeddings(model_name: str, api_key: str) -> Optional[EmbeddingModel]:
    """Try to load Google's embedding model via the google-genai SDK."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        embeddings = _GoogleEmbeddings(client, model_name)
        embeddings.embed_query("test")  # warm-up probe
        logger.info("Using Google embedding model: %s", model_name)
        return embeddings
    except Exception as exc:
        logger.warning("Google embeddings unavailable: %s", exc)
        return None


def _load_local_embeddings(model_name: str) -> EmbeddingModel:
    """Load sentence-transformers as a local fallback (no API key needed)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    logger.info("Using local HuggingFace embedding model: %s", model_name)
    return HuggingFaceEmbeddings(model_name=model_name)


def load_embedding_model(
    primary_model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EmbeddingModel:
    """Load the best available embedding model.

    Tries Google embeddings first; falls back to local sentence-transformers.
    """
    from src.core.config import settings

    primary = primary_model or settings.llm.embedding_model
    fallback = fallback_model or settings.llm.embedding_fallback
    key = api_key or settings.google_api_key

    if key:
        model = _load_google_embeddings(primary, key)
        if model is not None:
            return model

    return _load_local_embeddings(fallback)


# Singleton — initialized lazily on first access
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model()
    return _embedding_model
