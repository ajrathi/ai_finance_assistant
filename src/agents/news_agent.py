"""News Synthesizer Agent — summarizes and contextualizes financial news."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.base_agent import AgentOutput, BaseAgent, SourceAttribution
from src.utils.cache import get_news_cache

logger = logging.getLogger(__name__)

NEWS_DISCLAIMER = (
    "News information is for educational and informational purposes only. "
    "Financial news should not be used as the sole basis for investment decisions."
)

_SYSTEM_EXTRA = """
You specialize in financial news education. You summarize recent financial news
and explain what it means in plain terms.

Guidelines:
- Summarize news clearly and objectively
- Explain financial jargon in accessible terms
- Contextualize news within broader economic or market themes
- Do NOT make predictions about how news will affect markets
- Do NOT recommend trades based on news
- Always encourage users to read primary sources
"""

_PROMPT_TEMPLATE = """\
Recent financial news headlines and summaries:
{news_text}

Educational context:
{context}

User query: {query}

Please provide an educational news synthesis that:
1. Summarizes the key news stories
2. Explains relevant financial concepts the news relates to
3. Provides context for understanding the news
4. Notes any limitations or caveats about interpreting financial news

End with 2-3 educational follow-ups prefixed "Follow-up:"
"""


class NewsFetcher:
    """Fetches financial news from Alpha Vantage NEWS_SENTIMENT endpoint."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._cache = get_news_cache()

    def get_news(self, topics: str = "financial_markets", limit: int = 5) -> List[Dict]:
        """Fetch news with caching."""
        cache_key = f"news:{topics}:{datetime.now().strftime('%Y-%m-%d-%H')}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        if self._api_key:
            news = self._fetch_alpha_vantage_news(topics, limit)
            if news:
                self._cache.set(cache_key, news)
                return news

        # Fallback: yfinance news for major tickers
        news = self._fetch_yfinance_news(limit)
        if news:
            self._cache.set(cache_key, news)
        return news

    def _fetch_alpha_vantage_news(self, topics: str, limit: int) -> List[Dict]:
        try:
            import httpx
            from src.core.config import settings
            params = {
                "function": "NEWS_SENTIMENT",
                "topics": topics,
                "limit": limit,
                "apikey": self._api_key,
            }
            response = httpx.get(
                settings.apis.alpha_vantage.base_url,
                params=params,
                timeout=settings.apis.alpha_vantage.timeout,
            )
            data = response.json()
            items = data.get("feed", [])
            return [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "published": item.get("time_published", ""),
                    "sentiment": item.get("overall_sentiment_label", "Neutral"),
                }
                for item in items[:limit]
            ]
        except Exception as exc:
            logger.warning("Alpha Vantage news fetch failed: %s", exc)
            return []

    def _fetch_yfinance_news(self, limit: int) -> List[Dict]:
        try:
            import yfinance as yf
            ticker = yf.Ticker("^GSPC")  # S&P 500 news
            news = ticker.news or []
            return [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", item.get("title", "")),
                    "source": item.get("publisher", "Yahoo Finance"),
                    "published": str(item.get("providerPublishTime", "")),
                    "sentiment": "Neutral",
                }
                for item in news[:limit]
            ]
        except Exception as exc:
            logger.warning("yfinance news fetch failed: %s", exc)
            return []


def _format_news(news_items: List[Dict]) -> str:
    if not news_items:
        return "No recent news available. Using general financial education context."
    lines = []
    for i, item in enumerate(news_items, 1):
        lines.append(f"**{i}. {item.get('title', 'Untitled')}**")
        summary = item.get("summary", "")
        if summary and summary != item.get("title", ""):
            lines.append(f"   {summary[:300]}{'...' if len(summary) > 300 else ''}")
        source = item.get("source", "")
        if source:
            lines.append(f"   *Source: {source}*")
        lines.append("")
    return "\n".join(lines)


class NewsSynthesizerAgent(BaseAgent):
    """Financial news summarization and contextualization agent."""

    name = "News Synthesizer Agent"

    def __init__(self, llm_client=None, rag_pipeline=None, news_fetcher=None):
        super().__init__(llm_client=llm_client)
        self._rag = rag_pipeline
        self._news_fetcher = news_fetcher

    def _get_fetcher(self) -> NewsFetcher:
        if self._news_fetcher is None:
            from src.core.config import settings
            self._news_fetcher = NewsFetcher(api_key=settings.apis.alpha_vantage.api_key)
        return self._news_fetcher

    def _get_rag(self):
        if self._rag is None:
            try:
                from src.rag.pipeline import get_rag_pipeline
                self._rag = get_rag_pipeline()
            except Exception:
                self._rag = None
        return self._rag

    def _retrieve_context(self, query: str) -> tuple[str, List[SourceAttribution]]:
        rag = self._get_rag()
        if rag is None:
            return "", []
        try:
            from src.core.config import settings
            categories = settings.agents.news.rag_categories
            result = rag.retrieve_and_format(query, categories=categories)
            return result.context_text, result.sources
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return "", []

    def _parse_follow_ups(self, text: str) -> tuple[str, List[str]]:
        lines = text.splitlines()
        follow_ups: List[str] = []
        body_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("follow-up:"):
                q = stripped[len("follow-up:"):].strip()
                if q:
                    follow_ups.append(q)
            else:
                body_lines.append(line)
        return "\n".join(body_lines).strip(), follow_ups

    def process(self, state: Any) -> AgentOutput:
        try:
            if isinstance(state, dict):
                query = state.get("current_query", "")
                user_profile = state.get("user_profile", {})
            else:
                query = getattr(state, "current_query", "")
                user_profile = getattr(state, "user_profile", {})

            knowledge_level = (
                user_profile.get("knowledge_level", "intermediate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "knowledge_level", "intermediate")
            )

            # Fetch news
            news_items = []
            try:
                news_items = self._get_fetcher().get_news(limit=5)
            except Exception as exc:
                logger.warning("News fetch failed: %s", exc)

            news_text = _format_news(news_items)
            context_text, sources = self._retrieve_context(
                query or "financial news market trends"
            )

            system_prompt = self._build_system_prompt(knowledge_level, extra=_SYSTEM_EXTRA)
            user_prompt = _PROMPT_TEMPLATE.format(
                news_text=news_text,
                context=context_text or "No specific context retrieved.",
                query=query or "Summarize recent financial news.",
            )

            raw_response = self._llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )

            content, follow_ups = self._parse_follow_ups(raw_response)

            return AgentOutput(
                content=content,
                agent_name=self.name,
                confidence=0.8,
                sources=sources,
                disclaimers=[NEWS_DISCLAIMER],
                follow_up_questions=follow_ups[:3],
                metadata={"news_count": len(news_items)},
            )

        except Exception as exc:
            logger.error("NewsSynthesizerAgent error: %s", exc, exc_info=True)
            return self._make_error_output(str(exc))
