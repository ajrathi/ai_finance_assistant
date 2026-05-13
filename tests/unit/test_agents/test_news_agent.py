"""Unit tests for the News Synthesizer Agent."""
from unittest.mock import MagicMock, patch

import pytest

from src.agents.news_agent import (
    NewsFetcher,
    NewsSynthesizerAgent,
    _format_news,
    NEWS_DISCLAIMER,
)
from src.agents.base_agent import AgentOutput
from src.workflow.state import AgentState, UserProfile


# ─── _format_news ─────────────────────────────────────────────────────────────

class TestFormatNews:
    def test_empty_list_returns_no_news_message(self):
        result = _format_news([])
        assert "no recent news" in result.lower()

    def test_formats_single_item(self):
        items = [{"title": "Fed Raises Rates", "summary": "The Fed increased rates by 0.25%.", "source": "Reuters"}]
        result = _format_news(items)
        assert "Fed Raises Rates" in result
        assert "Reuters" in result

    def test_formats_multiple_items(self):
        items = [
            {"title": "Markets Rise", "summary": "Stocks gain.", "source": "AP"},
            {"title": "Inflation Data", "summary": "CPI falls.", "source": "Bloomberg"},
        ]
        result = _format_news(items)
        assert "Markets Rise" in result
        assert "Inflation Data" in result

    def test_truncates_long_summary(self):
        long_summary = "X" * 500
        items = [{"title": "Test", "summary": long_summary, "source": "Test"}]
        result = _format_news(items)
        assert "..." in result

    def test_handles_missing_fields(self):
        items = [{"title": "No Source Story"}]
        result = _format_news(items)
        assert "No Source Story" in result


# ─── NewsFetcher ──────────────────────────────────────────────────────────────

class TestNewsFetcher:
    def test_returns_cached_news(self):
        fetcher = NewsFetcher(api_key="")
        from datetime import datetime
        cache_key = f"news:financial_markets:{datetime.now().strftime('%Y-%m-%d-%H')}"
        cached_news = [{"title": "Cached Story", "summary": "Old news.", "source": "Cache"}]
        fetcher._cache.set(cache_key, cached_news)
        result = fetcher.get_news()
        assert result == cached_news

    def test_returns_empty_when_all_sources_fail(self):
        fetcher = NewsFetcher(api_key="")
        with patch.object(fetcher, "_fetch_yfinance_news", return_value=[]):
            result = fetcher.get_news()
            assert isinstance(result, list)

    def test_yfinance_fallback_called_when_no_api_key(self):
        fetcher = NewsFetcher(api_key="")
        mock_news = [{"title": "Market Update", "summary": "Markets move.", "source": "Yahoo Finance",
                      "published": "1234567890", "sentiment": "Neutral"}]
        # Isolate from cache state by mocking the cache entirely
        fetcher._cache = MagicMock()
        fetcher._cache.get.return_value = None
        with patch.object(fetcher, "_fetch_yfinance_news", return_value=mock_news) as mock_yf:
            result = fetcher.get_news()
            mock_yf.assert_called_once()

    def test_fetch_yfinance_news_handles_exception(self):
        fetcher = NewsFetcher(api_key="")
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = fetcher._fetch_yfinance_news(5)
            assert result == []

    def test_fetch_alpha_vantage_handles_exception(self):
        fetcher = NewsFetcher(api_key="test_key")
        with patch("httpx.get", side_effect=Exception("timeout")):
            result = fetcher._fetch_alpha_vantage_news("financial_markets", 5)
            assert result == []


# ─── NewsSynthesizerAgent ─────────────────────────────────────────────────────

class TestNewsSynthesizerAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = (
            "The Federal Reserve announced a rate decision today. "
            "This affects borrowing costs across the economy.\n\n"
            "Follow-up: How do interest rates affect bonds?\n"
            "Follow-up: What is the relationship between inflation and rates?"
        )
        self.mock_rag = MagicMock()
        self.mock_rag.retrieve_and_format.return_value = MagicMock(
            context_text="Market concepts and news context.",
            sources=[],
        )
        self.mock_fetcher = MagicMock()
        self.mock_fetcher.get_news.return_value = [
            {"title": "Fed Decision", "summary": "Fed holds rates steady.", "source": "Reuters",
             "published": "2026-04-10", "sentiment": "Neutral"},
        ]
        self.agent = NewsSynthesizerAgent(
            llm_client=self.mock_llm,
            rag_pipeline=self.mock_rag,
            news_fetcher=self.mock_fetcher,
        )

    def test_returns_agent_output(self):
        state = AgentState.initial(query="What is happening in the financial news today?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_disclaimer_present(self):
        state = AgentState.initial(query="Latest financial news")
        result = self.agent.process(state)
        assert any("educational" in d.lower() for d in result.disclaimers)

    def test_follow_ups_extracted(self):
        state = AgentState.initial(query="Summarize financial news")
        result = self.agent.process(state)
        assert len(result.follow_up_questions) >= 1
        assert all("Follow-up:" not in q for q in result.follow_up_questions)

    def test_follow_ups_not_in_content(self):
        state = AgentState.initial(query="Financial news summary")
        result = self.agent.process(state)
        assert "Follow-up:" not in result.content

    def test_news_count_in_metadata(self):
        state = AgentState.initial(query="News summary")
        result = self.agent.process(state)
        assert "news_count" in result.metadata
        assert result.metadata["news_count"] == 1

    def test_graceful_when_news_fetch_fails(self):
        self.mock_fetcher.get_news.side_effect = Exception("API error")
        state = AgentState.initial(query="What's in the news?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_beginner_profile_processes_successfully(self):
        profile = UserProfile(knowledge_level="beginner", risk_tolerance="conservative")
        state = AgentState.initial(query="What does the news say about stocks?", user_profile=profile)
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_agent_name(self):
        assert self.agent.name == "News Synthesizer Agent"

    def test_confidence_is_valid(self):
        state = AgentState.initial(query="Financial news")
        result = self.agent.process(state)
        assert 0.0 <= result.confidence <= 1.0

    def test_llm_failure_returns_error_output(self):
        self.mock_llm.generate.side_effect = Exception("LLM down")
        state = AgentState.initial(query="Financial news today")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)
        assert result.confidence == 0.0 or "error" in result.content.lower() or "unable" in result.content.lower()

    def test_empty_news_still_generates_response(self):
        self.mock_fetcher.get_news.return_value = []
        state = AgentState.initial(query="Any news today?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)
        assert result.metadata["news_count"] == 0
