"""Unit tests for the Market Analysis Agent."""
from unittest.mock import MagicMock, patch

import pytest

from src.agents.market_agent import (
    MarketAnalysisAgent,
    MarketDataFetcher,
    _format_market_data,
    MARKET_DISCLAIMER,
)
from src.agents.base_agent import AgentOutput
from src.workflow.state import AgentState, UserProfile


# ─── _format_market_data ──────────────────────────────────────────────────────

class TestFormatMarketData:
    def test_empty_dict_returns_unavailable(self):
        result = _format_market_data({})
        assert "unavailable" in result.lower()

    def test_formats_positive_change(self):
        data = {
            "S&P 500": {"symbol": "^GSPC", "price": 4900.5, "change": 25.3, "change_pct": 0.52}
        }
        result = _format_market_data(data)
        assert "S&P 500" in result
        assert "4,900.50" in result
        assert "+25.30" in result

    def test_formats_negative_change(self):
        data = {
            "NASDAQ": {"symbol": "^IXIC", "price": 15200.0, "change": -30.1, "change_pct": -0.20}
        }
        result = _format_market_data(data)
        assert "NASDAQ" in result
        assert "-30.10" in result

    def test_skips_entries_without_price(self):
        data = {"BadEntry": {"symbol": "X"}}
        result = _format_market_data(data)
        assert "unavailable" in result.lower()

    def test_multiple_entries(self):
        data = {
            "S&P 500": {"symbol": "^GSPC", "price": 4900.0, "change": 10.0, "change_pct": 0.2},
            "Dow Jones": {"symbol": "^DJI", "price": 38000.0, "change": 50.0, "change_pct": 0.13},
        }
        result = _format_market_data(data)
        assert "S&P 500" in result
        assert "Dow Jones" in result


# ─── MarketDataFetcher ────────────────────────────────────────────────────────

class TestMarketDataFetcher:
    def test_returns_cached_result(self, mock_market_data):
        fetcher = MarketDataFetcher()
        quote = {"symbol": "AAPL", "price": 190.0, "change": 1.0, "change_pct": 0.5}
        fetcher._cache.set("market:quote:AAPL", quote)
        result = fetcher.get_quote("AAPL")
        assert result == quote

    def test_returns_none_when_yfinance_fails(self):
        fetcher = MarketDataFetcher()
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = MagicMock(empty=True)
            result = fetcher._get_yfinance_quote("INVALID_TICKER_XYZ")
            assert result is None

    def test_get_market_overview_returns_dict(self, mock_market_data):
        fetcher = MarketDataFetcher()
        # Pre-populate cache for all three indexes
        fetcher._cache.set("market:quote:^GSPC", mock_market_data["S&P 500"])
        fetcher._cache.set("market:quote:^IXIC", mock_market_data["NASDAQ"])
        fetcher._cache.set("market:quote:^DJI", mock_market_data["Dow Jones"])
        result = fetcher.get_market_overview()
        assert isinstance(result, dict)
        assert "S&P 500" in result

    def test_get_quote_caches_result(self):
        fetcher = MarketDataFetcher()
        quote = {"symbol": "TSLA", "price": 250.0, "change": 5.0, "change_pct": 2.0, "source": "test"}
        with patch.object(fetcher, "_get_yfinance_quote", return_value=quote) as mock_fetch:
            fetcher.get_quote("TSLA")
            fetcher.get_quote("TSLA")  # second call — should use cache
            assert mock_fetch.call_count == 1


# ─── MarketAnalysisAgent ──────────────────────────────────────────────────────

class TestMarketAnalysisAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = (
            "The S&P 500 is an index tracking 500 large companies.\n\n"
            "Follow-up: What causes the market to rise?\n"
            "Follow-up: How is the S&P 500 rebalanced?"
        )
        self.mock_rag = MagicMock()
        self.mock_rag.retrieve_and_format.return_value = MagicMock(
            context_text="Market concepts context.",
            sources=[],
        )
        self.mock_fetcher = MagicMock()
        self.mock_fetcher.get_market_overview.return_value = {
            "S&P 500": {"symbol": "^GSPC", "price": 4900.0, "change": 20.0, "change_pct": 0.41}
        }
        self.agent = MarketAnalysisAgent(
            llm_client=self.mock_llm,
            rag_pipeline=self.mock_rag,
            data_fetcher=self.mock_fetcher,
        )

    def test_returns_agent_output(self):
        state = AgentState.initial(query="What is the S&P 500 doing today?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_disclaimer_present(self):
        state = AgentState.initial(query="How is the market today?")
        result = self.agent.process(state)
        assert any("informational" in d.lower() for d in result.disclaimers)

    def test_follow_ups_extracted(self):
        state = AgentState.initial(query="What moved the market today?")
        result = self.agent.process(state)
        assert len(result.follow_up_questions) >= 1
        assert all("Follow-up:" not in q for q in result.follow_up_questions)

    def test_market_data_in_metadata(self):
        state = AgentState.initial(query="Show me the market overview.")
        result = self.agent.process(state)
        assert "market_data" in result.metadata
        assert "timestamp" in result.metadata

    def test_uses_state_market_data_when_present(self, mock_market_data):
        state = AgentState.initial(query="Market overview")
        state["market_data"] = mock_market_data
        result = self.agent.process(state)
        # Fetcher should not have been called since market_data already in state
        self.mock_fetcher.get_market_overview.assert_not_called()
        assert isinstance(result, AgentOutput)

    def test_graceful_when_fetcher_fails(self):
        self.mock_fetcher.get_market_overview.side_effect = Exception("network error")
        state = AgentState.initial(query="Market prices")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_agent_name(self):
        assert self.agent.name == "Market Analysis Agent"

    def test_confidence_is_valid(self):
        state = AgentState.initial(query="Stock market today")
        result = self.agent.process(state)
        assert 0.0 <= result.confidence <= 1.0

    def test_beginner_profile_processes_successfully(self):
        profile = UserProfile(knowledge_level="beginner", risk_tolerance="conservative")
        state = AgentState.initial(query="What is a stock market index?", user_profile=profile)
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_llm_failure_returns_error_output(self):
        self.mock_llm.generate.side_effect = Exception("LLM unavailable")
        state = AgentState.initial(query="What's happening in the market?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)
        assert result.confidence == 0.0 or "error" in result.content.lower() or "unable" in result.content.lower()
