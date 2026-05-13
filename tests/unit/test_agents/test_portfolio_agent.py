"""Unit tests for Portfolio Analysis Agent."""
from unittest.mock import MagicMock

import pytest

from src.agents.portfolio_agent import (
    PortfolioAnalysisAgent,
    _compute_portfolio_metrics,
    _format_portfolio_summary,
    PORTFOLIO_DISCLAIMER,
)


class TestComputePortfolioMetrics:
    def test_empty_holdings(self):
        metrics = _compute_portfolio_metrics([])
        assert metrics == {}

    def test_single_holding(self):
        holdings = [{"ticker": "AAPL", "value": 1000.0}]
        metrics = _compute_portfolio_metrics(holdings)
        assert metrics["total_value"] == 1000.0
        assert metrics["holdings_count"] == 1
        assert metrics["allocations"][0]["percentage"] == 100.0

    def test_multiple_holdings_sum_to_100(self, sample_holdings):
        metrics = _compute_portfolio_metrics(sample_holdings)
        total_pct = sum(a["percentage"] for a in metrics["allocations"])
        assert abs(total_pct - 100.0) < 0.01

    def test_diversification_score_lower_for_concentrated(self):
        # Very concentrated: one holding is 99%
        concentrated = [
            {"ticker": "A", "value": 990.0},
            {"ticker": "B", "value": 10.0},
        ]
        metrics_concentrated = _compute_portfolio_metrics(concentrated)

        # Diversified: roughly equal
        diversified = [
            {"ticker": "A", "value": 500.0},
            {"ticker": "B", "value": 500.0},
        ]
        metrics_diversified = _compute_portfolio_metrics(diversified)

        assert metrics_concentrated["diversification_score"] < metrics_diversified["diversification_score"]

    def test_zero_total_value(self):
        holdings = [{"ticker": "AAPL", "value": 0}]
        metrics = _compute_portfolio_metrics(holdings)
        # Should handle gracefully (returns empty or zeros)
        assert metrics == {} or metrics.get("total_value") == 0


class TestPortfolioAnalysisAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = (
            "Your portfolio is diversified across stocks and bonds.\n\n"
            "Follow-up: What is optimal asset allocation?\n"
            "Follow-up: How do I rebalance?"
        )
        self.mock_rag = MagicMock()
        from src.rag.schemas import RAGContext
        self.mock_rag.retrieve_and_format.return_value = RAGContext(
            context_text="Asset allocation context.", sources=[]
        )
        self.agent = PortfolioAnalysisAgent(llm_client=self.mock_llm, rag_pipeline=self.mock_rag)

    def test_portfolio_disclaimer_present(self, portfolio_state):
        output = self.agent.process(portfolio_state)
        assert PORTFOLIO_DISCLAIMER in output.disclaimers

    def test_metrics_in_metadata(self, portfolio_state):
        output = self.agent.process(portfolio_state)
        assert "metrics" in output.metadata
        assert "total_value" in output.metadata["metrics"]

    def test_no_portfolio_data(self, sample_state):
        # Should still return a response (graceful)
        output = self.agent.process(sample_state)
        assert output is not None
        assert output.confidence <= 0.85  # Lower confidence without portfolio

    def test_follow_ups_parsed(self, portfolio_state):
        output = self.agent.process(portfolio_state)
        assert len(output.follow_up_questions) >= 1
