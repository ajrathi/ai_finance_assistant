"""Shared test fixtures for Finnie."""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from src.agents.base_agent import AgentOutput, SourceAttribution
from src.workflow.state import AgentState, UserProfile


# ─── User Profile Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def beginner_profile() -> UserProfile:
    return UserProfile(
        knowledge_level="beginner",
        risk_tolerance="conservative",
        goals=[],
    )


@pytest.fixture
def advanced_profile() -> UserProfile:
    return UserProfile(
        knowledge_level="advanced",
        risk_tolerance="aggressive",
        goals=[{"name": "Retire", "target_amount": 2_000_000, "time_horizon_years": 25,
                "current_savings": 100_000, "monthly_contribution": 2_000}],
    )


# ─── AgentState Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_state(beginner_profile) -> AgentState:
    return AgentState.initial(
        query="What is compound interest?",
        user_profile=beginner_profile,
    )


@pytest.fixture
def portfolio_state(advanced_profile) -> AgentState:
    portfolio_data = {
        "holdings": [
            {"ticker": "AAPL", "shares": 10, "price_per_share": 190.0, "value": 1900.0},
            {"ticker": "MSFT", "shares": 5, "price_per_share": 420.0, "value": 2100.0},
            {"ticker": "BND", "shares": 50, "price_per_share": 72.0, "value": 3600.0},
        ]
    }
    return AgentState.initial(
        query="Analyze my portfolio",
        user_profile=advanced_profile,
        portfolio_data=portfolio_data,
    )


# ─── Mock LLM Client ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """Mock LLM client that returns a canned response."""
    client = MagicMock()
    client.generate.return_value = (
        "This is an educational response about financial concepts.\n\n"
        "Follow-up: What is the Rule of 72?\n"
        "Follow-up: How does inflation affect savings?"
    )
    client.generate_structured.return_value = '{"intent": "general_finance", "confidence": 0.9}'
    client.stream.return_value = iter(["Educational ", "response ", "chunk."])
    return client


# ─── Mock RAG Pipeline ────────────────────────────────────────────────────────

@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline that returns sample context."""
    from src.rag.schemas import RAGContext
    pipeline = MagicMock()
    pipeline.retrieve_and_format.return_value = RAGContext(
        context_text="[Source 1: Compound Interest (basics)]\nCompound interest earns interest on interest.",
        sources=[
            SourceAttribution(
                title="Compound Interest",
                category="basics",
                relevance_score=0.92,
                excerpt="Compound interest earns interest on interest.",
            )
        ],
    )
    return pipeline


# ─── Mock Market Data ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_market_data() -> dict:
    return {
        "S&P 500": {"symbol": "^GSPC", "price": 4900.5, "change": 25.3, "change_pct": 0.52},
        "NASDAQ": {"symbol": "^IXIC", "price": 15200.0, "change": -30.1, "change_pct": -0.20},
        "Dow Jones": {"symbol": "^DJI", "price": 38500.0, "change": 120.0, "change_pct": 0.31},
    }


# ─── Sample Portfolio ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_holdings() -> List[dict]:
    return [
        {"ticker": "AAPL", "shares": 10, "price_per_share": 190.0, "value": 1900.0},
        {"ticker": "MSFT", "shares": 5, "price_per_share": 420.0, "value": 2100.0},
        {"ticker": "BND", "shares": 50, "price_per_share": 72.0, "value": 3600.0},
        {"ticker": "VTI", "shares": 20, "price_per_share": 240.0, "value": 4800.0},
    ]
