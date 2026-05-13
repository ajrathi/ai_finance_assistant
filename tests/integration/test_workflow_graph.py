"""Integration tests for the LangGraph workflow graph."""
from unittest.mock import MagicMock, patch

import pytest

from src.workflow.state import AgentState, UserProfile
from src.workflow.intent_labels import Intent


class TestWorkflowIntentClassifier:
    """Test intent classification without LLM calls."""

    def test_greeting_classified(self):
        from src.workflow.router import IntentClassifier, _keyword_classify
        result = _keyword_classify("hello there!")
        assert result == Intent.GREETING

    def test_tax_keywords(self):
        from src.workflow.router import _keyword_classify
        result = _keyword_classify("How do capital gains taxes work?")
        assert result == Intent.TAX_EDUCATION

    def test_portfolio_keywords(self):
        from src.workflow.router import _keyword_classify
        result = _keyword_classify("Can you analyze my portfolio?")
        assert result == Intent.PORTFOLIO_REVIEW

    def test_market_keywords(self):
        from src.workflow.router import _keyword_classify
        result = _keyword_classify("What is the S&P 500 doing today?")
        assert result == Intent.MARKET_DATA

    def test_goal_keywords(self):
        from src.workflow.router import _keyword_classify
        result = _keyword_classify("Help me plan for retirement")
        assert result == Intent.GOAL_PLANNING

    def test_none_for_unknown(self):
        from src.workflow.router import _keyword_classify
        result = _keyword_classify("What's the weather like?")
        assert result is None


class TestWorkflowNodes:
    """Test individual node functions."""

    def test_profile_loader_initializes_profile(self):
        from src.workflow.nodes import profile_loader_node
        state = AgentState.initial(query="test")
        del state["user_profile"]
        result = profile_loader_node(state)
        # Should not raise and return iteration count
        assert "iteration_count" in result

    def test_intent_classifier_node(self):
        from src.workflow.nodes import intent_classifier_node
        state = AgentState.initial(query="hello")
        result = intent_classifier_node(state)
        assert "intent" in result
        assert result["intent"] == Intent.GREETING.value

    def test_fallback_node_returns_output(self):
        from src.workflow.nodes import fallback_node
        state = AgentState.initial(query="what is the weather?")
        result = fallback_node(state)
        assert "agent_outputs" in result
        assert "fallback" in result["agent_outputs"]

    def test_disclaimer_injector_adds_disclaimer(self):
        from src.workflow.nodes import disclaimer_injector_node
        state = AgentState.initial(query="test")
        state["final_response"] = "Here is some information."
        state["intent"] = Intent.GENERAL_FINANCE.value
        result = disclaimer_injector_node(state)
        assert "final_response" in result
        assert "educational purposes" in result["final_response"].lower()

    def test_route_by_intent_finance(self):
        from src.workflow.nodes import route_by_intent
        state = AgentState.initial(query="test")
        state["intent"] = Intent.GENERAL_FINANCE.value
        assert route_by_intent(state) == "finance_qa_node"

    def test_route_by_intent_tax(self):
        from src.workflow.nodes import route_by_intent
        state = AgentState.initial(query="test")
        state["intent"] = Intent.TAX_EDUCATION.value
        assert route_by_intent(state) == "tax_node"

    def test_route_by_intent_portfolio(self):
        from src.workflow.nodes import route_by_intent
        state = AgentState.initial(query="test")
        state["intent"] = Intent.PORTFOLIO_REVIEW.value
        assert route_by_intent(state) == "portfolio_node"


class TestAgentStateMethods:
    """Test AgentState helper methods."""

    def test_initial_creates_valid_state(self):
        state = AgentState.initial(query="test query")
        assert state["current_query"] == "test query"
        assert state["intent"] == ""
        assert state["agent_outputs"] == {}
        assert state["iteration_count"] == 0

    def test_initial_with_profile(self):
        profile = UserProfile(knowledge_level="advanced", risk_tolerance="aggressive")
        state = AgentState.initial(query="test", user_profile=profile)
        assert state["user_profile"].knowledge_level == "advanced"

    def test_initial_with_portfolio(self):
        portfolio = {"holdings": [{"ticker": "AAPL", "value": 1000.0}]}
        state = AgentState.initial(query="test", portfolio_data=portfolio)
        assert state["portfolio_data"] == portfolio
