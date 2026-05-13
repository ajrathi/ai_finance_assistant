"""Integration tests for chat-based interaction flow.

Validates that the orchestrator routes queries correctly, conversation history
is threaded through state, and all key pipeline nodes produce correct outputs
for chat-style input.

All LLM / external-API calls are mocked so tests run offline.
"""
from __future__ import annotations

import inspect
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base_agent import AgentOutput, SourceAttribution, GENERAL_DISCLAIMER
from src.workflow.intent_labels import Intent
from src.workflow.state import AgentState, UserProfile


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _make_agent_output(content: str = "Educational response.") -> AgentOutput:
    return AgentOutput(
        content=content,
        agent_name="Finance Q&A Agent",
        confidence=0.9,
        sources=[SourceAttribution(title="Basics", category="basics", relevance_score=0.85)],
        disclaimers=[GENERAL_DISCLAIMER],
        follow_up_questions=["What is the Rule of 72?", "How does inflation affect savings?"],
    )


def _state_with(query: str = "test", intent: str = "") -> AgentState:
    """Convenience: create an AgentState with given query and intent."""
    s = AgentState.initial(query=query)
    s["intent"] = intent
    return s


# ─── Orchestrator routing via chat queries ────────────────────────────────────

class TestChatOrchestrationRouting:
    """Verify each chat query type routes to the correct agent node via keywords."""

    def _route(self, query: str) -> str:
        from src.workflow.nodes import route_by_intent
        from src.workflow.router import _keyword_classify

        intent = _keyword_classify(query)
        state = _state_with(query=query, intent=intent.value if intent else Intent.GENERAL_FINANCE.value)
        return route_by_intent(state)

    def test_general_finance_routes_to_finance_qa(self):
        assert self._route("What is compound interest?") == "finance_qa_node"

    def test_portfolio_routes_to_portfolio_node(self):
        assert self._route("Can you analyze my portfolio?") == "portfolio_node"

    def test_tax_routes_to_tax_node(self):
        assert self._route("How do capital gains taxes work?") == "tax_node"

    def test_goal_routes_to_goal_planning_node(self):
        assert self._route("Help me plan for retirement") == "goal_planning_node"

    def test_market_routes_to_market_node(self):
        assert self._route("What is the S&P 500 doing today?") == "market_node"

    def test_news_routes_to_news_node(self):
        assert self._route("Give me financial news headlines") == "news_node"

    def test_greeting_routes_to_finance_qa(self):
        assert self._route("hello there!") == "finance_qa_node"

    def test_out_of_scope_routes_to_fallback(self):
        from src.workflow.nodes import route_by_intent
        state = _state_with(intent=Intent.OUT_OF_SCOPE.value)
        assert route_by_intent(state) == "fallback_node"


# ─── Conversation history in state ────────────────────────────────────────────

class TestConversationHistoryInState:
    """conversation_history must flow correctly through AgentState and run_query."""

    def test_initial_state_carries_history(self):
        history = [
            {"role": "user",      "content": "What is an index fund?"},
            {"role": "assistant", "content": "An index fund tracks a market index."},
        ]
        state = AgentState.initial(query="Tell me more.", conversation_history=history)
        assert state["conversation_history"] == history

    def test_empty_history_when_none_provided(self):
        state = AgentState.initial(query="Hello")
        assert state["conversation_history"] == []

    def test_history_alongside_current_query(self):
        history = [{"role": "user", "content": "Prior message"}]
        state = AgentState.initial(query="New question", conversation_history=history)
        assert state["current_query"] == "New question"
        assert state["conversation_history"] == history

    def test_run_query_signature_accepts_conversation_history(self):
        """run_query() must declare conversation_history as a parameter."""
        from src.workflow.graph import run_query
        params = inspect.signature(run_query).parameters
        assert "conversation_history" in params

    def test_multiple_history_turns_preserved(self):
        history = [
            {"role": "user",      "content": "turn 1"},
            {"role": "assistant", "content": "reply 1"},
        ]
        state = AgentState.initial(query="latest", conversation_history=history)
        assert len(state["conversation_history"]) == 2


# ─── Pipeline nodes: sequential execution ─────────────────────────────────────

class TestPipelineNodeSequence:
    """Run the key nodes in sequence with a mocked agent process() to verify the
    chat pipeline produces correct final_response, sources, and disclaimers.

    This avoids the overhead of the compiled LangGraph and makes mocking simple.
    """

    def _run_pipeline_nodes(self, query: str, intent_str: str, mock_output: AgentOutput) -> dict:
        """Execute the six key pipeline nodes manually in order."""
        from src.workflow.nodes import (
            profile_loader_node,
            intent_classifier_node,
            response_synthesizer_node,
            disclaimer_injector_node,
            profile_updater_node,
        )

        # 1. Build initial state
        state = AgentState.initial(
            query=query,
            user_profile=UserProfile(knowledge_level="beginner", risk_tolerance="moderate"),
            conversation_history=[
                {"role": "user",      "content": "Prior question"},
                {"role": "assistant", "content": "Prior answer"},
            ],
        )

        # 2. profile_loader
        state.update(profile_loader_node(state))

        # 3. intent_classifier (bypass LLM — inject intent directly)
        state["intent"] = intent_str

        # 4. Simulate the chosen agent node producing its output
        state["agent_outputs"] = {"finance_qa": mock_output}

        # 5. response_synthesizer
        synth = response_synthesizer_node(state)
        state.update(synth)

        # 6. disclaimer_injector
        disc = disclaimer_injector_node(state)
        state.update(disc)

        # 7. profile_updater
        state.update(profile_updater_node(state))

        return state

    def test_final_response_present_and_non_empty(self):
        output = _make_agent_output("An index fund tracks a market index.")
        result = self._run_pipeline_nodes(
            "What is an index fund?", Intent.GENERAL_FINANCE.value, output
        )
        assert "final_response" in result
        assert len(result["final_response"]) > 0

    def test_agent_content_in_final_response(self):
        output = _make_agent_output("Index funds track an index.")
        result = self._run_pipeline_nodes(
            "What is an index fund?", Intent.GENERAL_FINANCE.value, output
        )
        # The agent content must appear before disclaimer is appended
        assert "Index funds track an index." in result["final_response"]

    def test_general_disclaimer_always_injected(self):
        output = _make_agent_output("Some educational content.")
        result = self._run_pipeline_nodes(
            "Explain diversification", Intent.GENERAL_FINANCE.value, output
        )
        assert "educational" in result["final_response"].lower()

    def test_portfolio_disclaimer_injected_for_portfolio_intent(self):
        output = _make_agent_output("Portfolio analysis.")
        result = self._run_pipeline_nodes(
            "Analyze my portfolio", Intent.PORTFOLIO_REVIEW.value, output
        )
        assert "investment advisor" in result["final_response"].lower()

    def test_tax_disclaimer_injected_for_tax_intent(self):
        output = _make_agent_output("Capital gains explained.")
        result = self._run_pipeline_nodes(
            "How are capital gains taxed?", Intent.TAX_EDUCATION.value, output
        )
        assert "tax professional" in result["final_response"].lower()

    def test_sources_preserved_from_agent_output(self):
        output = _make_agent_output()
        result = self._run_pipeline_nodes(
            "Tell me about IRAs", Intent.GENERAL_FINANCE.value, output
        )
        assert result["sources"] == output.sources

    def test_iteration_count_incremented(self):
        output = _make_agent_output()
        result = self._run_pipeline_nodes(
            "test query", Intent.GENERAL_FINANCE.value, output
        )
        assert result["iteration_count"] == 1

    def test_conversation_history_preserved_through_pipeline(self):
        """conversation_history must survive all node transformations intact."""
        output = _make_agent_output()
        result = self._run_pipeline_nodes(
            "Follow-up question", Intent.GENERAL_FINANCE.value, output
        )
        history = result["conversation_history"]
        assert history == [
            {"role": "user",      "content": "Prior question"},
            {"role": "assistant", "content": "Prior answer"},
        ]


# ─── Fallback node ────────────────────────────────────────────────────────────

class TestFallbackNode:
    """Out-of-scope queries must receive a helpful, graceful fallback."""

    def test_fallback_response_contains_finance_guidance(self):
        from src.workflow.nodes import fallback_node
        result = fallback_node(_state_with(query="what's the weather?"))
        content = result["agent_outputs"]["fallback"].content
        assert "financial" in content.lower()

    def test_fallback_includes_general_disclaimer(self):
        from src.workflow.nodes import fallback_node
        result = fallback_node(_state_with(query="tell me a joke"))
        assert GENERAL_DISCLAIMER in result["agent_outputs"]["fallback"].disclaimers

    def test_fallback_suggests_example_questions(self):
        from src.workflow.nodes import fallback_node
        result = fallback_node(_state_with(query="what is 2+2?"))
        content = result["agent_outputs"]["fallback"].content
        assert "compound interest" in content.lower() or "ira" in content.lower()


# ─── Disclaimer injector ──────────────────────────────────────────────────────

class TestDisclaimerInjectorInChatFlow:
    """disclaimer_injector_node must annotate responses correctly per intent."""

    def _inject(self, intent: str, response: str) -> dict:
        from src.workflow.nodes import disclaimer_injector_node
        state = _state_with(intent=intent)
        state["final_response"] = response
        state["disclaimers"] = []
        return disclaimer_injector_node(state)

    def test_portfolio_intent_adds_investment_advisor_text(self):
        result = self._inject(Intent.PORTFOLIO_REVIEW.value, "Portfolio response.")
        assert "investment advisor" in result["final_response"].lower()

    def test_tax_intent_adds_tax_professional_text(self):
        result = self._inject(Intent.TAX_EDUCATION.value, "Tax response.")
        assert "tax professional" in result["final_response"].lower()

    def test_goal_intent_adds_projections_disclaimer(self):
        result = self._inject(Intent.GOAL_PLANNING.value, "Goal projections.")
        assert "illustrative" in result["final_response"].lower()

    def test_market_intent_adds_past_performance_note(self):
        result = self._inject(Intent.MARKET_DATA.value, "Market data.")
        assert "past performance" in result["final_response"].lower()

    def test_general_disclaimer_always_appended(self):
        result = self._inject(Intent.GENERAL_FINANCE.value, "Some response.")
        assert "educational" in result["final_response"].lower()

    def test_disclaimers_list_populated(self):
        result = self._inject(Intent.GENERAL_FINANCE.value, "Response.")
        assert isinstance(result["disclaimers"], list)
        assert len(result["disclaimers"]) >= 1


# ─── Response synthesizer ─────────────────────────────────────────────────────

class TestResponseSynthesizerNode:
    def test_single_agent_output_passed_through(self):
        from src.workflow.nodes import response_synthesizer_node
        output = _make_agent_output("Direct response.")
        state = _state_with()
        state["agent_outputs"] = {"finance_qa": output}
        result = response_synthesizer_node(state)
        assert result["final_response"] == output.content
        assert result["sources"] == output.sources

    def test_empty_agent_outputs_returns_fallback_message(self):
        from src.workflow.nodes import response_synthesizer_node
        state = _state_with()
        state["agent_outputs"] = {}
        result = response_synthesizer_node(state)
        assert result["final_response"]
        assert "response" in result["final_response"].lower()
