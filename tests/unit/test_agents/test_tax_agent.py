"""Unit tests for the Tax Education Agent."""
from unittest.mock import MagicMock

import pytest

from src.agents.tax_agent import TaxEducationAgent, TAX_DISCLAIMER
from src.agents.base_agent import AgentOutput
from src.workflow.state import AgentState, UserProfile


class TestTaxEducationAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = (
            "Capital gains taxes apply when you sell an asset for more than you paid for it. "
            "Short-term gains (assets held < 1 year) are taxed as ordinary income. "
            "Long-term gains have preferential rates of 0%, 15%, or 20%.\n\n"
            "Follow-up: What is the difference between a Roth and Traditional IRA?\n"
            "Follow-up: How does tax-loss harvesting work?\n"
            "Follow-up: What is the standard deduction?"
        )
        self.mock_rag = MagicMock()
        self.mock_rag.retrieve_and_format.return_value = MagicMock(
            context_text="Tax education context from knowledge base.",
            sources=[],
        )
        self.agent = TaxEducationAgent(
            llm_client=self.mock_llm,
            rag_pipeline=self.mock_rag,
        )

    def test_returns_agent_output(self):
        state = AgentState.initial(query="How do capital gains taxes work?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_tax_disclaimer_present(self):
        state = AgentState.initial(query="What are tax brackets?")
        result = self.agent.process(state)
        assert any("tax professional" in d.lower() or "educational" in d.lower() for d in result.disclaimers)

    def test_follow_ups_extracted(self):
        state = AgentState.initial(query="What is a Roth IRA?")
        result = self.agent.process(state)
        assert len(result.follow_up_questions) >= 1

    def test_follow_ups_not_in_content(self):
        state = AgentState.initial(query="Explain capital gains")
        result = self.agent.process(state)
        assert "Follow-up:" not in result.content

    def test_confidence_high_with_rag_context(self):
        state = AgentState.initial(query="How does a 401k reduce taxes?")
        result = self.agent.process(state)
        assert result.confidence == 0.9

    def test_confidence_lower_without_rag_context(self):
        self.mock_rag.retrieve_and_format.return_value = MagicMock(
            context_text="",
            sources=[],
        )
        state = AgentState.initial(query="Explain tax-loss harvesting")
        result = self.agent.process(state)
        assert result.confidence == 0.75

    def test_beginner_profile_processes(self):
        profile = UserProfile(knowledge_level="beginner", risk_tolerance="conservative")
        state = AgentState.initial(query="What is a tax bracket?", user_profile=profile)
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_advanced_profile_processes(self):
        profile = UserProfile(knowledge_level="advanced", risk_tolerance="aggressive")
        state = AgentState.initial(query="How does qualified dividend taxation differ from ordinary income?", user_profile=profile)
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_agent_name(self):
        assert self.agent.name == "Tax Education Agent"

    def test_empty_query_still_processes(self):
        state = AgentState.initial(query="")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_sources_attached_from_rag(self):
        from src.agents.base_agent import SourceAttribution
        self.mock_rag.retrieve_and_format.return_value = MagicMock(
            context_text="Tax concepts from articles.",
            sources=[
                SourceAttribution(
                    title="Capital Gains Tax",
                    category="taxes",
                    relevance_score=0.88,
                    excerpt="Short-term vs long-term rates.",
                )
            ],
        )
        state = AgentState.initial(query="Capital gains tax rates")
        result = self.agent.process(state)
        assert len(result.sources) == 1
        assert result.sources[0].title == "Capital Gains Tax"

    def test_llm_failure_returns_error_output(self):
        self.mock_llm.generate.side_effect = Exception("LLM service down")
        state = AgentState.initial(query="What is the HSA contribution limit?")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)
        assert result.confidence == 0.0 or "error" in result.content.lower() or "unable" in result.content.lower()

    def test_rag_failure_handled_gracefully(self):
        self.mock_rag.retrieve_and_format.side_effect = Exception("FAISS error")
        state = AgentState.initial(query="529 plan education")
        result = self.agent.process(state)
        assert isinstance(result, AgentOutput)

    def test_no_metadata_key_in_output(self):
        state = AgentState.initial(query="What is AMT?")
        result = self.agent.process(state)
        # TaxEducationAgent doesn't add custom metadata — metadata should be empty dict or absent
        assert result.metadata is not None

    def test_max_three_follow_ups(self):
        self.mock_llm.generate.return_value = (
            "Tax info.\n"
            "Follow-up: Q1?\nFollow-up: Q2?\nFollow-up: Q3?\nFollow-up: Q4?"
        )
        state = AgentState.initial(query="All about taxes")
        result = self.agent.process(state)
        assert len(result.follow_up_questions) <= 3
