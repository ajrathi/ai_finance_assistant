"""Unit tests for Finance Q&A Agent."""
from unittest.mock import MagicMock

import pytest

from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.base_agent import GENERAL_DISCLAIMER


class TestFinanceQAAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = (
            "Compound interest is interest earned on interest.\n\n"
            "Follow-up: What is the Rule of 72?\n"
            "Follow-up: How does compound interest compare to simple interest?"
        )
        self.mock_rag = MagicMock()
        from src.rag.schemas import RAGContext
        from src.agents.base_agent import SourceAttribution
        self.mock_rag.retrieve_and_format.return_value = RAGContext(
            context_text="Compound interest earns on both principal and prior interest.",
            sources=[SourceAttribution(title="Compound Interest", category="basics", relevance_score=0.9)],
        )
        self.agent = FinanceQAAgent(llm_client=self.mock_llm, rag_pipeline=self.mock_rag)

    def test_process_returns_agent_output(self, sample_state):
        output = self.agent.process(sample_state)
        assert output is not None
        assert output.agent_name == "Finance Q&A Agent"
        assert len(output.content) > 0

    def test_general_disclaimer_always_present(self, sample_state):
        output = self.agent.process(sample_state)
        assert GENERAL_DISCLAIMER in output.disclaimers

    def test_follow_ups_extracted(self, sample_state):
        output = self.agent.process(sample_state)
        assert len(output.follow_up_questions) >= 1
        # Follow-ups should not be in main content
        for q in output.follow_up_questions:
            assert not q.lower().startswith("follow-up:")

    def test_sources_attached(self, sample_state):
        output = self.agent.process(sample_state)
        assert len(output.sources) >= 1

    def test_empty_query_returns_error_output(self, beginner_profile):
        state = {"current_query": "", "user_profile": beginner_profile}
        output = self.agent.process(state)
        assert output.confidence == 0.0

    def test_llm_failure_returns_graceful_error(self, sample_state):
        self.mock_llm.generate.side_effect = Exception("LLM unavailable")
        output = self.agent.process(sample_state)
        assert "issue" in output.content.lower() or output.confidence == 0.0

    def test_beginner_level_state(self, beginner_profile):
        state = {"current_query": "What is a stock?", "user_profile": beginner_profile}
        output = self.agent.process(state)
        # System prompt should be called with beginner prefix
        assert output is not None
        call_args = self.mock_llm.generate.call_args
        assert "beginner" in call_args.kwargs.get("system_prompt", "").lower() or \
               "new to" in call_args.kwargs.get("system_prompt", "").lower()
