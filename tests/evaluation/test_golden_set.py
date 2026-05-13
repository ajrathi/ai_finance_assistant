"""Golden-set evaluations for Finnie's six domain agents.

Each test runs a representative query through the full agent → pipeline chain
with a mocked LLM and asserts that:
  1. The correct agent class processes the query
  2. The output has the expected structural properties (disclaimers, confidence, sources)
  3. Domain-specific constraints are enforced (e.g. tax queries get tax disclaimers)

All LLM and RAG calls are mocked — no external API required.
"""
from __future__ import annotations

from unittest.mock import MagicMock
from typing import Type

import pytest

from src.agents.base_agent import AgentOutput, GENERAL_DISCLAIMER, SourceAttribution
from src.agents.finance_qa_agent import FinanceQAAgent
from src.agents.portfolio_agent import PortfolioAnalysisAgent
from src.agents.market_agent import MarketAnalysisAgent
from src.agents.goal_planning_agent import GoalPlanningAgent
from src.agents.news_agent import NewsSynthesizerAgent
from src.agents.tax_agent import TaxEducationAgent
from src.rag.schemas import RAGContext
from src.workflow.state import AgentState, UserProfile


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _make_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response_text
    return llm


def _make_rag(context: str = "Educational context.", category: str = "basics") -> MagicMock:
    rag = MagicMock()
    rag.retrieve_and_format.return_value = RAGContext(
        context_text=context,
        sources=[SourceAttribution(title="Reference Article", category=category, relevance_score=0.85)],
    )
    return rag


def _state(query: str, profile: UserProfile = None) -> AgentState:
    return AgentState.initial(query=query, user_profile=profile or UserProfile())


# ─── Golden set definition ────────────────────────────────────────────────────
#
# Each entry: (label, agent_class, rag_category, query, llm_response, assertions)
# "assertions" is a dict of {description: callable(output) -> bool}
#
GOLDEN_SET = [
    {
        "label": "compound_interest_qa",
        "agent_class": FinanceQAAgent,
        "rag_category": "basics",
        "query": "What is compound interest and why does it matter?",
        "llm_response": (
            "Compound interest is interest calculated on both the initial principal "
            "and the accumulated interest from previous periods.\n\n"
            "Follow-up: What is the Rule of 72?\n"
            "Follow-up: How does inflation erode compound growth?"
        ),
        "assertions": {
            "has_content": lambda o: len(o.content) > 20,
            "general_disclaimer_present": lambda o: GENERAL_DISCLAIMER in o.disclaimers,
            "follow_ups_extracted": lambda o: len(o.follow_up_questions) >= 1,
            "follow_ups_clean": lambda o: all("follow-up:" not in q.lower() for q in o.follow_up_questions),
            "confidence_in_range": lambda o: 0.0 <= o.confidence <= 1.0,
            "sources_attached": lambda o: len(o.sources) >= 1,
        },
    },
    {
        "label": "roth_ira_tax",
        "agent_class": TaxEducationAgent,
        "rag_category": "taxes",
        "query": "What are the tax advantages of a Roth IRA?",
        "llm_response": (
            "A Roth IRA offers tax-free growth because contributions are made with after-tax dollars.\n\n"
            "Follow-up: What is the Roth IRA contribution limit?\n"
            "Follow-up: Can I withdraw Roth contributions early?"
        ),
        "assertions": {
            "has_content": lambda o: len(o.content) > 20,
            "tax_disclaimer_present": lambda o: any("tax professional" in d.lower() for d in o.disclaimers),
            "general_disclaimer_present": lambda o: any("educational" in d.lower() or "informational" in d.lower() or "not" in d.lower() for d in o.disclaimers),
            "has_follow_ups": lambda o: len(o.follow_up_questions) >= 1,
            "confidence_is_valid": lambda o: 0.0 <= o.confidence <= 1.0,
        },
    },
    {
        "label": "retirement_goal_planning",
        "agent_class": GoalPlanningAgent,
        "rag_category": "retirement",
        "query": "I want to retire with $1 million in 20 years. How much do I need to save monthly?",
        "llm_response": (
            "To accumulate $1 million in 20 years you would need approximately $2,000/month "
            "at a 7% annual return.\n\n"
            "Follow-up: What is the 4% safe withdrawal rule?\n"
            "Follow-up: Should I choose stocks or bonds for retirement?"
        ),
        "assertions": {
            "has_content": lambda o: len(o.content) > 20,
            "projections_disclaimer_present": lambda o: any("illustrative" in d.lower() for d in o.disclaimers),
            "has_follow_ups": lambda o: len(o.follow_up_questions) >= 1,
            "confidence_is_valid": lambda o: 0.0 <= o.confidence <= 1.0,
        },
    },
    {
        "label": "market_overview",
        "agent_class": MarketAnalysisAgent,
        "rag_category": "market_concepts",
        "query": "What is the S&P 500 and how is it performing?",
        "llm_response": (
            "The S&P 500 is a market index tracking 500 large US companies.\n\n"
            "Follow-up: How is the S&P 500 rebalanced?\n"
            "Follow-up: What caused the last major market correction?"
        ),
        "assertions": {
            "has_content": lambda o: len(o.content) > 20,
            "market_disclaimer_present": lambda o: any("informational" in d.lower() or "delayed" in d.lower() for d in o.disclaimers),
            "has_follow_ups": lambda o: len(o.follow_up_questions) >= 1,
            "metadata_has_timestamp": lambda o: "timestamp" in o.metadata,
            "confidence_is_valid": lambda o: 0.0 <= o.confidence <= 1.0,
        },
    },
    {
        "label": "portfolio_analysis",
        "agent_class": PortfolioAnalysisAgent,
        "rag_category": "investing",
        "query": "Am I too heavily weighted in tech stocks?",
        "llm_response": (
            "A heavy concentration in tech stocks increases sector risk.\n\n"
            "Follow-up: What is an appropriate sector allocation?\n"
            "Follow-up: How do I rebalance a tech-heavy portfolio?"
        ),
        "assertions": {
            "has_content": lambda o: len(o.content) > 20,
            "portfolio_disclaimer_present": lambda o: any("investment advisor" in d.lower() for d in o.disclaimers),
            "has_follow_ups": lambda o: len(o.follow_up_questions) >= 1,
            "confidence_is_valid": lambda o: 0.0 <= o.confidence <= 1.0,
        },
    },
    {
        "label": "news_summary",
        "agent_class": NewsSynthesizerAgent,
        "rag_category": "market_concepts",
        "query": "What is happening in financial markets this week?",
        "llm_response": (
            "Markets are reacting to central bank interest rate decisions this week.\n\n"
            "Follow-up: How do interest rate changes affect bonds?\n"
            "Follow-up: What sectors benefit from lower rates?"
        ),
        "assertions": {
            "has_content": lambda o: len(o.content) > 20,
            "general_disclaimer_present": lambda o: any("educational" in d.lower() or "informational" in d.lower() or "not" in d.lower() for d in o.disclaimers),
            "has_follow_ups": lambda o: len(o.follow_up_questions) >= 1,
            "confidence_is_valid": lambda o: 0.0 <= o.confidence <= 1.0,
        },
    },
]


# ─── Knowledge-level adaptation golden set ────────────────────────────────────

KNOWLEDGE_LEVEL_CASES = [
    ("beginner", "new to personal finance", "You are talking to someone who is new"),
    ("intermediate", "basic understanding", "basic understanding"),
    ("advanced", "solid financial knowledge", "solid financial knowledge"),
]


# ─── Parametrised golden-set tests ────────────────────────────────────────────

class TestGoldenSetAgentOutputs:
    """Run each golden-set entry through its agent and assert output constraints."""

    @pytest.mark.parametrize("entry", GOLDEN_SET, ids=[e["label"] for e in GOLDEN_SET])
    def test_agent_output_structure(self, entry):
        llm = _make_llm(entry["llm_response"])
        rag = _make_rag(category=entry["rag_category"])

        agent_class: Type = entry["agent_class"]

        # MarketAnalysisAgent has an extra data_fetcher argument
        if agent_class is MarketAnalysisAgent:
            fetcher = MagicMock()
            fetcher.get_market_overview.return_value = {
                "S&P 500": {"symbol": "^GSPC", "price": 5000.0, "change": 10.0, "change_pct": 0.2}
            }
            agent = agent_class(llm_client=llm, rag_pipeline=rag, data_fetcher=fetcher)
        else:
            agent = agent_class(llm_client=llm, rag_pipeline=rag)

        state = _state(entry["query"])
        output: AgentOutput = agent.process(state)

        for description, assertion in entry["assertions"].items():
            assert assertion(output), (
                f"[{entry['label']}] Assertion failed: {description}\n"
                f"  content={output.content[:80]!r}\n"
                f"  disclaimers={output.disclaimers}\n"
                f"  follow_ups={output.follow_up_questions}\n"
                f"  confidence={output.confidence}"
            )

    @pytest.mark.parametrize("entry", GOLDEN_SET, ids=[e["label"] for e in GOLDEN_SET])
    def test_agent_output_type_is_agent_output(self, entry):
        llm = _make_llm(entry["llm_response"])
        rag = _make_rag(category=entry["rag_category"])
        agent_class: Type = entry["agent_class"]

        if agent_class is MarketAnalysisAgent:
            fetcher = MagicMock()
            fetcher.get_market_overview.return_value = {}
            agent = agent_class(llm_client=llm, rag_pipeline=rag, data_fetcher=fetcher)
        else:
            agent = agent_class(llm_client=llm, rag_pipeline=rag)

        output = agent.process(_state(entry["query"]))
        assert isinstance(output, AgentOutput)

    @pytest.mark.parametrize("entry", GOLDEN_SET, ids=[e["label"] for e in GOLDEN_SET])
    def test_agent_is_resilient_to_llm_failure(self, entry):
        """Every agent must return a graceful error output when the LLM fails."""
        llm = MagicMock()
        llm.generate.side_effect = Exception("LLM unavailable")
        rag = _make_rag(category=entry["rag_category"])
        agent_class: Type = entry["agent_class"]

        if agent_class is MarketAnalysisAgent:
            fetcher = MagicMock()
            fetcher.get_market_overview.return_value = {}
            agent = agent_class(llm_client=llm, rag_pipeline=rag, data_fetcher=fetcher)
        else:
            agent = agent_class(llm_client=llm, rag_pipeline=rag)

        output = agent.process(_state(entry["query"]))
        # Must not raise; must return an AgentOutput (error or otherwise)
        assert isinstance(output, AgentOutput)
        # Error outputs have confidence 0.0
        assert output.confidence == 0.0 or len(output.content) > 0


# ─── Knowledge-level prompt adaptation ───────────────────────────────────────

class TestKnowledgeLevelAdaptation:
    """Each knowledge level must produce a different system prompt."""

    @pytest.mark.parametrize("level,expected_phrase,prompt_fragment", KNOWLEDGE_LEVEL_CASES)
    def test_system_prompt_reflects_knowledge_level(self, level, expected_phrase, prompt_fragment):
        llm = _make_llm("Educational response.\n\nFollow-up: Learn more?")
        rag = _make_rag()
        agent = FinanceQAAgent(llm_client=llm, rag_pipeline=rag)
        profile = UserProfile(knowledge_level=level)
        state = _state("What is a stock?", profile=profile)
        agent.process(state)

        call_kwargs = llm.generate.call_args.kwargs
        system_prompt = call_kwargs.get("system_prompt", "")
        assert prompt_fragment.lower() in system_prompt.lower(), (
            f"Expected {prompt_fragment!r} in system prompt for level={level!r}.\n"
            f"Got: {system_prompt[:200]!r}"
        )


# ─── Cross-agent disclaimer coverage ─────────────────────────────────────────

class TestDisclaimerCoverage:
    """Every agent output must carry at least the general disclaimer."""

    @pytest.mark.parametrize("entry", GOLDEN_SET, ids=[e["label"] for e in GOLDEN_SET])
    def test_general_disclaimer_always_present(self, entry):
        llm = _make_llm(entry["llm_response"])
        rag = _make_rag(category=entry["rag_category"])
        agent_class: Type = entry["agent_class"]

        if agent_class is MarketAnalysisAgent:
            fetcher = MagicMock()
            fetcher.get_market_overview.return_value = {}
            agent = agent_class(llm_client=llm, rag_pipeline=rag, data_fetcher=fetcher)
        else:
            agent = agent_class(llm_client=llm, rag_pipeline=rag)

        output = agent.process(_state(entry["query"]))
        all_disclaimer_text = " ".join(output.disclaimers).lower()
        assert (
            "educational" in all_disclaimer_text
            or "informational" in all_disclaimer_text
            or "not" in all_disclaimer_text
        ), f"[{entry['label']}] No meaningful disclaimer found: {output.disclaimers}"
