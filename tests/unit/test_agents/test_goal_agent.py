"""Unit tests for Goal Planning Agent."""
from unittest.mock import MagicMock

import pytest

from src.agents.goal_planning_agent import (
    GoalPlanningAgent,
    _compound_growth,
    _generate_projections,
    GOAL_DISCLAIMER,
)


class TestCompoundGrowth:
    def test_zero_rate(self):
        result = _compound_growth(10_000, 500, 0.0, 10)
        assert result == pytest.approx(10_000 + 500 * 12 * 10, rel=0.001)

    def test_positive_rate(self):
        result = _compound_growth(10_000, 0, 0.07, 10)
        expected = 10_000 * (1.07 / 12 + 1) ** 120  # rough
        # Just check it's significantly higher than principal
        assert result > 10_000

    def test_no_contributions(self):
        # $10k at 7% for 30 years with no contributions (monthly compounding)
        result = _compound_growth(10_000, 0, 0.07, 30)
        # Monthly compounding: 10000 * (1 + 0.07/12)^360 ≈ 81,165
        assert result == pytest.approx(81_165, rel=0.05)

    def test_no_principal(self):
        # $500/month at 7% for 30 years
        result = _compound_growth(0, 500, 0.07, 30)
        assert result > 500 * 12 * 30  # More than raw contributions


class TestGenerateProjections:
    def test_returns_three_scenarios(self):
        scenarios = _generate_projections(10_000, 500, 20, 500_000)
        assert len(scenarios) == 3

    def test_on_track_detection(self):
        # Very small target, large savings, long horizon — should be on track
        scenarios = _generate_projections(1_000_000, 0, 1, 1_000)
        for label, data in scenarios.items():
            assert data["on_track"] is True

    def test_not_on_track_shows_gap(self):
        # Impossible target
        scenarios = _generate_projections(100, 50, 1, 10_000_000)
        for label, data in scenarios.items():
            assert data["on_track"] is False
            assert data["gap"] > 0

    def test_required_monthly_when_not_on_track(self):
        scenarios = _generate_projections(0, 0, 30, 1_000_000)
        for label, data in scenarios.items():
            if not data["on_track"]:
                assert data.get("required_monthly") is not None
                assert data["required_monthly"] > 0


class TestGoalPlanningAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = (
            "Based on your goal of saving $1M...\n\n"
            "Follow-up: How does inflation affect my goal?\n"
            "Follow-up: Should I use a Roth IRA?"
        )
        self.mock_rag = MagicMock()
        from src.rag.schemas import RAGContext
        self.mock_rag.retrieve_and_format.return_value = RAGContext(
            context_text="Retirement planning context.", sources=[]
        )
        self.agent = GoalPlanningAgent(llm_client=self.mock_llm, rag_pipeline=self.mock_rag)

    def test_goal_disclaimer_present(self, sample_state):
        output = self.agent.process(sample_state)
        assert GOAL_DISCLAIMER in output.disclaimers

    def test_handles_missing_goal_data(self, sample_state):
        output = self.agent.process(sample_state)
        assert output is not None
        assert len(output.content) > 0

    def test_processes_goal_from_profile(self, advanced_profile):
        state = {
            "current_query": "Am I on track for retirement?",
            "user_profile": advanced_profile,
        }
        output = self.agent.process(state)
        assert output is not None
        assert output.agent_name == "Goal Planning Agent"
