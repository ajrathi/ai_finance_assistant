"""Goal Planning Agent — financial goal setting with compound interest projections."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from src.agents.base_agent import AgentOutput, BaseAgent, SourceAttribution

logger = logging.getLogger(__name__)

GOAL_DISCLAIMER = (
    "Financial projections shown are illustrative scenarios only, "
    "not predictions or guarantees of future results."
)

_SYSTEM_EXTRA = """
You specialize in financial goal planning education. You help users understand
concepts like saving rates, compound growth, and how to think about financial goals.

Guidelines:
- Walk through goal-related calculations clearly and show your math
- Provide multiple scenarios (conservative/moderate/optimistic)
- Emphasize that projections are ILLUSTRATIVE scenarios, not guarantees
- Never promise specific returns or outcomes
- Ask clarifying questions if goal details are incomplete
- Connect goals to actionable concepts (saving rates, tax-advantaged accounts)
"""

_PROMPT_TEMPLATE = """\
User's financial goal information:
{goal_summary}

Illustrative projection scenarios:
{projections}

Educational context:
{context}

User query: {query}

Please provide an educational goal planning response that:
1. Acknowledges the user's goal
2. Explains the projection scenarios (using the numbers above)
3. Discusses relevant concepts (compound interest, savings rate, time horizon)
4. Suggests relevant strategies to consider (tax-advantaged accounts, automation, etc.)
5. Notes clearly that all projections are illustrative

End with 2-3 follow-up questions prefixed "Follow-up:"
"""


def _compound_growth(
    principal: float,
    monthly_contribution: float,
    annual_rate: float,
    years: int,
) -> float:
    """Calculate future value with regular contributions using compound interest."""
    if annual_rate == 0:
        return principal + monthly_contribution * 12 * years
    monthly_rate = annual_rate / 12
    months = years * 12
    # Future value of lump sum + future value of annuity
    fv_lump = principal * ((1 + monthly_rate) ** months)
    fv_annuity = monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    return fv_lump + fv_annuity


def _generate_projections(
    current_savings: float,
    monthly_contribution: float,
    time_horizon_years: int,
    target_amount: float,
) -> Dict:
    """Generate conservative / moderate / optimistic scenarios."""
    rates = {"conservative (5%)": 0.05, "moderate (7%)": 0.07, "optimistic (9%)": 0.09}
    scenarios = {}
    for label, rate in rates.items():
        fv = _compound_growth(current_savings, monthly_contribution, rate, time_horizon_years)
        on_track = fv >= target_amount
        gap = target_amount - fv
        required_monthly = None
        if not on_track and gap > 0:
            # Calculate required monthly contribution to hit target
            monthly_rate = rate / 12
            months = time_horizon_years * 12
            if monthly_rate > 0:
                fv_principal = current_savings * ((1 + monthly_rate) ** months)
                needed_from_contributions = target_amount - fv_principal
                if needed_from_contributions > 0:
                    required_monthly = needed_from_contributions * monthly_rate / (
                        (1 + monthly_rate) ** months - 1
                    )
        scenarios[label] = {
            "future_value": round(fv, 2),
            "on_track": on_track,
            "gap": round(max(0, gap), 2),
            "required_monthly": round(required_monthly, 2) if required_monthly else None,
        }
    return scenarios


def _format_projections(
    scenarios: Dict,
    target_amount: float,
    time_horizon_years: int,
) -> str:
    lines = [f"Target: ${target_amount:,.0f} in {time_horizon_years} years\n"]
    for scenario, data in scenarios.items():
        status = "✓ On track" if data["on_track"] else "✗ Short"
        lines.append(
            f"**{scenario.title()}**: "
            f"${data['future_value']:,.0f} — {status}"
        )
        if not data["on_track"] and data.get("required_monthly"):
            lines.append(
                f"  → To reach target: ~${data['required_monthly']:,.0f}/month needed"
            )
    return "\n".join(lines)


class GoalPlanningAgent(BaseAgent):
    """Educational goal planning with illustrative compound interest projections."""

    name = "Goal Planning Agent"

    def __init__(self, llm_client=None, rag_pipeline=None):
        super().__init__(llm_client=llm_client)
        self._rag = rag_pipeline

    def _get_rag(self):
        if self._rag is None:
            try:
                from src.rag.pipeline import get_rag_pipeline
                self._rag = get_rag_pipeline()
            except Exception:
                self._rag = None
        return self._rag

    def _retrieve_context(self, query: str) -> tuple[str, List[SourceAttribution]]:
        rag = self._get_rag()
        if rag is None:
            return "", []
        try:
            from src.core.config import settings
            categories = settings.agents.goal_planning.rag_categories
            result = rag.retrieve_and_format(query, categories=categories)
            return result.context_text, result.sources
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return "", []

    def _parse_follow_ups(self, text: str) -> tuple[str, List[str]]:
        lines = text.splitlines()
        follow_ups: List[str] = []
        body_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("follow-up:"):
                q = stripped[len("follow-up:"):].strip()
                if q:
                    follow_ups.append(q)
            else:
                body_lines.append(line)
        return "\n".join(body_lines).strip(), follow_ups

    def process(self, state: Any) -> AgentOutput:
        try:
            if isinstance(state, dict):
                query = state.get("current_query", "")
                user_profile = state.get("user_profile", {})
            else:
                query = getattr(state, "current_query", "")
                user_profile = getattr(state, "user_profile", {})

            knowledge_level = (
                user_profile.get("knowledge_level", "intermediate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "knowledge_level", "intermediate")
            )
            goals = (
                user_profile.get("goals", [])
                if isinstance(user_profile, dict)
                else getattr(user_profile, "goals", [])
            )
            risk_tolerance = (
                user_profile.get("risk_tolerance", "moderate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "risk_tolerance", "moderate")
            )

            # Extract goal parameters from user profile goals or use defaults
            goal_data: Dict = {}
            if goals and isinstance(goals, list) and goals:
                first_goal = goals[0] if isinstance(goals[0], dict) else {}
                goal_data = first_goal

            target_amount = float(goal_data.get("target_amount", 0))
            time_horizon = int(goal_data.get("time_horizon_years", 10))
            current_savings = float(goal_data.get("current_savings", 0))
            monthly_contribution = float(goal_data.get("monthly_contribution", 0))

            # Build goal summary text
            goal_summary_lines = [f"Query: {query}"]
            if target_amount > 0:
                goal_summary_lines.extend([
                    f"Target amount: ${target_amount:,.0f}",
                    f"Time horizon: {time_horizon} years",
                    f"Current savings: ${current_savings:,.0f}",
                    f"Monthly contribution: ${monthly_contribution:,.0f}",
                    f"Risk tolerance: {risk_tolerance}",
                ])
            goal_summary = "\n".join(goal_summary_lines)

            # Generate projections if we have enough data
            projections_text = ""
            if target_amount > 0 and time_horizon > 0:
                scenarios = _generate_projections(
                    current_savings, monthly_contribution, time_horizon, target_amount
                )
                projections_text = _format_projections(scenarios, target_amount, time_horizon)
            else:
                projections_text = (
                    "No specific goal data provided. "
                    "Discuss general goal planning concepts based on the user's question."
                )

            context_text, sources = self._retrieve_context(
                query or "financial goal planning compound interest savings"
            )

            system_prompt = self._build_system_prompt(knowledge_level, extra=_SYSTEM_EXTRA)
            user_prompt = _PROMPT_TEMPLATE.format(
                goal_summary=goal_summary,
                projections=projections_text,
                context=context_text or "No specific context retrieved.",
                query=query or "Help me understand financial goal planning.",
            )

            raw_response = self._llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
            )

            content, follow_ups = self._parse_follow_ups(raw_response)

            return AgentOutput(
                content=content,
                agent_name=self.name,
                confidence=0.85,
                sources=sources,
                disclaimers=[GOAL_DISCLAIMER],
                follow_up_questions=follow_ups[:3],
                metadata={"goal_data": goal_data},
            )

        except Exception as exc:
            logger.error("GoalPlanningAgent error: %s", exc, exc_info=True)
            return self._make_error_output(str(exc))
