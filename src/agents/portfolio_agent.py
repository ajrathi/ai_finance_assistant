"""Portfolio Analysis Agent — analyzes user portfolio allocations and metrics."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.base_agent import AgentOutput, BaseAgent, SourceAttribution

logger = logging.getLogger(__name__)

PORTFOLIO_DISCLAIMER = (
    "Portfolio analysis shown is illustrative and educational. "
    "Consult a registered investment advisor before making investment decisions."
)

_SYSTEM_EXTRA = """
You specialize in portfolio education and analysis. Given a user's portfolio holdings,
you explain what the numbers mean, how their allocation compares to general principles,
and what concepts they should understand.

Guidelines:
- Describe the portfolio composition clearly (allocation %, sector exposure)
- Explain diversification concepts as they apply to THIS portfolio
- Compare their risk profile to their stated risk tolerance (if available)
- DO NOT recommend specific buy/sell actions
- DO NOT say "you should buy X" or "you should sell Y"
- DO say "portfolios like this tend to have characteristics such as..."
- Frame everything as educational illustration, not personalized advice
"""

_PROMPT_TEMPLATE = """\
User portfolio data:
{portfolio_summary}

User's stated risk tolerance: {risk_tolerance}
User's stated goals: {goals}

Educational context:
{context}

Please provide an educational analysis of this portfolio covering:
1. Overall allocation breakdown (stocks vs bonds vs other)
2. Diversification assessment (geographic, sector, asset class)
3. How this allocation relates to the user's stated risk tolerance
4. Key concepts illustrated by this portfolio
5. General educational observations (no specific buy/sell advice)

End with 2-3 educational follow-up questions.
Prefix each follow-up with "Follow-up:"
"""


def _compute_portfolio_metrics(holdings: List[Dict]) -> Dict:
    """Compute basic portfolio metrics from holdings list."""
    if not holdings:
        return {}

    total_value = sum(h.get("value", 0) for h in holdings)
    if total_value == 0:
        return {}

    metrics = {
        "total_value": total_value,
        "holdings_count": len(holdings),
        "allocations": [],
        "diversification_score": 0,
    }

    # Allocation percentages
    for h in holdings:
        value = h.get("value", 0)
        pct = (value / total_value * 100) if total_value > 0 else 0
        metrics["allocations"].append({
            "ticker": h.get("ticker", "Unknown"),
            "value": value,
            "percentage": round(pct, 2),
        })

    # Simple diversification score: penalize high concentration
    max_pct = max(a["percentage"] for a in metrics["allocations"])
    metrics["diversification_score"] = round(max(0, 100 - max_pct), 1)
    metrics["largest_position_pct"] = max_pct

    return metrics


def _format_portfolio_summary(holdings: List[Dict], metrics: Dict) -> str:
    """Format holdings into a readable summary for the LLM prompt."""
    if not holdings or not metrics:
        return "No portfolio data provided."

    lines = [
        f"Total Portfolio Value: ${metrics.get('total_value', 0):,.2f}",
        f"Number of Holdings: {metrics.get('holdings_count', 0)}",
        f"Diversification Score: {metrics.get('diversification_score', 0)}/100",
        "\nHoldings:",
    ]
    for alloc in metrics.get("allocations", []):
        lines.append(
            f"  {alloc['ticker']}: ${alloc['value']:,.2f} ({alloc['percentage']:.1f}%)"
        )
    return "\n".join(lines)


class PortfolioAnalysisAgent(BaseAgent):
    """Educational portfolio analysis — metrics, allocation, and diversification concepts."""

    name = "Portfolio Analysis Agent"

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
            categories = settings.agents.portfolio.rag_categories
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
                portfolio_data = state.get("portfolio_data") or {}
            else:
                query = getattr(state, "current_query", "")
                user_profile = getattr(state, "user_profile", {})
                portfolio_data = getattr(state, "portfolio_data", None) or {}

            knowledge_level = (
                user_profile.get("knowledge_level", "intermediate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "knowledge_level", "intermediate")
            )
            risk_tolerance = (
                user_profile.get("risk_tolerance", "moderate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "risk_tolerance", "moderate")
            )
            goals = (
                user_profile.get("goals", [])
                if isinstance(user_profile, dict)
                else getattr(user_profile, "goals", [])
            )

            holdings = portfolio_data.get("holdings", []) if portfolio_data else []
            metrics = _compute_portfolio_metrics(holdings)
            portfolio_summary = _format_portfolio_summary(holdings, metrics)

            context_query = query or "portfolio diversification asset allocation"
            context_text, sources = self._retrieve_context(context_query)

            system_prompt = self._build_system_prompt(knowledge_level, extra=_SYSTEM_EXTRA)
            user_prompt = _PROMPT_TEMPLATE.format(
                portfolio_summary=portfolio_summary,
                risk_tolerance=risk_tolerance,
                goals=", ".join(
                    g.get("name", str(g)) if isinstance(g, dict) else str(g)
                    for g in goals
                ) if goals else "Not specified",
                context=context_text or "No specific context retrieved.",
            )

            raw_response = self._llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )

            content, follow_ups = self._parse_follow_ups(raw_response)

            return AgentOutput(
                content=content,
                agent_name=self.name,
                confidence=0.85 if holdings else 0.5,
                sources=sources,
                disclaimers=[PORTFOLIO_DISCLAIMER],
                follow_up_questions=follow_ups[:3],
                metadata={"metrics": metrics},
            )

        except Exception as exc:
            logger.error("PortfolioAnalysisAgent error: %s", exc, exc_info=True)
            return self._make_error_output(str(exc))
