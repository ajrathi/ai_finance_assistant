"""LangGraph node functions for the Finnie workflow."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict

from src.workflow.intent_labels import Intent, INTENT_TO_AGENT
from src.workflow.router import IntentClassifier
from src.workflow.state import AgentState, UserProfile

logger = logging.getLogger(__name__)

# ─── Safety Guardrail ─────────────────────────────────────────────────────────

_HARMFUL_RE = re.compile(
    r"\b("
    r"suicid[ae]?"                              # suicide, suicidal
    r"|kill\s+(my|your|him|her|them)self"       # kill myself/yourself
    r"|self[- ]?harm"                           # self-harm / self harm
    r"|self[- ]?hurt"
    r"|ways?\s+to\s+die"
    r"|end\s+(my|your|his|her|their)\s+life"
    r"|hurt\s+(my|your)self"
    r"|cut(ting)?\s+(my|your)self"
    r"|hang(ing)?\s+(my|your)self"
    r")",
    re.IGNORECASE,
)

_CRISIS_RESPONSE = (
    "I'm a financial education assistant and I'm not able to help with that request.\n\n"
    "If you or someone you know is in crisis or having thoughts of self-harm, "
    "please reach out for support right away:\n\n"
    "- **988 Suicide & Crisis Lifeline** — Call or text **988** (US, available 24/7)\n"
    "- **Crisis Text Line** — Text **HOME** to **741741**\n"
    "- **International resources** — https://findahelpline.com\n\n"
    "You are not alone. Please talk to someone who can help. 💙"
)

# Disclaimer registry
DISCLAIMERS = {
    "general": "This information is for educational purposes only and does not constitute financial advice.",
    "portfolio": "Portfolio analysis is illustrative. Consult a registered investment advisor before making investment decisions.",
    "tax": "Tax information is educational. Consult a qualified tax professional for advice specific to your situation.",
    "market": "Market data is for informational purposes and may be delayed. Past performance does not indicate future results.",
    "projections": "Financial projections shown are illustrative scenarios only, not predictions or guarantees.",
}

_classifier: IntentClassifier = None


def _get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


# ─── Node Functions ────────────────────────────────────────────────────────────

def safety_check_node(state: AgentState) -> Dict:
    """Block harmful queries before any LLM call. Returns crisis response if triggered."""
    query = state.get("current_query", "")
    if _HARMFUL_RE.search(query):
        logger.warning("Safety guardrail triggered — harmful content detected (query redacted)")
        return {
            "intent": "safety_block",
            "final_response": _CRISIS_RESPONSE,
        }
    return {}


def route_after_safety(state: AgentState) -> str:
    """Route past safety check: skip pipeline entirely if blocked."""
    if state.get("intent") == "safety_block":
        return "profile_updater_node"
    return "intent_classifier_node"


def profile_loader_node(state: AgentState) -> Dict:
    """Load or initialize user profile. In production this would load from a store."""
    if not state.get("user_profile"):
        state["user_profile"] = UserProfile()
    return {"iteration_count": state.get("iteration_count", 0)}


def intent_classifier_node(state: AgentState) -> Dict:
    """Classify the current query into an intent."""
    query = state.get("current_query", "")
    try:
        result = _get_classifier().classify(query)
        intent_str = result.intent.value
        logger.info("Intent classified: %s (%.2f)", intent_str, result.confidence)
    except Exception as exc:
        logger.error("Intent classification failed: %s", exc)
        intent_str = Intent.GENERAL_FINANCE.value
    return {"intent": intent_str}


def finance_qa_node(state: AgentState) -> Dict:
    """Run the Finance Q&A agent."""
    from src.agents.finance_qa_agent import FinanceQAAgent
    agent = FinanceQAAgent()
    output = agent.process(state)
    return {"agent_outputs": {"finance_qa": output}}


def portfolio_node(state: AgentState) -> Dict:
    """Run the Portfolio Analysis agent."""
    from src.agents.portfolio_agent import PortfolioAnalysisAgent
    agent = PortfolioAnalysisAgent()
    output = agent.process(state)
    return {"agent_outputs": {"portfolio": output}}


def market_node(state: AgentState) -> Dict:
    """Run the Market Analysis agent."""
    from src.agents.market_agent import MarketAnalysisAgent
    agent = MarketAnalysisAgent()
    output = agent.process(state)
    # Also store market data back for UI consumption
    return {
        "agent_outputs": {"market": output},
        "market_data": output.metadata.get("market_data"),
    }


def goal_planning_node(state: AgentState) -> Dict:
    """Run the Goal Planning agent."""
    from src.agents.goal_planning_agent import GoalPlanningAgent
    agent = GoalPlanningAgent()
    output = agent.process(state)
    return {"agent_outputs": {"goal_planning": output}}


def news_node(state: AgentState) -> Dict:
    """Run the News Synthesizer agent."""
    from src.agents.news_agent import NewsSynthesizerAgent
    agent = NewsSynthesizerAgent()
    output = agent.process(state)
    return {"agent_outputs": {"news": output}}


def tax_node(state: AgentState) -> Dict:
    """Run the Tax Education agent."""
    from src.agents.tax_agent import TaxEducationAgent
    agent = TaxEducationAgent()
    output = agent.process(state)
    return {"agent_outputs": {"tax": output}}


def fallback_node(state: AgentState) -> Dict:
    """Return a graceful fallback response for out-of-scope or error states."""
    content = (
        "I specialize in financial education topics like investing, budgeting, "
        "retirement planning, and taxes. I didn't quite understand your question "
        "in that context. Could you rephrase it or ask about a financial topic? "
        "\n\nFor example, you could ask:\n"
        "- 'What is compound interest?'\n"
        "- 'How do I start investing?'\n"
        "- 'What's the difference between a Roth and Traditional IRA?'"
    )
    from src.agents.base_agent import AgentOutput
    fallback_output = AgentOutput(
        content=content,
        agent_name="Fallback",
        confidence=1.0,
        disclaimers=[DISCLAIMERS["general"]],
    )
    return {"agent_outputs": {"fallback": fallback_output}}


def response_synthesizer_node(state: AgentState) -> Dict:
    """Merge all agent outputs into a single final response."""
    agent_outputs = state.get("agent_outputs", {})

    if not agent_outputs:
        return {"final_response": "I wasn't able to generate a response. Please try again."}

    outputs = list(agent_outputs.values())

    if len(outputs) == 1:
        # Single agent — use its response directly
        output = outputs[0]
        return {
            "final_response": output.content,
            "sources": output.sources,
            "disclaimers": output.disclaimers,
        }

    # Multi-agent: synthesize with Gemini
    try:
        from src.core.llm_client import get_llm_client
        llm = get_llm_client()

        sections = []
        all_sources = []
        all_disclaimers = set()
        for key, output in agent_outputs.items():
            sections.append(f"**{output.agent_name}:**\n{output.content}")
            all_sources.extend(output.sources)
            all_disclaimers.update(output.disclaimers)

        combined = "\n\n---\n\n".join(sections)
        synth_prompt = (
            f"You have received responses from multiple specialized financial education agents:\n\n"
            f"{combined}\n\n"
            f"Synthesize these into a single, coherent, well-structured educational response. "
            f"Don't repeat information. Keep the response clear and organized. "
            f"Preserve all factual content and nuance."
        )
        final = llm.generate(user_prompt=synth_prompt, temperature=0.2)

        # Deduplicate sources
        seen = set()
        unique_sources = []
        for src in all_sources:
            key = src.title if hasattr(src, "title") else str(src)
            if key not in seen:
                unique_sources.append(src)
                seen.add(key)

        return {
            "final_response": final,
            "sources": unique_sources,
            "disclaimers": list(all_disclaimers),
        }

    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        # Fallback: concatenate
        combined = "\n\n---\n\n".join(
            f"**{o.agent_name}:**\n{o.content}" for o in outputs
        )
        return {"final_response": combined}


def disclaimer_injector_node(state: AgentState) -> Dict:
    """Append context-appropriate disclaimers to the final response."""
    final_response = state.get("final_response", "")
    disclaimers = state.get("disclaimers", [])
    intent = state.get("intent", "")

    # Add intent-specific disclaimers if not already present
    intent_disclaimer_map = {
        Intent.PORTFOLIO_REVIEW.value: DISCLAIMERS["portfolio"],
        Intent.TAX_EDUCATION.value: DISCLAIMERS["tax"],
        Intent.MARKET_DATA.value: DISCLAIMERS["market"],
        Intent.GOAL_PLANNING.value: DISCLAIMERS["projections"],
    }

    for intent_val, disclaimer in intent_disclaimer_map.items():
        if intent == intent_val and disclaimer not in disclaimers:
            disclaimers = [disclaimer] + list(disclaimers)

    # Always include general disclaimer
    general = DISCLAIMERS["general"]
    if general not in disclaimers:
        disclaimers.append(general)

    if disclaimers:
        disclaimer_text = "\n\n---\n> *" + " | ".join(set(disclaimers)) + "*"
        final_response = (final_response or "") + disclaimer_text

    return {"final_response": final_response, "disclaimers": list(set(disclaimers))}


def profile_updater_node(state: AgentState) -> Dict:
    """Update user profile based on interaction (lightweight heuristic)."""
    user_profile = state.get("user_profile")
    if not user_profile:
        return {}

    # Increment iteration count
    return {"iteration_count": state.get("iteration_count", 0) + 1}


def route_by_intent(state: AgentState) -> str:
    """Conditional edge function: return the next node name based on intent."""
    intent = state.get("intent", Intent.GENERAL_FINANCE.value)

    route_map = {
        Intent.GENERAL_FINANCE.value: "finance_qa_node",
        Intent.PORTFOLIO_REVIEW.value: "portfolio_node",
        Intent.MARKET_DATA.value: "market_node",
        Intent.GOAL_PLANNING.value: "goal_planning_node",
        Intent.NEWS_SUMMARY.value: "news_node",
        Intent.TAX_EDUCATION.value: "tax_node",
        Intent.GREETING.value: "finance_qa_node",
        Intent.OUT_OF_SCOPE.value: "fallback_node",
        Intent.MULTI_DOMAIN.value: "finance_qa_node",
    }
    return route_map.get(intent, "finance_qa_node")
