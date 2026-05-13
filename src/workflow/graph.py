"""LangGraph StateGraph assembly for the Finnie workflow."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the Finnie LangGraph StateGraph."""
    from langgraph.graph import StateGraph, END

    from src.workflow.state import FinnieState
    from src.workflow.nodes import (
        safety_check_node,
        route_after_safety,
        profile_loader_node,
        intent_classifier_node,
        finance_qa_node,
        portfolio_node,
        market_node,
        goal_planning_node,
        news_node,
        tax_node,
        fallback_node,
        response_synthesizer_node,
        disclaimer_injector_node,
        profile_updater_node,
        route_by_intent,
    )

    graph = StateGraph(FinnieState)

    # Register all nodes
    graph.add_node("safety_check_node", safety_check_node)
    graph.add_node("profile_loader_node", profile_loader_node)
    graph.add_node("intent_classifier_node", intent_classifier_node)
    graph.add_node("finance_qa_node", finance_qa_node)
    graph.add_node("portfolio_node", portfolio_node)
    graph.add_node("market_node", market_node)
    graph.add_node("goal_planning_node", goal_planning_node)
    graph.add_node("news_node", news_node)
    graph.add_node("tax_node", tax_node)
    graph.add_node("fallback_node", fallback_node)
    graph.add_node("response_synthesizer_node", response_synthesizer_node)
    graph.add_node("disclaimer_injector_node", disclaimer_injector_node)
    graph.add_node("profile_updater_node", profile_updater_node)

    # Entry point
    graph.set_entry_point("safety_check_node")

    # Safety gate: harmful queries bypass the entire pipeline
    graph.add_conditional_edges(
        "safety_check_node",
        route_after_safety,
        {
            "intent_classifier_node": "profile_loader_node",
            "profile_updater_node": "profile_updater_node",
        },
    )

    # Linear flow: loader → classifier → router
    graph.add_edge("profile_loader_node", "intent_classifier_node")

    # Conditional routing from classifier to appropriate agent
    graph.add_conditional_edges(
        "intent_classifier_node",
        route_by_intent,
        {
            "finance_qa_node": "finance_qa_node",
            "portfolio_node": "portfolio_node",
            "market_node": "market_node",
            "goal_planning_node": "goal_planning_node",
            "news_node": "news_node",
            "tax_node": "tax_node",
            "fallback_node": "fallback_node",
        },
    )

    # All agent nodes → synthesizer
    for agent_node in [
        "finance_qa_node",
        "portfolio_node",
        "market_node",
        "goal_planning_node",
        "news_node",
        "tax_node",
        "fallback_node",
    ]:
        graph.add_edge(agent_node, "response_synthesizer_node")

    # Post-synthesis flow
    graph.add_edge("response_synthesizer_node", "disclaimer_injector_node")
    graph.add_edge("disclaimer_injector_node", "profile_updater_node")
    graph.add_edge("profile_updater_node", END)

    compiled = graph.compile()
    logger.info("Finnie LangGraph compiled successfully")
    return compiled


# Module-level compiled graph singleton
_compiled_graph = None


def get_compiled_graph():
    """Return the compiled graph, building it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(
    query: str,
    user_profile=None,
    portfolio_data=None,
    messages=None,
    conversation_history=None,
) -> dict:
    """Convenience function: run a query through the full graph pipeline.

    Args:
        query: The current user query.
        user_profile: UserProfile dataclass instance.
        portfolio_data: Dict with 'holdings' list, or None.
        messages: LangChain BaseMessage list (reserved for future use).
        conversation_history: Recent chat turns as list of
            {"role": "user"|"assistant", "content": str} dicts.
            Passed to agents for multi-turn context awareness.

    Returns:
        Final state dict with ``final_response``, ``sources``, and
        ``disclaimers`` keys.
    """
    from src.workflow.state import AgentState, UserProfile

    if user_profile is None:
        user_profile = UserProfile()

    initial_state = AgentState.initial(
        query=query,
        user_profile=user_profile,
        portfolio_data=portfolio_data,
        messages=messages or [],
        conversation_history=conversation_history or [],
    )

    graph = get_compiled_graph()
    final_state = graph.invoke(initial_state)
    return final_state
