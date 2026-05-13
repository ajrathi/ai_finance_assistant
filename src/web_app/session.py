"""Session state management for the Finnie Streamlit app."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from src.workflow.state import UserProfile

# Number of recent messages passed to the orchestrator as conversation context.
_HISTORY_CONTEXT_WINDOW = 10


def init_session() -> None:
    """Initialize all session state keys on first load."""
    defaults: Dict[str, Any] = {
        "user_profile": UserProfile(),
        "messages": [],          # List[Dict] — {role, content, sources?, chart_type?, chart_data?, follow_ups?}
        "portfolio_holdings": [], # List[Dict] from CSV or manual entry
        "goals": [],             # List[Dict] goal objects
        "market_data": None,
        "onboarding_complete": False,
        "sources": [],
        "graph_initialized": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_user_profile() -> UserProfile:
    init_session()
    return st.session_state.user_profile


def set_user_profile(profile: UserProfile) -> None:
    st.session_state.user_profile = profile


def get_messages() -> List[Dict]:
    init_session()
    return st.session_state.messages


def add_message(
    role: str,
    content: str,
    sources: Optional[List] = None,
    chart_type: Optional[str] = None,
    chart_data: Optional[Dict] = None,
    follow_ups: Optional[List[str]] = None,
) -> None:
    """Append a message to the conversation history.

    Args:
        role: ``"user"`` or ``"assistant"``.
        content: Markdown-formatted message body.
        sources: List of SourceAttribution objects or dicts (assistant only).
        chart_type: One of ``"portfolio"``, ``"goals"``, ``"market"`` — triggers
            inline chart rendering after the message bubble.
        chart_data: Serialisable dict that the chart renderer will consume.
        follow_ups: Suggested follow-up question strings (assistant only).
    """
    init_session()
    msg: Dict[str, Any] = {"role": role, "content": content}
    if sources:
        msg["sources"] = sources
    if chart_type:
        msg["chart_type"] = chart_type
    if chart_data is not None:
        msg["chart_data"] = chart_data
    if follow_ups:
        msg["follow_ups"] = follow_ups

    st.session_state.messages.append(msg)

    # Trim to configured maximum
    from src.core.config import settings
    max_history = settings.ui.max_chat_history
    if len(st.session_state.messages) > max_history:
        st.session_state.messages = st.session_state.messages[-max_history:]


def get_conversation_history() -> List[Dict[str, str]]:
    """Return recent messages as plain role/content dicts for graph context.

    Only includes the last ``_HISTORY_CONTEXT_WINDOW`` messages and strips
    UI-only fields (sources, chart data, follow-ups) so the payload is clean.
    """
    messages = get_messages()
    recent = messages[-_HISTORY_CONTEXT_WINDOW:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def get_portfolio() -> List[Dict]:
    init_session()
    return st.session_state.portfolio_holdings


def set_portfolio(holdings: List[Dict]) -> None:
    st.session_state.portfolio_holdings = holdings
    # Mirror into user profile for orchestrator access
    profile = get_user_profile()
    profile.portfolio_snapshot = {"holdings": holdings}


def get_goals() -> List[Dict]:
    init_session()
    return st.session_state.goals


def add_goal(goal: Dict) -> None:
    init_session()
    st.session_state.goals.append(goal)
    profile = get_user_profile()
    profile.goals = st.session_state.goals


def clear_goals() -> None:
    st.session_state.goals = []
    profile = get_user_profile()
    profile.goals = []


def build_graph_input(query: str) -> Dict:
    """Assemble the AgentState-compatible dict from current session state.

    Includes the user profile, portfolio snapshot, and recent conversation
    history so the orchestrator has full context for each request.
    """
    profile = get_user_profile()
    portfolio = get_portfolio()
    portfolio_data = {"holdings": portfolio} if portfolio else None
    conversation_history = get_conversation_history()

    return {
        "current_query": query,
        "user_profile": profile,
        "portfolio_data": portfolio_data,
        "conversation_history": conversation_history,
        "messages": [],
        "intent": "",
        "active_agents": [],
        "agent_outputs": {},
        "retrieved_docs": [],
        "sources": [],
        "market_data": None,
        "iteration_count": 0,
        "error_state": None,
        "final_response": None,
        "disclaimers": [],
    }
