"""Main Streamlit entry point for Finnie — AI Finance Assistant.

Single unified chat interface. All agent routing is handled by the LangGraph
orchestrator — there is no intent-specific logic in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on the path when running via streamlit
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.core.config import settings
from src.utils.logging_config import setup_logging
from src.web_app.session import (
    add_message,
    build_graph_input,
    get_goals,
    get_messages,
    get_portfolio,
    get_user_profile,
    init_session,
    set_user_profile,
)
from src.web_app.components.context_sidebar import render_sidebar
from src.web_app.components.disclaimer_banner import show_disclaimer_banner
from src.web_app.components.source_attribution import show_sources
from src.workflow.state import UserProfile

setup_logging(settings.app.log_level)

st.set_page_config(
    page_title=settings.ui.page_title,
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Starter prompts shown when conversation is empty ─────────────────────────

_STARTER_QUESTIONS = [
    "What is compound interest and why does it matter?",
    "How do index funds work?",
    "What's the difference between a Roth and Traditional IRA?",
    "How should I think about building an emergency fund?",
    "Can you explain dollar-cost averaging?",
    "What should I know about asset allocation?",
]


# ─── Onboarding ───────────────────────────────────────────────────────────────

def _render_onboarding() -> None:
    """Full-page onboarding form shown on first visit."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## Welcome to Finnie! 👋")
        st.markdown(
            "I'm your AI financial education assistant. "
            "Tell me a little about yourself so I can tailor my responses."
        )
        st.info(
            "**Remember:** Finnie provides financial *education*, not personalized "
            "financial advice. Always consult qualified professionals before making "
            "financial decisions.",
            icon="ℹ️",
        )
        with st.form("onboarding_form"):
            knowledge_level = st.select_slider(
                "Your financial knowledge level:",
                options=["beginner", "intermediate", "advanced"],
                value="beginner",
                help="Finnie adapts its language to your level.",
            )
            risk_tolerance = st.select_slider(
                "Your investment risk tolerance:",
                options=["conservative", "moderate", "aggressive"],
                value="moderate",
                help="Helps Finnie contextualise portfolio and goal discussions.",
            )
            primary_goal = st.selectbox(
                "Your primary financial goal:",
                [
                    "Build an emergency fund",
                    "Pay off debt",
                    "Start investing",
                    "Save for retirement",
                    "Save for a major purchase",
                    "Learn about personal finance",
                    "Other",
                ],
            )
            if st.form_submit_button("Let's Get Started!", type="primary"):
                profile = UserProfile(
                    knowledge_level=knowledge_level,
                    risk_tolerance=risk_tolerance,
                    goals=[{
                        "name": primary_goal,
                        "target_amount": 0,
                        "time_horizon_years": 10,
                        "current_savings": 0,
                        "monthly_contribution": 0,
                    }],
                )
                set_user_profile(profile)
                st.session_state.onboarding_complete = True
                st.rerun()


# ─── Graph invocation ─────────────────────────────────────────────────────────

def _run_graph(query: str) -> dict:
    """Invoke the LangGraph orchestrator and return the final state."""
    from src.workflow.graph import get_compiled_graph
    state_input = build_graph_input(query)
    return get_compiled_graph().invoke(state_input)


# ─── Chart data extraction ────────────────────────────────────────────────────

def _extract_chart_data(intent: str, final_state: dict) -> Tuple[Optional[str], Optional[Dict]]:
    """Determine whether to render a chart alongside the agent response.

    Returns a ``(chart_type, chart_data)`` tuple. ``chart_type`` is one of
    ``"portfolio"``, ``"goals"``, ``"market"``, or ``None``.
    """
    from src.workflow.intent_labels import Intent

    if intent == Intent.PORTFOLIO_REVIEW.value:
        holdings = get_portfolio()
        if holdings:
            return "portfolio", {"holdings": holdings}

    elif intent == Intent.GOAL_PLANNING.value:
        goals = get_goals()
        if goals:
            return "goals", goals[0]

    elif intent == Intent.MARKET_DATA.value:
        market_data = final_state.get("market_data")
        if market_data:
            return "market", market_data

    return None, None


def _extract_follow_ups(final_state: dict) -> List[str]:
    """Pull follow-up questions from whichever agent responded."""
    for output in final_state.get("agent_outputs", {}).values():
        questions = getattr(output, "follow_up_questions", [])
        if questions:
            return questions[:3]
    return []


# ─── Chart rendering ──────────────────────────────────────────────────────────

def _render_chart(chart_type: str, chart_data: Dict) -> None:
    """Render an inline chart below the assistant message bubble.

    Charts are rendered outside the ``st.chat_message`` context so Streamlit
    has the full column width available.
    """
    from src.web_app.components.charts import (
        goal_projection_chart,
        market_index_bar,
        portfolio_bar_chart,
        portfolio_pie_chart,
    )

    if chart_type == "portfolio":
        holdings = chart_data.get("holdings", [])
        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.plotly_chart(portfolio_pie_chart(holdings), use_container_width=True)
        with col_bar:
            st.plotly_chart(portfolio_bar_chart(holdings), use_container_width=True)
        st.caption(
            "*Charts are for educational illustration only. "
            "Portfolio data is not stored or shared externally.*"
        )

    elif chart_type == "goals":
        fig = goal_projection_chart(
            current_savings=chart_data.get("current_savings", 0),
            monthly_contribution=chart_data.get("monthly_contribution", 0),
            time_horizon_years=chart_data.get("time_horizon_years", 10),
            target_amount=chart_data.get("target_amount", 0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "*Projections use compound interest with constant assumed returns. "
            "Actual results vary and are not guaranteed.*"
        )

    elif chart_type == "market":
        st.plotly_chart(market_index_bar(chart_data), use_container_width=True)
        st.caption("*Data may be delayed 15–20 minutes. For informational purposes only.*")


# ─── Message rendering ────────────────────────────────────────────────────────

def _render_message(msg: Dict) -> None:
    """Render a single stored message, including any attached chart."""
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            show_sources(msg["sources"])

    # Charts live outside the bubble for full-width rendering
    if role == "assistant" and msg.get("chart_type") and msg.get("chart_data"):
        _render_chart(msg["chart_type"], msg["chart_data"])


def _render_follow_up_buttons(follow_ups: List[str], key_prefix: str) -> Optional[str]:
    """Render follow-up question buttons and return clicked text, if any."""
    if not follow_ups:
        return None
    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(min(len(follow_ups), 2))
    for i, question in enumerate(follow_ups):
        if cols[i % 2].button(question, key=f"{key_prefix}_{i}", use_container_width=True):
            return question
    return None


def _render_starter_buttons() -> Optional[str]:
    """Show example questions when the conversation is empty."""
    st.markdown("---")
    st.markdown("### Not sure where to start? Try asking:")
    cols = st.columns(2)
    for i, question in enumerate(_STARTER_QUESTIONS):
        if cols[i % 2].button(question, key=f"starter_{i}", use_container_width=True):
            return question
    return None


# ─── Query processing ─────────────────────────────────────────────────────────

def _process_query(query: str) -> None:
    """Run a query through the orchestrator and render the response inline.

    Adds both the user and assistant messages to session state so they appear
    in subsequent history renders. Displays a spinner during inference.
    """
    # Persist and render user message
    add_message("user", query)
    with st.chat_message("user"):
        st.markdown(query)

    # Invoke orchestrator and render assistant response
    chart_type: Optional[str] = None
    chart_data: Optional[Dict] = None

    with st.chat_message("assistant"):
        with st.spinner("Finnie is thinking…"):
            try:
                final_state = _run_graph(query)
                response = final_state.get("final_response", "I couldn't generate a response.")
                sources = final_state.get("sources", [])
                intent = final_state.get("intent", "")
                chart_type, chart_data = _extract_chart_data(intent, final_state)
                follow_ups = _extract_follow_ups(final_state)

                st.markdown(response)
                if sources:
                    show_sources(sources)

                add_message(
                    "assistant",
                    response,
                    sources=sources,
                    chart_type=chart_type,
                    chart_data=chart_data,
                    follow_ups=follow_ups,
                )

            except Exception as exc:
                error_msg = (
                    f"I encountered an issue: {exc}\n\n"
                    "Please check your API keys in `.env` and try again."
                )
                st.error(error_msg)
                add_message("assistant", error_msg)

    # Render chart outside the bubble (full width)
    if chart_type and chart_data:
        _render_chart(chart_type, chart_data)


# ─── Main application ─────────────────────────────────────────────────────────

def main() -> None:
    """Application entry point: unified chat interface."""
    init_session()

    # Show onboarding for first-time visitors
    if not st.session_state.get("onboarding_complete", False):
        _render_onboarding()
        return

    # Sidebar: profile, portfolio, goals, market, system status
    render_sidebar()

    # Page header
    st.markdown("## 💬 Finnie — AI Finance Assistant")
    show_disclaimer_banner("general")

    messages = get_messages()

    # Render conversation history (all prior messages + their charts)
    for msg in messages:
        _render_message(msg)

    # Interactive suggestions: follow-ups after last response, or starters
    pending_query: Optional[str] = None

    if messages and messages[-1]["role"] == "assistant":
        follow_ups = messages[-1].get("follow_ups", [])
        clicked = _render_follow_up_buttons(follow_ups, key_prefix="fu")
        if clicked:
            pending_query = clicked
    elif not messages:
        clicked = _render_starter_buttons()
        if clicked:
            pending_query = clicked

    # Primary chat input (Streamlit pins this to the bottom of the viewport)
    if prompt := st.chat_input("Ask Finnie anything about personal finance…"):
        pending_query = prompt

    # Process whichever query arrived (text input or button click)
    if pending_query:
        _process_query(pending_query)
        st.rerun()


if __name__ == "__main__":
    main()
