"""Chat tab — the main conversational interface for Finnie."""
from __future__ import annotations

import streamlit as st

from src.web_app.session import (
    add_message,
    build_graph_input,
    get_messages,
)
from src.web_app.components.disclaimer_banner import show_disclaimer_banner
from src.web_app.components.source_attribution import show_sources


def _render_message(msg: dict) -> None:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    with st.chat_message(role):
        st.markdown(content)
        if sources and role == "assistant":
            show_sources(sources)


def _run_graph(query: str) -> dict:
    """Invoke the LangGraph pipeline and return the final state."""
    from src.workflow.graph import get_compiled_graph
    state_input = build_graph_input(query)
    graph = get_compiled_graph()
    return graph.invoke(state_input)


def render_chat_tab() -> None:
    """Render the full Chat tab UI."""
    show_disclaimer_banner("general")

    # Render chat history
    messages = get_messages()
    for msg in messages:
        _render_message(msg)

    # Chat input
    if prompt := st.chat_input("Ask Finnie anything about personal finance…"):
        # Display user message immediately
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run graph and stream response
        with st.chat_message("assistant"):
            with st.spinner("Finnie is thinking…"):
                try:
                    final_state = _run_graph(prompt)
                    response = final_state.get("final_response", "I couldn't generate a response.")
                    sources = final_state.get("sources", [])
                    st.markdown(response)
                    if sources:
                        show_sources(sources)
                    add_message("assistant", response, sources=sources)
                except Exception as exc:
                    error_msg = (
                        f"I encountered an issue: {exc}\n\n"
                        "Please check your API keys in the `.env` file and try again."
                    )
                    st.error(error_msg)
                    add_message("assistant", error_msg)

    # Suggested starter questions
    if not messages:
        st.markdown("---")
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        starters = [
            "What is compound interest?",
            "How do index funds work?",
            "What's the difference between a Roth and Traditional IRA?",
            "How much should I save for retirement?",
        ]
        for i, q in enumerate(starters):
            col = cols[i % 2]
            if col.button(q, key=f"starter_{i}", use_container_width=True):
                add_message("user", q)
                with st.chat_message("assistant"):
                    with st.spinner("Finnie is thinking…"):
                        try:
                            final_state = _run_graph(q)
                            response = final_state.get("final_response", "")
                            sources = final_state.get("sources", [])
                            st.markdown(response)
                            if sources:
                                show_sources(sources)
                            add_message("assistant", response, sources=sources)
                        except Exception as exc:
                            st.error(f"Error: {exc}")
                st.rerun()
