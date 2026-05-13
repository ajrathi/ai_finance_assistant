"""Unit tests for src/web_app/session.py.

Streamlit session_state supports both attribute access (st.session_state.key)
and item access (st.session_state["key"]). The FakeSessionState helper below
emulates this dual-access pattern so no live Streamlit server is required.
"""
from __future__ import annotations

from typing import Dict, List
from unittest.mock import patch

import pytest


# ─── Streamlit session_state emulator ────────────────────────────────────────

class FakeSessionState:
    """Dict-backed mock that honours both attribute and item access.

    Streamlit uses ``st.session_state.key`` (attribute) AND
    ``st.session_state[key]`` (item) interchangeably.  This helper ensures
    both access patterns read/write the same underlying store so session.py
    works correctly under test.
    """

    def __init__(self, initial: Dict | None = None) -> None:
        object.__setattr__(self, "_store", dict(initial or {}))

    # ── item protocol ─────────────────────────────────────────────────────
    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_store")

    def __getitem__(self, key: str):
        return object.__getattribute__(self, "_store")[key]

    def __setitem__(self, key: str, value) -> None:
        object.__getattribute__(self, "_store")[key] = value

    def __delitem__(self, key: str) -> None:
        del object.__getattribute__(self, "_store")[key]

    # ── attribute protocol (mirrors item access) ──────────────────────────
    def __getattr__(self, key: str):
        store = object.__getattribute__(self, "_store")
        if key not in store:
            raise AttributeError(key)
        return store[key]

    def __setattr__(self, key: str, value) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            object.__getattribute__(self, "_store")[key] = value

    # ── dict-like helpers ─────────────────────────────────────────────────
    def get(self, key: str, default=None):
        return object.__getattribute__(self, "_store").get(key, default)

    def pop(self, key: str, *args):
        return object.__getattribute__(self, "_store").pop(key, *args)

    def keys(self):
        return object.__getattribute__(self, "_store").keys()

    def as_dict(self) -> Dict:
        """Return a copy of the underlying store (for assertions)."""
        return dict(object.__getattribute__(self, "_store"))


# ─── init_session ─────────────────────────────────────────────────────────────

class TestInitSession:
    def test_creates_all_expected_keys(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()

        store = fake.as_dict()
        expected = {
            "user_profile", "messages", "portfolio_holdings",
            "goals", "market_data", "onboarding_complete", "sources",
            "graph_initialized",
        }
        assert expected.issubset(store.keys())

    def test_does_not_overwrite_existing_values(self):
        fake = FakeSessionState({"onboarding_complete": True})
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
        assert fake["onboarding_complete"] is True

    def test_messages_initialised_to_empty_list(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
        assert fake["messages"] == []


# ─── add_message ──────────────────────────────────────────────────────────────

class TestAddMessage:
    def _run(self, *args, initial: Dict | None = None, **kwargs) -> FakeSessionState:
        fake = FakeSessionState(initial)
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            s.add_message(*args, **kwargs)
        return fake

    def test_adds_user_message(self):
        fake = self._run("user", "Hello!")
        assert fake["messages"][-1] == {"role": "user", "content": "Hello!"}

    def test_adds_assistant_message_with_sources(self):
        sources = [{"title": "Compound Interest", "category": "basics"}]
        fake = self._run("assistant", "Interest earns interest.", sources=sources)
        msg = fake["messages"][-1]
        assert msg["role"] == "assistant"
        assert msg["sources"] == sources

    def test_chart_type_and_data_stored(self):
        chart_data = {"holdings": [{"ticker": "AAPL", "value": 1000}]}
        fake = self._run("assistant", "Portfolio overview.",
                         chart_type="portfolio", chart_data=chart_data)
        msg = fake["messages"][-1]
        assert msg["chart_type"] == "portfolio"
        assert msg["chart_data"] == chart_data

    def test_follow_ups_stored(self):
        follow_ups = ["What is diversification?", "How do I rebalance?"]
        fake = self._run("assistant", "Great question.", follow_ups=follow_ups)
        assert fake["messages"][-1]["follow_ups"] == follow_ups

    def test_no_optional_fields_on_bare_message(self):
        """Bare messages must not carry extra keys."""
        fake = self._run("user", "Simple question.")
        msg = fake["messages"][-1]
        assert "sources" not in msg
        assert "chart_type" not in msg
        assert "chart_data" not in msg
        assert "follow_ups" not in msg

    def test_message_trimming(self):
        """Messages exceeding max_chat_history should be trimmed from the front."""
        fake = FakeSessionState()

        # Patch `settings` at its canonical location so the lazy import inside
        # add_message (`from src.core.config import settings`) picks it up.
        class _UIConf:
            max_chat_history = 3

        class _MockSettings:
            ui = _UIConf()

        with patch("streamlit.session_state", fake):
            with patch("src.core.config.settings", _MockSettings()):
                from src.web_app import session as s
                s.init_session()
                for i in range(5):
                    s.add_message("user", f"msg {i}")

        assert len(fake["messages"]) == 3
        assert fake["messages"][0]["content"] == "msg 2"
        assert fake["messages"][-1]["content"] == "msg 4"


# ─── get_conversation_history ─────────────────────────────────────────────────

class TestGetConversationHistory:
    def test_returns_role_content_dicts(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            fake["messages"] = [
                {"role": "user", "content": "What is an IRA?", "sources": []},
                {"role": "assistant", "content": "An IRA is…",
                 "sources": [{"title": "IRA guide"}], "chart_type": "goals"},
            ]
            history = s.get_conversation_history()

        assert history == [
            {"role": "user", "content": "What is an IRA?"},
            {"role": "assistant", "content": "An IRA is…"},
        ]

    def test_respects_context_window(self):
        """Only the last _HISTORY_CONTEXT_WINDOW messages should be included."""
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            fake["messages"] = [
                {"role": "user", "content": f"msg {i}"} for i in range(20)
            ]
            window = s._HISTORY_CONTEXT_WINDOW
            history = s.get_conversation_history()

        assert len(history) == window
        assert history[0]["content"] == f"msg {20 - window}"
        assert history[-1]["content"] == "msg 19"

    def test_empty_when_no_messages(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            history = s.get_conversation_history()
        assert history == []


# ─── build_graph_input ────────────────────────────────────────────────────────

class TestBuildGraphInput:
    def test_includes_conversation_history(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            fake["messages"] = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            graph_input = s.build_graph_input("What is a Roth IRA?")

        assert "conversation_history" in graph_input
        assert len(graph_input["conversation_history"]) == 2
        assert graph_input["conversation_history"][0]["role"] == "user"
        assert graph_input["conversation_history"][1]["role"] == "assistant"

    def test_current_query_set(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            graph_input = s.build_graph_input("Explain index funds.")
        assert graph_input["current_query"] == "Explain index funds."

    def test_portfolio_data_included_when_holdings_exist(self):
        holdings = [{"ticker": "AAPL", "shares": 10, "price_per_share": 190, "value": 1900}]
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            fake["portfolio_holdings"] = holdings
            graph_input = s.build_graph_input("Analyze my portfolio.")
        assert graph_input["portfolio_data"] == {"holdings": holdings}

    def test_portfolio_data_none_when_empty(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            graph_input = s.build_graph_input("What is diversification?")
        assert graph_input["portfolio_data"] is None

    def test_required_graph_keys_present(self):
        """All fields expected by AgentState must be present in graph input."""
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            graph_input = s.build_graph_input("test query")

        required_keys = {
            "current_query", "user_profile", "portfolio_data",
            "conversation_history", "messages", "intent", "active_agents",
            "agent_outputs", "retrieved_docs", "sources", "market_data",
            "iteration_count", "error_state", "final_response", "disclaimers",
        }
        assert required_keys.issubset(graph_input.keys())


# ─── get_portfolio / set_portfolio ────────────────────────────────────────────

class TestPortfolioSession:
    def test_set_and_get_portfolio(self):
        holdings = [{"ticker": "MSFT", "shares": 5, "price_per_share": 420, "value": 2100}]
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            s.set_portfolio(holdings)
            result = s.get_portfolio()
        assert result == holdings

    def test_set_portfolio_updates_user_profile_snapshot(self):
        holdings = [{"ticker": "MSFT", "shares": 5, "price_per_share": 420, "value": 2100}]
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            s.set_portfolio(holdings)
            profile = s.get_user_profile()
        assert profile.portfolio_snapshot == {"holdings": holdings}

    def test_empty_portfolio_returns_empty_list(self):
        fake = FakeSessionState()
        with patch("streamlit.session_state", fake):
            from src.web_app import session as s
            s.init_session()
            result = s.get_portfolio()
        assert result == []
