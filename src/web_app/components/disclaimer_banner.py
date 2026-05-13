"""Reusable disclaimer banner component for Streamlit."""
from __future__ import annotations

import streamlit as st


def show_disclaimer_banner(level: str = "general") -> None:
    """Render a styled disclaimer banner.

    level: 'general' | 'portfolio' | 'tax' | 'market' | 'info'
    """
    messages = {
        "general": (
            "**Educational Use Only** — Finnie provides financial education, "
            "not personalized financial advice. Always consult qualified professionals "
            "before making financial decisions."
        ),
        "portfolio": (
            "**Portfolio Analysis Notice** — This analysis is illustrative and educational. "
            "Consult a registered investment advisor before making investment decisions."
        ),
        "tax": (
            "**Tax Information Notice** — Tax information is educational. "
            "Consult a qualified tax professional for advice specific to your situation."
        ),
        "market": (
            "**Market Data Notice** — Data is for informational purposes and may be delayed. "
            "Past performance does not indicate future results."
        ),
        "info": "For informational and educational purposes only.",
    }
    msg = messages.get(level, messages["general"])
    st.info(msg, icon="ℹ️")


def show_inline_disclaimer(text: str) -> None:
    """Render a small inline disclaimer in gray text."""
    st.caption(f"*{text}*")
