"""Source attribution display component for Streamlit."""
from __future__ import annotations

from typing import List

import streamlit as st


def show_sources(sources: List) -> None:
    """Render source attributions in a collapsed expander."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            title = src.title if hasattr(src, "title") else src.get("title", "Source")
            category = src.category if hasattr(src, "category") else src.get("category", "")
            score = src.relevance_score if hasattr(src, "relevance_score") else src.get("relevance_score", 0)
            excerpt = src.excerpt if hasattr(src, "excerpt") else src.get("excerpt", "")

            st.markdown(f"**{i}. {title}**" + (f" *({category})*" if category else ""))
            if excerpt:
                st.caption(excerpt)
            if score:
                st.progress(min(score, 1.0), text=f"Relevance: {score:.0%}")
            if i < len(sources):
                st.divider()
