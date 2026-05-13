"""Education tab — browse and search the financial knowledge base."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from src.web_app.components.disclaimer_banner import show_inline_disclaimer


_ARTICLES_DIR = Path(__file__).parent.parent.parent.parent / "src" / "data" / "articles"

_CATEGORY_LABELS = {
    "basics": "Basics",
    "investing": "Investing",
    "retirement": "Retirement",
    "taxes": "Taxes",
    "budgeting": "Budgeting",
    "market_concepts": "Market Concepts",
}


def _load_article_metadata() -> list[dict]:
    """Scan article directory and load frontmatter metadata."""
    articles = []
    for md_file in sorted(_ARTICLES_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    meta = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                    articles.append({
                        "title": meta.get("title", md_file.stem.replace("_", " ").title()),
                        "category": meta.get("category", md_file.parent.name),
                        "difficulty": meta.get("difficulty", "intermediate"),
                        "tags": meta.get("tags", []),
                        "body": body,
                        "path": str(md_file),
                    })
        except Exception:
            pass
    return articles


def render_education_tab() -> None:
    """Render the Education tab — knowledge base browser."""
    st.markdown("## Financial Education Library")
    st.markdown(
        f"Browse {len(list(_ARTICLES_DIR.rglob('*.md')))} educational articles "
        "curated to help you understand personal finance."
    )

    # Load articles
    articles = _load_article_metadata()

    # Filters
    col_cat, col_diff, col_search = st.columns([2, 2, 3])
    with col_cat:
        categories = ["All"] + sorted(_CATEGORY_LABELS.values())
        selected_cat = st.selectbox("Category", categories)

    with col_diff:
        difficulties = ["All", "Beginner", "Intermediate", "Advanced"]
        selected_diff = st.selectbox("Difficulty", difficulties)

    with col_search:
        search_term = st.text_input("Search articles…", placeholder="compound interest, IRA…")

    # Filter
    filtered = articles
    if selected_cat != "All":
        cat_key = next((k for k, v in _CATEGORY_LABELS.items() if v == selected_cat), None)
        if cat_key:
            filtered = [a for a in filtered if a["category"] == cat_key]

    if selected_diff != "All":
        filtered = [a for a in filtered if a["difficulty"].lower() == selected_diff.lower()]

    if search_term:
        term = search_term.lower()
        filtered = [
            a for a in filtered
            if term in a["title"].lower()
            or term in " ".join(a.get("tags", [])).lower()
            or term in a["body"].lower()[:500]
        ]

    st.markdown(f"**{len(filtered)} articles found**")
    st.markdown("---")

    if not filtered:
        st.info("No articles match your filters. Try broadening your search.")
        return

    # Article list
    for article in filtered:
        cat_label = _CATEGORY_LABELS.get(article["category"], article["category"])
        diff = article["difficulty"].title()
        diff_color = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}.get(diff, "⚪")

        with st.expander(
            f"{article['title']}  —  *{cat_label}* {diff_color} {diff}",
            expanded=False,
        ):
            tags = article.get("tags", [])
            if tags:
                st.caption("Tags: " + ", ".join(map(str, tags)))
            st.markdown(article["body"])
            show_inline_disclaimer("This article is for educational purposes only.")
