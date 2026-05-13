"""Markdown and number formatters for Finnie responses."""
from __future__ import annotations

from typing import List, Optional


def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """Format a float as a currency string: $1,234.56"""
    return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a float (0-100) as a percentage string: 12.34%"""
    return f"{value:.{decimals}f}%"


def format_large_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1_000_000_000:
        return f"{sign}{abs_val / 1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"{sign}{abs_val / 1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"{sign}{abs_val / 1_000:.2f}K"
    return f"{sign}{abs_val:.2f}"


def format_sources_markdown(sources: List[dict]) -> str:
    """Render a list of source dicts as a markdown 'Sources' section."""
    if not sources:
        return ""
    lines = ["\n---\n**Sources:**"]
    for i, src in enumerate(sources, 1):
        title = src.get("title", "Unknown")
        category = src.get("category", "")
        cat_tag = f" *({category})*" if category else ""
        lines.append(f"{i}. {title}{cat_tag}")
    return "\n".join(lines)


def format_follow_ups_markdown(questions: List[str]) -> str:
    """Render follow-up question suggestions as a markdown block."""
    if not questions:
        return ""
    lines = ["\n**You might also ask:**"]
    for q in questions:
        lines.append(f"- {q}")
    return "\n".join(lines)


def format_disclaimer_markdown(disclaimers: List[str]) -> str:
    """Render disclaimers as a small italic markdown block."""
    if not disclaimers:
        return ""
    combined = " | ".join(disclaimers)
    return f"\n\n> *{combined}*"
