"""Sidebar component for the Finnie chat interface.

Consolidates all secondary UI widgets (portfolio upload, goal setup, market
overview, profile, system status) into a single collapsible sidebar, freeing
the main area for the unified chat interface.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.config import settings
from src.web_app.session import (
    add_goal,
    clear_goals,
    get_goals,
    get_portfolio,
    get_user_profile,
    set_portfolio,
    set_user_profile,
)
from src.utils.validators import validate_goal_inputs, validate_portfolio


# ─── CSV sample used for the download helper ─────────────────────────────────

_SAMPLE_CSV = (
    "ticker,shares,price_per_share\n"
    "AAPL,10,190.00\n"
    "MSFT,5,420.00\n"
    "BND,50,72.00\n"
    "VTI,20,240.00\n"
)


# ─── Portfolio utilities (moved from portfolio_tab.py) ────────────────────────

def _parse_csv(uploaded_file) -> List[Dict]:
    """Parse an uploaded CSV file into a holdings list."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()
    holdings = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        shares = float(row.get("shares", row.get("quantity", 0)))
        price = float(row.get("price_per_share", row.get("price", 0)))
        value = shares * price
        holdings.append({
            "ticker": ticker,
            "shares": shares,
            "price_per_share": price,
            "value": value,
        })
    return holdings


def _enrich_with_live_prices(holdings: List[Dict]) -> List[Dict]:
    """Fetch live prices from yFinance for holdings that have no price data."""
    enriched = []
    for h in holdings:
        if h.get("price_per_share", 0) == 0:
            try:
                import yfinance as yf
                info = yf.Ticker(h["ticker"]).fast_info
                price = round(float(info.last_price or 0), 2)
                h = {**h, "price_per_share": price, "value": h["shares"] * price}
            except Exception:
                pass  # Leave price at 0; agent will note missing data
        enriched.append(h)
    return enriched


# ─── Sidebar sections ─────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown("# Finnie 💰")
    st.markdown("*AI Finance Assistant*")


def _render_profile_section() -> None:
    """Show current profile and allow inline edits."""
    profile = get_user_profile()

    st.markdown("**Your Profile**")
    col1, col2 = st.columns(2)
    col1.metric("Knowledge", profile.knowledge_level.title())
    col2.metric("Risk", profile.risk_tolerance.title())

    with st.expander("Edit Profile"):
        new_level = st.select_slider(
            "Knowledge Level",
            options=["beginner", "intermediate", "advanced"],
            value=profile.knowledge_level,
            key="sidebar_knowledge",
        )
        new_risk = st.select_slider(
            "Risk Tolerance",
            options=["conservative", "moderate", "aggressive"],
            value=profile.risk_tolerance,
            key="sidebar_risk",
        )
        if st.button("Update Profile", use_container_width=True, key="update_profile_btn"):
            profile.knowledge_level = new_level
            profile.risk_tolerance = new_risk
            set_user_profile(profile)
            st.success("Profile updated!")
            st.rerun()


def _render_portfolio_section() -> None:
    """Portfolio upload and holdings summary."""
    st.markdown("#### Portfolio")

    tab_csv, tab_manual = st.tabs(["CSV Upload", "Manual"])

    with tab_csv:
        st.download_button(
            "Download sample CSV",
            data=_SAMPLE_CSV,
            file_name="sample_portfolio.csv",
            mime="text/csv",
            use_container_width=True,
        )
        uploaded = st.file_uploader(
            "Upload portfolio CSV",
            type=["csv"],
            help="Columns: ticker, shares, price_per_share",
            key="portfolio_csv_uploader",
        )
        if uploaded:
            try:
                holdings = _parse_csv(uploaded)
                with st.spinner("Fetching live prices…"):
                    holdings = _enrich_with_live_prices(holdings)
                ok, errors = validate_portfolio(holdings)
                if ok:
                    set_portfolio(holdings)
                    st.success(f"Loaded {len(holdings)} holdings.")
                else:
                    for err in errors:
                        st.error(err)
            except Exception as exc:
                st.error(f"Error parsing CSV: {exc}")

    with tab_manual:
        with st.form("sidebar_manual_holding_form", clear_on_submit=True):
            ticker = st.text_input("Ticker", placeholder="AAPL").upper().strip()
            shares = st.number_input("Shares", min_value=0.001, step=0.001, value=1.0)
            price = st.number_input(
                "Price ($)",
                min_value=0.0, step=0.01, value=0.0,
                help="0 = auto-fetch",
            )
            if st.form_submit_button("Add Holding", use_container_width=True) and ticker:
                if price == 0:
                    try:
                        import yfinance as yf
                        price = round(float(yf.Ticker(ticker).fast_info.last_price or 0), 2)
                    except Exception:
                        st.warning(f"Could not auto-fetch price for {ticker}.")
                holding = {
                    "ticker": ticker,
                    "shares": shares,
                    "price_per_share": price,
                    "value": shares * price,
                }
                current = get_portfolio()
                existing = [h["ticker"] for h in current]
                if ticker in existing:
                    idx = existing.index(ticker)
                    current[idx] = holding
                else:
                    current.append(holding)
                set_portfolio(current)
                st.success(f"Added {ticker}")
                st.rerun()

    # Holdings summary
    holdings = get_portfolio()
    if holdings:
        total = sum(h.get("value", 0) for h in holdings)
        st.metric("Portfolio Value", f"${total:,.2f}")
        st.caption(f"{len(holdings)} holdings — ask Finnie to analyze your portfolio")
        if st.button("Clear Portfolio", type="secondary", use_container_width=True,
                     key="clear_portfolio_btn"):
            set_portfolio([])
            st.rerun()
    else:
        st.caption("No holdings yet.")


def _render_goal_section() -> None:
    """Compact goal setup form."""
    st.markdown("#### Financial Goal")

    goals = get_goals()
    if goals:
        goal = goals[0]
        st.success(f"**{goal['name']}** — ${goal['target_amount']:,.0f} in "
                   f"{goal['time_horizon_years']} yrs")
        st.caption("Ask Finnie to discuss your goal in chat.")
        if st.button("Clear Goal", type="secondary", use_container_width=True,
                     key="clear_goal_btn"):
            clear_goals()
            st.rerun()
    else:
        with st.form("sidebar_goal_form", clear_on_submit=False):
            goal_name = st.text_input("Goal Name", placeholder="Retirement…")
            target = st.number_input(
                "Target ($)", min_value=1_000.0, max_value=50_000_000.0,
                value=500_000.0, step=10_000.0, format="%.0f",
            )
            horizon = st.slider("Years", min_value=1, max_value=50, value=20)
            savings = st.number_input(
                "Current Savings ($)", min_value=0.0, max_value=10_000_000.0,
                value=0.0, step=1_000.0, format="%.0f",
            )
            monthly = st.number_input(
                "Monthly ($)", min_value=0.0, max_value=100_000.0,
                value=500.0, step=50.0, format="%.0f",
            )
            if st.form_submit_button("Save Goal", type="primary", use_container_width=True):
                ok, err = validate_goal_inputs(target, horizon, savings, monthly)
                if not ok:
                    st.error(err)
                else:
                    clear_goals()
                    add_goal({
                        "name": goal_name or "My Goal",
                        "target_amount": target,
                        "time_horizon_years": horizon,
                        "current_savings": savings,
                        "monthly_contribution": monthly,
                    })
                    st.success("Goal saved!")
                    st.rerun()


def _render_market_section() -> None:
    """Quick market overview with refresh button."""
    st.markdown("#### Market Overview")
    col_refresh, _ = st.columns([1, 2])
    with col_refresh:
        if st.button("Refresh", key="market_refresh_btn", use_container_width=True):
            from src.utils.cache import get_market_cache
            get_market_cache().clear()
            st.rerun()

    try:
        from src.agents.market_agent import MarketDataFetcher
        market_data = MarketDataFetcher().get_market_overview()
        if market_data:
            for name, quote in market_data.items():
                price = quote.get("price", 0)
                change_pct = quote.get("change_pct", 0)
                st.metric(
                    label=name,
                    value=f"${price:,.2f}",
                    delta=f"{change_pct:+.2f}%",
                )
            st.caption("*Data may be delayed 15–20 min.*")
            st.caption("Ask Finnie for a full market analysis in chat.")
        else:
            st.warning("Market data unavailable.")
    except Exception:
        st.warning("Market data unavailable.")


def _render_system_status() -> None:
    """Show API key and model status."""
    st.markdown("**System**")
    google_key = settings.google_api_key
    av_key = settings.apis.alpha_vantage.api_key
    st.markdown(f"🤖 Gemini: {'✅' if google_key else '❌ Set GOOGLE_API_KEY'}")
    st.markdown(f"📈 Alpha Vantage: {'✅' if av_key else '⚠️ yFinance fallback'}")


# ─── Public entry point ───────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the complete sidebar for the chat interface."""
    with st.sidebar:
        _render_header()
        st.divider()

        _render_profile_section()
        st.divider()

        with st.expander("📊 Portfolio", expanded=True):
            _render_portfolio_section()

        with st.expander("🎯 Goals", expanded=False):
            _render_goal_section()

        with st.expander("📈 Market", expanded=False):
            _render_market_section()

        st.divider()
        _render_system_status()
        st.divider()
        st.caption(
            "**Disclaimer:** Finnie provides financial education, not financial advice. "
            "Always consult qualified professionals."
        )
