"""Portfolio tab — upload, enter, and analyze your portfolio."""
from __future__ import annotations

import io
from typing import List, Dict

import pandas as pd
import streamlit as st

from src.web_app.session import (
    add_message,
    build_graph_input,
    get_portfolio,
    set_portfolio,
)
from src.web_app.components.charts import portfolio_pie_chart, portfolio_bar_chart
from src.web_app.components.disclaimer_banner import show_disclaimer_banner, show_inline_disclaimer
from src.web_app.components.source_attribution import show_sources
from src.utils.validators import validate_portfolio


_SAMPLE_CSV = "ticker,shares,price_per_share\nAAPL,10,190.00\nMSFT,5,420.00\nBND,50,72.00\nVTI,20,240.00\n"


def _parse_csv(uploaded_file) -> List[Dict]:
    """Parse uploaded CSV into holdings list."""
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
    """Attempt to fetch live prices for holdings missing price data."""
    enriched = []
    for h in holdings:
        if h.get("price_per_share", 0) == 0:
            try:
                import yfinance as yf
                ticker = yf.Ticker(h["ticker"])
                price = ticker.fast_info.last_price or 0
                h["price_per_share"] = round(float(price), 2)
                h["value"] = h["shares"] * h["price_per_share"]
            except Exception:
                pass
        enriched.append(h)
    return enriched


def render_portfolio_tab() -> None:
    """Render the Portfolio tab UI."""
    show_disclaimer_banner("portfolio")

    st.markdown("## Portfolio Analysis")
    st.markdown(
        "Enter your portfolio below to see an educational analysis of your allocation, "
        "diversification, and how it relates to your stated risk tolerance."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Add Holdings")
        tab_csv, tab_manual = st.tabs(["CSV Upload", "Manual Entry"])

        with tab_csv:
            st.download_button(
                "Download sample CSV",
                data=_SAMPLE_CSV,
                file_name="sample_portfolio.csv",
                mime="text/csv",
            )
            uploaded = st.file_uploader(
                "Upload your portfolio CSV",
                type=["csv"],
                help="Columns: ticker, shares, price_per_share",
            )
            if uploaded:
                try:
                    holdings = _parse_csv(uploaded)
                    with st.spinner("Fetching live prices where needed…"):
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
            st.markdown("Add holdings manually:")
            with st.form("manual_holding_form"):
                ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper().strip()
                shares = st.number_input("Number of Shares", min_value=0.001, step=0.001, value=1.0)
                price = st.number_input("Price per Share ($)", min_value=0.0, step=0.01, value=0.0,
                                        help="Leave at 0 to auto-fetch live price")
                add_btn = st.form_submit_button("Add Holding")

                if add_btn and ticker:
                    if price == 0:
                        try:
                            import yfinance as yf
                            info = yf.Ticker(ticker).fast_info
                            price = round(float(info.last_price or 0), 2)
                        except Exception:
                            st.warning(f"Could not fetch price for {ticker}. Enter manually.")
                    holding = {
                        "ticker": ticker,
                        "shares": shares,
                        "price_per_share": price,
                        "value": shares * price,
                    }
                    current = get_portfolio()
                    # Update or add
                    existing_tickers = [h["ticker"] for h in current]
                    if ticker in existing_tickers:
                        idx = existing_tickers.index(ticker)
                        current[idx] = holding
                    else:
                        current.append(holding)
                    set_portfolio(current)
                    st.success(f"Added {ticker}")
                    st.rerun()

    holdings = get_portfolio()

    with col_right:
        if holdings:
            st.markdown("### Current Holdings")
            df = pd.DataFrame(holdings)
            total = df["value"].sum()
            df["% Allocation"] = (df["value"] / total * 100).round(2)
            st.dataframe(
                df[["ticker", "shares", "price_per_share", "value", "% Allocation"]].rename(columns={
                    "ticker": "Ticker",
                    "shares": "Shares",
                    "price_per_share": "Price ($)",
                    "value": "Value ($)",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.metric("Total Portfolio Value", f"${total:,.2f}")

            if st.button("Clear Portfolio", type="secondary"):
                set_portfolio([])
                st.rerun()
        else:
            st.info("No holdings yet. Upload a CSV or add manually.")

    # Visualizations
    if holdings:
        st.markdown("---")
        st.markdown("### Visualizations")
        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.plotly_chart(portfolio_pie_chart(holdings), use_container_width=True)
        with col_bar:
            st.plotly_chart(portfolio_bar_chart(holdings), use_container_width=True)

        show_inline_disclaimer(
            "Charts are for educational illustration only. Portfolio data is not stored or shared."
        )

        # Portfolio analysis via agent
        st.markdown("---")
        if st.button("Analyze My Portfolio", type="primary", use_container_width=True):
            with st.spinner("Analyzing portfolio…"):
                try:
                    from src.workflow.graph import get_compiled_graph
                    from src.web_app.session import build_graph_input, add_message
                    state_input = build_graph_input("Please analyze my portfolio and explain my allocation.")
                    state_input["intent"] = "portfolio_review"
                    graph = get_compiled_graph()
                    final_state = graph.invoke(state_input)
                    response = final_state.get("final_response", "")
                    sources = final_state.get("sources", [])
                    st.markdown("### Portfolio Analysis")
                    st.markdown(response)
                    if sources:
                        show_sources(sources)
                    add_message("user", "Please analyze my portfolio.")
                    add_message("assistant", response, sources=sources)
                except Exception as exc:
                    st.error(f"Analysis error: {exc}")
