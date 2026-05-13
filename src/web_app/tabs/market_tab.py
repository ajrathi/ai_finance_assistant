"""Market tab — live market overview with educational context."""
from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

from src.web_app.components.charts import market_index_bar
from src.web_app.components.disclaimer_banner import show_disclaimer_banner, show_inline_disclaimer


def _fetch_market_overview() -> dict:
    """Fetch major index data with caching."""
    from src.agents.market_agent import MarketDataFetcher
    fetcher = MarketDataFetcher()
    return fetcher.get_market_overview()


def _fetch_quote(symbol: str) -> dict:
    from src.agents.market_agent import MarketDataFetcher
    fetcher = MarketDataFetcher()
    return fetcher.get_quote(symbol) or {}


def render_market_tab() -> None:
    """Render the Market Overview tab UI."""
    show_disclaimer_banner("market")

    st.markdown("## Market Overview")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Refresh", use_container_width=True):
            # Clear market cache to force refresh
            from src.utils.cache import get_market_cache
            get_market_cache().clear()
            st.rerun()

    # Major Indexes
    with st.spinner("Loading market data…"):
        try:
            market_data = _fetch_market_overview()
        except Exception:
            market_data = {}

    if market_data:
        st.markdown("### Major Indexes")
        cols = st.columns(len(market_data))
        for i, (name, quote) in enumerate(market_data.items()):
            with cols[i]:
                price = quote.get("price", 0)
                change = quote.get("change", 0)
                change_pct = quote.get("change_pct", 0)
                delta_str = f"{change:+.2f} ({change_pct:+.2f}%)"
                st.metric(
                    label=name,
                    value=f"${price:,.2f}",
                    delta=delta_str,
                )

        st.plotly_chart(market_index_bar(market_data), use_container_width=True)
        show_inline_disclaimer(
            "Data may be delayed 15–20 minutes. For informational purposes only."
        )
    else:
        st.warning(
            "Market data is temporarily unavailable. "
            "Check your API keys or try again later."
        )

    st.markdown("---")

    # Stock Quote Lookup
    st.markdown("### Stock Quote Lookup")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input(
            "Enter ticker symbol",
            placeholder="AAPL, MSFT, GOOGL…",
            label_visibility="collapsed",
        ).upper().strip()
    with col_btn:
        lookup_btn = st.button("Look Up", use_container_width=True)

    if lookup_btn and ticker_input:
        with st.spinner(f"Fetching {ticker_input}…"):
            try:
                quote = _fetch_quote(ticker_input)
                if quote:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Price", f"${quote.get('price', 0):,.2f}")
                    change = quote.get("change", 0)
                    change_pct = quote.get("change_pct", 0)
                    c2.metric("Change", f"{change:+.2f}", f"{change_pct:+.2f}%")
                    c3.metric("Volume", f"{quote.get('volume', 0):,}")
                    show_inline_disclaimer("Data may be delayed. Not investment advice.")
                else:
                    st.error(f"Could not find data for '{ticker_input}'. Check the symbol.")
            except Exception as exc:
                st.error(f"Error fetching data: {exc}")

    st.markdown("---")

    # Educational Context
    st.markdown("### Understanding Market Data")
    with st.expander("What do these numbers mean?", expanded=False):
        st.markdown("""
**Index Value**: The S&P 500, NASDAQ, and Dow Jones are indexes that track baskets of stocks.
They're benchmarks for overall market performance.

**Daily Change**: How much the index moved today vs. yesterday's close.

**Green (positive change)**: The market or stock is up today vs. prior close.
**Red (negative change)**: The market or stock is down today vs. prior close.

**Important**: Day-to-day market movements are normal and expected.
Short-term volatility does not indicate the long-term direction of the market.

*For educational purposes only. Past performance does not indicate future results.*
        """)
