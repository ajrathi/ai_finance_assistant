"""Market Analysis Agent — fetches live market data and provides educational context."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.base_agent import AgentOutput, BaseAgent, SourceAttribution
from src.utils.cache import get_market_cache

logger = logging.getLogger(__name__)

MARKET_DISCLAIMER = (
    "Market data shown is for informational purposes and may be delayed. "
    "Past performance does not indicate future results."
)

_SYSTEM_EXTRA = """
You specialize in market data education. You provide factual market information
and explain what the data means in an educational context.

Guidelines:
- Present market data clearly with timestamps
- Explain what the data means (e.g., what a sector decline implies)
- Contextualize data within broader economic concepts
- NEVER make predictions ("this stock will go up")
- NEVER recommend specific trades
- Always note that data is for informational purposes only
"""

_PROMPT_TEMPLATE = """\
Current market data (as of {timestamp}):
{market_data_text}

Educational context about market concepts:
{context}

User question: {query}

Provide an educational response that:
1. Addresses the user's question using the market data
2. Explains what the data means in plain terms
3. Connects it to relevant financial concepts
4. Notes any important caveats

End with 2-3 educational follow-ups prefixed with "Follow-up:"
"""


class MarketDataFetcher:
    """Fetches market data from Alpha Vantage with yFinance fallback."""

    def __init__(self, alpha_vantage_key: str = ""):
        self._av_key = alpha_vantage_key
        self._cache = get_market_cache()

    def _get_yfinance_quote(self, symbol: str) -> Optional[Dict]:
        """Fetch quote using yfinance (no API key required)."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            hist = ticker.history(period="2d")
            if hist.empty:
                return None
            latest = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else latest
            change = latest["Close"] - prev["Close"]
            change_pct = (change / prev["Close"] * 100) if prev["Close"] != 0 else 0
            return {
                "symbol": symbol,
                "price": round(float(latest["Close"]), 2),
                "change": round(float(change), 2),
                "change_pct": round(float(change_pct), 2),
                "volume": int(latest["Volume"]),
                "source": "yfinance",
            }
        except Exception as exc:
            logger.warning("yfinance failed for %s: %s", symbol, exc)
            return None

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get a stock/index quote, using cache and fallbacks."""
        cache_key = f"market:quote:{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        data = self._get_yfinance_quote(symbol)
        if data:
            self._cache.set(cache_key, data)
        return data

    def get_market_overview(self) -> Dict:
        """Get an overview of major indexes."""
        symbols = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "Dow Jones": "^DJI"}
        overview = {}
        for name, symbol in symbols.items():
            quote = self.get_quote(symbol)
            if quote:
                overview[name] = quote
        return overview


# Words that look like tickers but are finance/common abbreviations, not symbols
_NON_TICKERS = frozenset({
    # Two-letter words that aren't real tickers
    "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN",
    "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US",
    "WE", "OK",
    # Finance/business abbreviations that aren't tradeable tickers
    "ETF", "IPO", "CEO", "CFO", "COO", "CTO", "IRA", "USA", "GDP", "CPI",
    "APR", "APY", "ESG", "EPS", "ROI", "NAV", "AUM", "LLC", "INC", "LTD",
    "SEC", "FED", "IMF", "NYSE", "USD", "EUR", "GBP",
    "OTC", "FDIC", "SIPC",
})

# Match only tokens that are already ALL-CAPS in the original query (users type tickers in caps)
_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")


def _extract_tickers(query: str) -> List[str]:
    """Extract plausible stock ticker symbols from a query string.

    Only matches tokens that are already uppercase in the original text so that
    normal lowercase words ('the', 'what', etc.) are never treated as tickers.
    """
    candidates = _TICKER_RE.findall(query)  # operate on original, not .upper()
    return [t for t in dict.fromkeys(candidates) if t not in _NON_TICKERS]


def _format_market_data(data: Dict) -> str:
    """Format market data dict into readable text for LLM."""
    if not data:
        return "Market data temporarily unavailable."

    lines = []
    for name, quote in data.items():
        if isinstance(quote, dict) and "price" in quote:
            change_sign = "+" if quote.get("change", 0) >= 0 else ""
            lines.append(
                f"{name} ({quote.get('symbol', '')}): "
                f"${quote['price']:,.2f} "
                f"({change_sign}{quote.get('change', 0):.2f} / "
                f"{change_sign}{quote.get('change_pct', 0):.2f}%)"
            )
    return "\n".join(lines) if lines else "Market data temporarily unavailable."


class MarketAnalysisAgent(BaseAgent):
    """Live market data agent with educational context."""

    name = "Market Analysis Agent"

    def __init__(self, llm_client=None, rag_pipeline=None, data_fetcher=None):
        super().__init__(llm_client=llm_client)
        self._rag = rag_pipeline
        self._fetcher = data_fetcher

    def _get_fetcher(self) -> MarketDataFetcher:
        if self._fetcher is None:
            from src.core.config import settings
            self._fetcher = MarketDataFetcher(
                alpha_vantage_key=settings.apis.alpha_vantage.api_key
            )
        return self._fetcher

    def _get_rag(self):
        if self._rag is None:
            try:
                from src.rag.pipeline import get_rag_pipeline
                self._rag = get_rag_pipeline()
            except Exception:
                self._rag = None
        return self._rag

    def _retrieve_context(self, query: str) -> tuple[str, List[SourceAttribution]]:
        rag = self._get_rag()
        if rag is None:
            return "", []
        try:
            from src.core.config import settings
            categories = settings.agents.market.rag_categories
            result = rag.retrieve_and_format(query, categories=categories)
            return result.context_text, result.sources
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return "", []

    def _parse_follow_ups(self, text: str) -> tuple[str, List[str]]:
        lines = text.splitlines()
        follow_ups: List[str] = []
        body_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("follow-up:"):
                q = stripped[len("follow-up:"):].strip()
                if q:
                    follow_ups.append(q)
            else:
                body_lines.append(line)
        return "\n".join(body_lines).strip(), follow_ups

    def process(self, state: Any) -> AgentOutput:
        try:
            if isinstance(state, dict):
                query = state.get("current_query", "")
                user_profile = state.get("user_profile", {})
                market_data = state.get("market_data")
            else:
                query = getattr(state, "current_query", "")
                user_profile = getattr(state, "user_profile", {})
                market_data = getattr(state, "market_data", None)

            knowledge_level = (
                user_profile.get("knowledge_level", "intermediate")
                if isinstance(user_profile, dict)
                else getattr(user_profile, "knowledge_level", "intermediate")
            )

            # Fetch fresh market data if not already in state
            if not market_data:
                try:
                    market_data = self._get_fetcher().get_market_overview()
                except Exception as exc:
                    logger.warning("Market data fetch failed: %s", exc)
                    market_data = {}

            # Also fetch any individual tickers mentioned in the query
            tickers = _extract_tickers(query or "")
            if tickers:
                fetcher = self._get_fetcher()
                for ticker in tickers:
                    if ticker not in market_data:
                        quote = fetcher.get_quote(ticker)
                        if quote:
                            market_data[ticker] = quote

            market_data_text = _format_market_data(market_data)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

            context_text, sources = self._retrieve_context(
                query or "stock market index volatility"
            )

            system_prompt = self._build_system_prompt(knowledge_level, extra=_SYSTEM_EXTRA)
            user_prompt = _PROMPT_TEMPLATE.format(
                timestamp=timestamp,
                market_data_text=market_data_text,
                context=context_text or "No specific context retrieved.",
                query=query or "Provide a market overview.",
            )

            raw_response = self._llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
            )

            content, follow_ups = self._parse_follow_ups(raw_response)

            return AgentOutput(
                content=content,
                agent_name=self.name,
                confidence=0.85,
                sources=sources,
                disclaimers=[MARKET_DISCLAIMER],
                follow_up_questions=follow_ups[:3],
                metadata={"market_data": market_data, "timestamp": timestamp},
            )

        except Exception as exc:
            logger.error("MarketAnalysisAgent error: %s", exc, exc_info=True)
            return self._make_error_output(str(exc))
