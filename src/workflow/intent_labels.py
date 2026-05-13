"""Intent taxonomy for the Finnie workflow router."""
from enum import Enum


class Intent(str, Enum):
    GENERAL_FINANCE = "general_finance"
    PORTFOLIO_REVIEW = "portfolio_review"
    MARKET_DATA = "market_data"
    GOAL_PLANNING = "goal_planning"
    NEWS_SUMMARY = "news_summary"
    TAX_EDUCATION = "tax_education"
    MULTI_DOMAIN = "multi_domain"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"


# Map intents to agent node names
INTENT_TO_AGENT: dict[Intent, list[str]] = {
    Intent.GENERAL_FINANCE: ["finance_qa_node"],
    Intent.PORTFOLIO_REVIEW: ["portfolio_node"],
    Intent.MARKET_DATA: ["market_node"],
    Intent.GOAL_PLANNING: ["goal_planning_node"],
    Intent.NEWS_SUMMARY: ["news_node"],
    Intent.TAX_EDUCATION: ["tax_node"],
    Intent.MULTI_DOMAIN: ["finance_qa_node"],   # synthesizer handles multi
    Intent.GREETING: ["finance_qa_node"],
    Intent.OUT_OF_SCOPE: ["finance_qa_node"],
}

# Keyword triggers for fast classification (no LLM call needed)
KEYWORD_MAP: dict[str, Intent] = {
    # Greetings
    "hello": Intent.GREETING,
    "hi": Intent.GREETING,
    "hey": Intent.GREETING,
    "good morning": Intent.GREETING,
    "good afternoon": Intent.GREETING,
    # Portfolio
    "portfolio": Intent.PORTFOLIO_REVIEW,
    "my holdings": Intent.PORTFOLIO_REVIEW,
    "my stocks": Intent.PORTFOLIO_REVIEW,
    "allocation": Intent.PORTFOLIO_REVIEW,
    "rebalance": Intent.PORTFOLIO_REVIEW,
    # Market
    "market today": Intent.MARKET_DATA,
    "stock price": Intent.MARKET_DATA,
    "s&p": Intent.MARKET_DATA,
    "nasdaq": Intent.MARKET_DATA,
    "dow": Intent.MARKET_DATA,
    "ticker": Intent.MARKET_DATA,
    # Goals
    "goal": Intent.GOAL_PLANNING,
    "retire": Intent.GOAL_PLANNING,
    "retirement": Intent.GOAL_PLANNING,
    "save for": Intent.GOAL_PLANNING,
    "down payment": Intent.GOAL_PLANNING,
    "projection": Intent.GOAL_PLANNING,
    # News
    "news": Intent.NEWS_SUMMARY,
    "headlines": Intent.NEWS_SUMMARY,
    "what happened": Intent.NEWS_SUMMARY,
    # Tax
    "tax": Intent.TAX_EDUCATION,
    "capital gains": Intent.TAX_EDUCATION,
    "ira": Intent.TAX_EDUCATION,
    "401k": Intent.TAX_EDUCATION,
    "hsa": Intent.TAX_EDUCATION,
    "roth": Intent.TAX_EDUCATION,
    "529": Intent.TAX_EDUCATION,
    "deduction": Intent.TAX_EDUCATION,
}
