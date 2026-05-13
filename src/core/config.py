"""Configuration loader for Finnie.

Reads config.yaml and merges with environment variables from .env.
Exposes a typed Config dataclass as a module-level singleton.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv

from src.core.exceptions import ConfigError

# Load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class LLMTemperatureConfig:
    classification: float = 0.0
    qa: float = 0.2
    news: float = 0.3
    goal_planning: float = 0.5
    default: float = 0.3


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0


@dataclass
class LLMConfig:
    model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"
    embedding_fallback: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens: int = 2048
    temperature: LLMTemperatureConfig = field(default_factory=LLMTemperatureConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)


@dataclass
class AlphaVantageConfig:
    base_url: str = "https://www.alphavantage.co/query"
    daily_quota: int = 25
    timeout: int = 10
    api_key: str = ""


@dataclass
class YFinanceConfig:
    timeout: int = 10


@dataclass
class APIsConfig:
    alpha_vantage: AlphaVantageConfig = field(default_factory=AlphaVantageConfig)
    yfinance: YFinanceConfig = field(default_factory=YFinanceConfig)


@dataclass
class CacheConfig:
    market_data_ttl: int = 1800
    news_ttl: int = 900
    llm_response_ttl: int = 86400
    faiss_retrieval_ttl: int = 3600
    max_size: int = 500


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_initial: int = 10
    top_k_final: int = 5
    similarity_threshold: float = 0.3
    index_path: str = "src/data/faiss_index"


@dataclass
class AgentConfig:
    name: str = ""
    rag_categories: List[str] = field(default_factory=list)


@dataclass
class AgentsConfig:
    finance_qa: AgentConfig = field(default_factory=AgentConfig)
    portfolio: AgentConfig = field(default_factory=AgentConfig)
    market: AgentConfig = field(default_factory=AgentConfig)
    goal_planning: AgentConfig = field(default_factory=AgentConfig)
    news: AgentConfig = field(default_factory=AgentConfig)
    tax: AgentConfig = field(default_factory=AgentConfig)


@dataclass
class UIConfig:
    page_title: str = "Finnie - AI Finance Assistant"
    page_icon: str = ""
    layout: str = "wide"
    market_refresh_seconds: int = 300
    max_chat_history: int = 50
    tabs: List[str] = field(default_factory=lambda: ["Chat", "Portfolio", "Market", "Goals", "Education"])


@dataclass
class AppConfig:
    name: str = "Finnie - AI Finance Assistant"
    version: str = "1.0.0"
    description: str = ""
    debug: bool = False
    log_level: str = "INFO"


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    apis: APIsConfig = field(default_factory=APIsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    google_api_key: str = ""
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config.yaml not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_config(raw: dict) -> Config:
    llm_raw = raw.get("llm", {})
    temp_raw = llm_raw.get("temperature", {})
    rl_raw = llm_raw.get("rate_limit", {})

    agents_raw = raw.get("agents", {})

    def _agent(key: str) -> AgentConfig:
        a = agents_raw.get(key, {})
        return AgentConfig(name=a.get("name", ""), rag_categories=a.get("rag_categories", []))

    av_raw = raw.get("apis", {}).get("alpha_vantage", {})
    yf_raw = raw.get("apis", {}).get("yfinance", {})
    cache_raw = raw.get("cache", {})
    rag_raw = raw.get("rag", {})
    ui_raw = raw.get("ui", {})
    app_raw = raw.get("app", {})

    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    av_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

    # Allow env override of debug/log_level
    debug = os.environ.get("FINNIE_DEBUG", str(app_raw.get("debug", False))).lower() in ("true", "1", "yes")
    log_level = os.environ.get("FINNIE_LOG_LEVEL", app_raw.get("log_level", "INFO"))

    return Config(
        app=AppConfig(
            name=app_raw.get("name", "Finnie"),
            version=app_raw.get("version", "1.0.0"),
            description=app_raw.get("description", ""),
            debug=debug,
            log_level=log_level,
        ),
        llm=LLMConfig(
            model=llm_raw.get("model", "gemini-2.0-flash"),
            embedding_model=llm_raw.get("embedding_model", "models/text-embedding-004"),
            embedding_fallback=llm_raw.get("embedding_fallback", "sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=llm_raw.get("max_tokens", 2048),
            temperature=LLMTemperatureConfig(
                classification=temp_raw.get("classification", 0.0),
                qa=temp_raw.get("qa", 0.2),
                news=temp_raw.get("news", 0.3),
                goal_planning=temp_raw.get("goal_planning", 0.5),
                default=temp_raw.get("default", 0.3),
            ),
            rate_limit=RateLimitConfig(
                requests_per_minute=rl_raw.get("requests_per_minute", 60),
                retry_attempts=rl_raw.get("retry_attempts", 3),
                retry_base_delay=rl_raw.get("retry_base_delay", 1.0),
                retry_max_delay=rl_raw.get("retry_max_delay", 30.0),
            ),
        ),
        apis=APIsConfig(
            alpha_vantage=AlphaVantageConfig(
                base_url=av_raw.get("base_url", "https://www.alphavantage.co/query"),
                daily_quota=av_raw.get("daily_quota", 25),
                timeout=av_raw.get("timeout", 10),
                api_key=av_api_key,
            ),
            yfinance=YFinanceConfig(timeout=yf_raw.get("timeout", 10)),
        ),
        cache=CacheConfig(
            market_data_ttl=cache_raw.get("market_data_ttl", 1800),
            news_ttl=cache_raw.get("news_ttl", 900),
            llm_response_ttl=cache_raw.get("llm_response_ttl", 86400),
            faiss_retrieval_ttl=cache_raw.get("faiss_retrieval_ttl", 3600),
            max_size=cache_raw.get("max_size", 500),
        ),
        rag=RAGConfig(
            chunk_size=rag_raw.get("chunk_size", 512),
            chunk_overlap=rag_raw.get("chunk_overlap", 50),
            top_k_initial=rag_raw.get("top_k_initial", 10),
            top_k_final=rag_raw.get("top_k_final", 5),
            similarity_threshold=rag_raw.get("similarity_threshold", 0.3),
            index_path=rag_raw.get("index_path", "src/data/faiss_index"),
        ),
        agents=AgentsConfig(
            finance_qa=_agent("finance_qa"),
            portfolio=_agent("portfolio"),
            market=_agent("market"),
            goal_planning=_agent("goal_planning"),
            news=_agent("news"),
            tax=_agent("tax"),
        ),
        ui=UIConfig(
            page_title=ui_raw.get("page_title", "Finnie - AI Finance Assistant"),
            page_icon=ui_raw.get("page_icon", ""),
            layout=ui_raw.get("layout", "wide"),
            market_refresh_seconds=ui_raw.get("market_refresh_seconds", 300),
            max_chat_history=ui_raw.get("max_chat_history", 50),
            tabs=ui_raw.get("tabs", ["Chat", "Portfolio", "Market", "Goals", "Education"]),
        ),
        google_api_key=google_api_key,
        project_root=_PROJECT_ROOT,
    )


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load and return Config from config.yaml + environment variables."""
    path = config_path or (_PROJECT_ROOT / "config.yaml")
    raw = _load_yaml(path)
    return _build_config(raw)


# Module-level singleton — import this throughout the codebase
settings: Config = load_config()
