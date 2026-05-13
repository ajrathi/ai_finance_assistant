"""Unit tests for configuration loading."""
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.config import load_config, _build_config


class TestConfigLoading:
    def test_config_loads_from_yaml(self):
        config = load_config()
        assert config is not None
        assert config.llm.model == "gemini-2.5-flash"
        assert config.llm.rate_limit.requests_per_minute == 60

    def test_cache_ttls_correct(self):
        config = load_config()
        assert config.cache.market_data_ttl == 1800
        assert config.cache.news_ttl == 900
        assert config.cache.llm_response_ttl == 86400

    def test_rag_config(self):
        config = load_config()
        assert config.rag.chunk_size == 512
        assert config.rag.chunk_overlap == 50
        assert config.rag.top_k_final == 5

    def test_agent_configs_present(self):
        config = load_config()
        assert config.agents.finance_qa.name == "Finance Q&A Agent"
        assert config.agents.portfolio.name == "Portfolio Analysis Agent"
        assert config.agents.tax.name == "Tax Education Agent"

    def test_env_var_overrides_api_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}):
            config = load_config()
            assert config.google_api_key == "test-key-123"

    def test_env_var_debug_flag(self):
        with patch.dict(os.environ, {"FINNIE_DEBUG": "true"}):
            config = load_config()
            assert config.app.debug is True

    def test_temperature_config(self):
        config = load_config()
        assert config.llm.temperature.classification == 0.0
        assert config.llm.temperature.qa == 0.2
        assert config.llm.temperature.goal_planning == 0.5

    def test_project_root_is_path(self):
        config = load_config()
        assert isinstance(config.project_root, Path)
        assert config.project_root.exists()
