"""Unit tests for intent classification — keyword matching and LLM fallback.

Covers gaps not addressed by existing integration tests:
- LLM fallback path when no keyword matches
- Ambiguous / multi-signal queries
- Out-of-scope detection via LLM
- Malformed LLM JSON is handled gracefully
- Full IntentClassifier.classify() contract
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.workflow.intent_labels import Intent
from src.workflow.router import IntentClassifier, _keyword_classify, _llm_classify


# ─── Keyword classifier — additional edge cases ────────────────────────────────

class TestKeywordClassifierEdgeCases:
    """Cases that supplement the existing keyword tests."""

    def test_roth_keyword_maps_to_tax(self):
        assert _keyword_classify("What is a Roth IRA?") == Intent.TAX_EDUCATION

    def test_401k_keyword_maps_to_tax(self):
        assert _keyword_classify("Should I max out my 401k?") == Intent.TAX_EDUCATION

    def test_news_keyword_maps_to_news(self):
        assert _keyword_classify("Give me the latest financial news") == Intent.NEWS_SUMMARY

    def test_dow_keyword_maps_to_market(self):
        assert _keyword_classify("How is the Dow performing?") == Intent.MARKET_DATA

    def test_keyword_match_is_case_insensitive(self):
        # KEYWORD_MAP uses .lower() comparison inside _keyword_classify
        assert _keyword_classify("HELLO there") == Intent.GREETING

    def test_query_with_no_matching_keyword_returns_none(self):
        assert _keyword_classify("Can you explain bonds?") is None

    def test_query_about_crypto_returns_none(self):
        # Crypto is not a keyword — falls through to LLM
        assert _keyword_classify("Should I buy Bitcoin?") is None

    def test_very_short_query_without_keyword_returns_none(self):
        assert _keyword_classify("why") is None


# ─── LLM fallback classifier ──────────────────────────────────────────────────

class TestLLMClassifier:
    """_llm_classify must parse the LLM JSON and handle failures gracefully."""

    def _mock_llm(self, raw_response: str) -> MagicMock:
        llm = MagicMock()
        llm.generate_structured.return_value = raw_response
        return llm

    def test_parses_general_finance_response(self):
        llm = self._mock_llm('{"intent": "general_finance", "confidence": 0.9}')
        result = _llm_classify("What is a mutual fund?", llm)
        assert result is not None
        assert result.intent == Intent.GENERAL_FINANCE
        assert result.confidence == pytest.approx(0.9)

    def test_parses_out_of_scope_response(self):
        llm = self._mock_llm('{"intent": "out_of_scope", "confidence": 0.95}')
        result = _llm_classify("What is the weather in Paris?", llm)
        assert result.intent == Intent.OUT_OF_SCOPE

    def test_parses_tax_intent(self):
        llm = self._mock_llm('{"intent": "tax_education", "confidence": 0.85}')
        result = _llm_classify("Are there tax benefits to an HSA?", llm)
        assert result.intent == Intent.TAX_EDUCATION

    def test_unknown_intent_string_falls_back_to_general_finance(self):
        llm = self._mock_llm('{"intent": "completely_made_up", "confidence": 0.7}')
        result = _llm_classify("Something unusual", llm)
        assert result is not None
        assert result.intent == Intent.GENERAL_FINANCE

    def test_malformed_json_returns_none(self):
        llm = self._mock_llm("this is not json at all")
        result = _llm_classify("What is a bond?", llm)
        assert result is None

    def test_llm_exception_returns_none(self):
        llm = MagicMock()
        llm.generate_structured.side_effect = Exception("API down")
        result = _llm_classify("What is a bond?", llm)
        assert result is None

    def test_json_embedded_in_text_is_extracted(self):
        # LLM may wrap JSON in prose — the regex extractor must still find it
        llm = self._mock_llm(
            'Sure, here is my answer: {"intent": "portfolio_review", "confidence": 0.8} thanks!'
        )
        result = _llm_classify("Analyze my stocks", llm)
        assert result is not None
        assert result.intent == Intent.PORTFOLIO_REVIEW


# ─── IntentClassifier.classify() — full contract ──────────────────────────────

class TestIntentClassifierClassify:
    """End-to-end classify() method: keyword fast-path, LLM fallback, defaults."""

    def _classifier_with_llm(self, raw_llm_response: str) -> IntentClassifier:
        llm = MagicMock()
        llm.generate_structured.return_value = raw_llm_response
        return IntentClassifier(llm_client=llm)

    def test_empty_query_returns_greeting(self):
        clf = IntentClassifier(llm_client=MagicMock())
        result = clf.classify("")
        assert result.intent == Intent.GREETING
        assert result.confidence == 1.0

    def test_whitespace_only_query_returns_greeting(self):
        clf = IntentClassifier(llm_client=MagicMock())
        result = clf.classify("   ")
        assert result.intent == Intent.GREETING

    def test_keyword_match_skips_llm(self):
        llm = MagicMock()
        clf = IntentClassifier(llm_client=llm)
        clf.classify("What is the S&P 500 today?")
        llm.generate_structured.assert_not_called()

    def test_keyword_match_confidence_is_high(self):
        clf = IntentClassifier(llm_client=MagicMock())
        result = clf.classify("Can you analyze my portfolio?")
        assert result.confidence >= 0.9

    def test_ambiguous_query_calls_llm(self):
        llm = MagicMock()
        llm.generate_structured.return_value = '{"intent": "general_finance", "confidence": 0.7}'
        clf = IntentClassifier(llm_client=llm)
        clf.classify("Should I buy or rent a house?")
        llm.generate_structured.assert_called_once()

    def test_llm_failure_falls_back_to_general_finance(self):
        llm = MagicMock()
        llm.generate_structured.side_effect = Exception("network error")
        clf = IntentClassifier(llm_client=llm)
        result = clf.classify("Why do businesses use leverage?")
        assert result.intent == Intent.GENERAL_FINANCE

    def test_result_has_required_attributes(self):
        clf = self._classifier_with_llm('{"intent": "general_finance", "confidence": 0.8}')
        result = clf.classify("Tell me about bonds")
        assert hasattr(result, "intent")
        assert hasattr(result, "confidence")
        assert hasattr(result, "secondary_intents")
