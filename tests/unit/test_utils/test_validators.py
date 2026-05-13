"""Unit tests for input validators."""
import pytest

from src.utils.validators import (
    sanitize_query,
    validate_goal_inputs,
    validate_portfolio,
    validate_portfolio_csv_row,
    validate_ticker,
)


class TestSanitizeQuery:
    def test_normal_text_unchanged(self):
        assert sanitize_query("What is compound interest?") == "What is compound interest?"

    def test_strips_control_characters(self):
        result = sanitize_query("test\x00query\x07here")
        assert "\x00" not in result
        assert "\x07" not in result

    def test_truncates_long_query(self):
        long_query = "a" * 3000
        result = sanitize_query(long_query, max_length=2000)
        assert len(result) == 2000

    def test_strips_whitespace(self):
        assert sanitize_query("  hello  ") == "hello"


class TestValidatePortfolioRow:
    def test_valid_row(self):
        ok, msg = validate_portfolio_csv_row({"ticker": "AAPL", "shares": 10, "price_per_share": 190.0})
        assert ok is True
        assert msg is None

    def test_invalid_ticker(self):
        ok, msg = validate_portfolio_csv_row({"ticker": "INVALID TICKER!", "shares": 10})
        assert ok is False

    def test_zero_shares(self):
        ok, msg = validate_portfolio_csv_row({"ticker": "AAPL", "shares": 0})
        assert ok is False

    def test_negative_price(self):
        ok, msg = validate_portfolio_csv_row({"ticker": "AAPL", "shares": 10, "price_per_share": -1})
        assert ok is False

    def test_no_price_ok(self):
        ok, msg = validate_portfolio_csv_row({"ticker": "AAPL", "shares": 5})
        assert ok is True


class TestValidatePortfolio:
    def test_valid_portfolio(self):
        holdings = [
            {"ticker": "AAPL", "shares": 10},
            {"ticker": "MSFT", "shares": 5},
        ]
        ok, errors = validate_portfolio(holdings)
        assert ok is True
        assert errors == []

    def test_empty_portfolio(self):
        ok, errors = validate_portfolio([])
        assert ok is False

    def test_one_invalid_row(self):
        holdings = [
            {"ticker": "AAPL", "shares": 10},
            {"ticker": "BAD TICKER!!!", "shares": 5},
        ]
        ok, errors = validate_portfolio(holdings)
        assert ok is False
        assert len(errors) == 1


class TestValidateGoalInputs:
    def test_valid_inputs(self):
        ok, msg = validate_goal_inputs(1_000_000, 20, 50_000, 1_000)
        assert ok is True
        assert msg is None

    def test_negative_target(self):
        ok, msg = validate_goal_inputs(-100, 20, 0, 0)
        assert ok is False

    def test_zero_time_horizon(self):
        ok, msg = validate_goal_inputs(100_000, 0, 0, 100)
        assert ok is False

    def test_negative_savings(self):
        ok, msg = validate_goal_inputs(100_000, 10, -100, 100)
        assert ok is False


class TestValidateTicker:
    def test_valid_tickers(self):
        assert validate_ticker("AAPL") is True
        assert validate_ticker("BRK.B") is True
        assert validate_ticker("^GSPC") is True

    def test_invalid_tickers(self):
        assert validate_ticker("TOOLONG_TICKER_NAME") is False
        assert validate_ticker("HAS SPACE") is False
        assert validate_ticker("") is False
