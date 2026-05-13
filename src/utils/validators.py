"""Input validation and sanitization for Finnie."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def sanitize_query(query: str, max_length: int = 2000) -> str:
    """Strip control characters and truncate overly long queries."""
    # Remove non-printable control chars (keep newlines/tabs)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", query)
    return cleaned[:max_length].strip()


def validate_portfolio_csv_row(row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a single portfolio CSV row.

    Expected columns: ticker, shares, price_per_share (optional)
    Returns (is_valid, error_message).
    """
    ticker = row.get("ticker", "")
    if not ticker or not re.match(r"^[A-Za-z.\-]{1,10}$", str(ticker)):
        return False, f"Invalid ticker symbol: '{ticker}'"

    shares_raw = row.get("shares", row.get("quantity", 0))
    try:
        shares = float(shares_raw)
        if shares <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return False, f"Shares must be a positive number, got: '{shares_raw}'"

    price_raw = row.get("price_per_share", row.get("price", None))
    if price_raw is not None:
        try:
            price = float(price_raw)
            if price < 0:
                raise ValueError
        except (TypeError, ValueError):
            return False, f"Price must be a non-negative number, got: '{price_raw}'"

    return True, None


def validate_portfolio(holdings: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate a full portfolio list. Returns (is_valid, list_of_errors)."""
    errors: List[str] = []
    if not holdings:
        return False, ["Portfolio is empty."]
    for i, row in enumerate(holdings):
        ok, msg = validate_portfolio_csv_row(row)
        if not ok:
            errors.append(f"Row {i + 1}: {msg}")
    return len(errors) == 0, errors


def validate_goal_inputs(
    target_amount: float,
    time_horizon_years: int,
    current_savings: float,
    monthly_contribution: float,
) -> Tuple[bool, Optional[str]]:
    """Validate goal planning inputs."""
    if target_amount <= 0:
        return False, "Target amount must be positive."
    if not (1 <= time_horizon_years <= 60):
        return False, "Time horizon must be between 1 and 60 years."
    if current_savings < 0:
        return False, "Current savings cannot be negative."
    if monthly_contribution < 0:
        return False, "Monthly contribution cannot be negative."
    return True, None


def validate_ticker(ticker: str) -> bool:
    """Return True if ticker looks like a valid stock symbol."""
    return bool(re.match(r"^[A-Za-z.\-\^]{1,10}$", ticker.strip()))
