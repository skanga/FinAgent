"""
Test input validation via Pydantic models and helper functions.

Tests PortfolioRequest Pydantic validation and parse_weights() helper function.
This test file validates that the integration between main.py and Pydantic models works correctly.
"""

import pytest
from pydantic import ValidationError
from main import parse_weights
from models import PortfolioRequest


class TestWeightValidation:
    """Test portfolio weight parsing and validation via Pydantic."""

    def test_parse_weights_rejects_wrong_count(self):
        """Wrong number of weights should raise ValueError from parse_weights."""
        with pytest.raises(ValueError, match="(Number of weights|must match)"):
            parse_weights("0.5,0.5", ["AAPL", "MSFT", "GOOGL"])

    def test_parse_weights_with_pydantic_rejects_negative(self):
        """Negative weights should be caught by Pydantic when creating PortfolioRequest."""
        weights_dict = parse_weights("0.5,0.5", ["AAPL", "MSFT"])
        # Manually create invalid weights to test Pydantic validation
        weights_dict["AAPL"] = -0.5
        with pytest.raises(ValidationError, match="cannot be negative"):
            PortfolioRequest(tickers=["AAPL", "MSFT"], period="1y", weights=weights_dict)

    def test_parse_weights_with_pydantic_rejects_over_one(self):
        """Weights > 1.0 should be caught by Pydantic."""
        weights_dict = {"AAPL": 1.5, "MSFT": -0.5}
        with pytest.raises(ValidationError):
            PortfolioRequest(tickers=["AAPL", "MSFT"], period="1y", weights=weights_dict)

    def test_parse_weights_with_pydantic_rejects_wrong_sum(self):
        """Weights not summing to 1.0 should be caught by Pydantic."""
        weights_dict = parse_weights("0.3,0.3,0.3", ["AAPL", "MSFT", "GOOGL"])
        # Sum = 0.9, not 1.0
        with pytest.raises(ValidationError, match="sum"):
            PortfolioRequest(tickers=["AAPL", "MSFT", "GOOGL"], period="1y", weights=weights_dict)

    def test_parse_weights_accepts_valid(self):
        """Valid weights should be parsed correctly into a dictionary."""
        result = parse_weights("0.5,0.3,0.2", ["AAPL", "MSFT", "GOOGL"])
        assert result == {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

    def test_parse_weights_accepts_within_tolerance(self):
        """Weights within tolerance (sum ≈ 1.0) should be accepted."""
        result = parse_weights("0.333,0.333,0.334", ["AAPL", "MSFT", "GOOGL"])
        assert result is not None
        assert len(result) == 3

    @pytest.mark.parametrize(
        "weights,tickers,expected",
        [
            ("0.5,0.5", ["AAPL", "MSFT"], {"AAPL": 0.5, "MSFT": 0.5}),
            ("1.0", ["AAPL"], {"AAPL": 1.0}),
            ("0.25,0.25,0.5", ["A", "B", "C"], {"A": 0.25, "B": 0.25, "C": 0.5}),
            (
                "0.2,0.2,0.3,0.3",
                ["A", "B", "C", "D"],
                {"A": 0.2, "B": 0.2, "C": 0.3, "D": 0.3},
            ),
        ],
    )
    def test_parse_weights(self, weights, tickers, expected):
        """Test various valid weight configurations are parsed correctly."""
        result = parse_weights(weights, tickers)
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_weights,tickers,error_source",
        [
            # parse_weights validates count
            ("0.5", ["AAPL", "MSFT"], "parse_weights"),  # Wrong count

            # Pydantic validates value ranges and sum
            ("-1.0", ["AAPL"], "pydantic"),  # Negative (parsed but Pydantic rejects)
            ("2.0", ["AAPL"], "pydantic"),  # > 1.0 (parsed but Pydantic rejects)
            ("0.5,0.6", ["AAPL", "MSFT"], "pydantic"),  # Sum > 1.0
            ("0.3,0.3", ["AAPL", "MSFT"], "pydantic"),  # Sum < 1.0
        ],
    )
    def test_parse_invalid_weights(self, invalid_weights, tickers, error_source):
        """Test invalid weight configurations."""
        if error_source == "parse_weights":
            # parse_weights should reject wrong count
            with pytest.raises(ValueError):
                parse_weights(invalid_weights, tickers)
        else:
            # parse_weights parses successfully, but Pydantic rejects
            weights_dict = parse_weights(invalid_weights, tickers)
            with pytest.raises(ValidationError):
                PortfolioRequest(tickers=tickers, period="1y", weights=weights_dict)


class TestPeriodValidation:
    """Test time period validation via Pydantic."""

    @pytest.mark.parametrize(
        "period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    )
    def test_validate_period_accepts_valid(self, period):
        """All valid yfinance periods should be accepted by Pydantic."""
        # Should not raise
        request = PortfolioRequest(tickers=["AAPL"], period=period)
        assert request.period == period

    def test_validate_period_rejects_invalid(self):
        """Invalid period '1w' should raise ValidationError."""
        with pytest.raises(ValidationError, match="(?i)(invalid period|must be one of)"):
            PortfolioRequest(tickers=["AAPL"], period="1w")

    def test_validate_period_rejects_typo(self):
        """Common typo '1yr' should raise ValidationError."""
        with pytest.raises(ValidationError, match="(?i)(invalid period|must be one of)"):
            PortfolioRequest(tickers=["AAPL"], period="1yr")

    def test_validate_period_case_sensitive(self):
        """Uppercase '1Y' should raise ValidationError (periods are lowercase)."""
        with pytest.raises(ValidationError, match="(?i)(invalid period|must be one of)"):
            PortfolioRequest(tickers=["AAPL"], period="1Y")


class TestTickerValidation:
    """Test ticker symbol validation via Pydantic."""

    def test_validate_tickers_accepts_valid(self):
        """Valid ticker list should be accepted without error."""
        request = PortfolioRequest(tickers=["AAPL", "MSFT", "GOOGL"], period="1y")
        assert request.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_validate_tickers_rejects_empty(self):
        """Empty ticker list should raise ValidationError."""
        with pytest.raises(ValidationError, match="(?i)at least"):
            PortfolioRequest(tickers=[], period="1y")

    def test_validate_tickers_rejects_too_many(self):
        """More than MAX_TICKERS_ALLOWED (20) should raise ValidationError."""
        too_many = [f"TICK{i}" for i in range(25)]
        with pytest.raises(ValidationError, match="(?i)(at most)"):
            PortfolioRequest(tickers=too_many, period="1y")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
