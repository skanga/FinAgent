"""
Test Pydantic validation for request models.
"""

import pytest
from pydantic import ValidationError
from models import TickerRequest, PortfolioRequest, NaturalLanguageRequest


class TestTickerRequestValidation:
    """Test TickerRequest Pydantic validation."""

    def test_valid_ticker_request(self):
        """Test valid ticker request creation."""
        req = TickerRequest(ticker="aapl", period="1y")
        assert req.ticker == "AAPL", "Ticker should be uppercase"
        assert req.period == "1y"

    def test_empty_ticker_fails(self):
        """Test that empty ticker raises ValidationError."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="", period="1y")

    def test_ticker_too_long_fails(self):
        """Test that ticker longer than max length fails."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="THISISTOOLONG", period="1y")

    def test_invalid_ticker_characters_fails(self):
        """Test that special characters in ticker fail."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="AAP$L", period="1y")

    def test_suspicious_ticker_fails(self):
        """Test that suspicious ticker names fail."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="test", period="1y")

    def test_invalid_period_fails(self):
        """Test that invalid period fails."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="AAPL", period="1w")

    @pytest.mark.parametrize(
        "period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    )
    def test_all_valid_periods_pass(self, period):
        """Test all valid yfinance periods are accepted."""
        req = TickerRequest(ticker="AAPL", period=period)
        assert req.period == period

    def test_ticker_normalization(self):
        """Test ticker is normalized to uppercase."""
        req = TickerRequest(ticker="msft", period="1y")
        assert req.ticker == "MSFT"


class TestPortfolioRequestValidation:
    """Test PortfolioRequest Pydantic validation."""

    def test_valid_portfolio_without_weights(self):
        """Test valid portfolio request without weights."""
        req = PortfolioRequest(tickers=["AAPL", "MSFT", "GOOGL"], period="1y")
        assert req.tickers == ["AAPL", "MSFT", "GOOGL"]
        assert req.weights is None

    def test_valid_portfolio_with_weights(self):
        """Test valid portfolio request with weights."""
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2},
        )
        assert req.weights == {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

    def test_empty_ticker_list_fails(self):
        """Test that empty ticker list raises ValidationError."""
        with pytest.raises(ValidationError):
            PortfolioRequest(tickers=[], period="1y")

    def test_too_many_tickers_fails(self):
        """Test that exceeding max tickers fails."""
        many_tickers = [f"TICK{i}" for i in range(25)]  # MAX is 20
        with pytest.raises(ValidationError):
            PortfolioRequest(tickers=many_tickers, period="1y")

    def test_ticker_normalization_in_list(self):
        """Test tickers in list are normalized to uppercase."""
        req = PortfolioRequest(tickers=["aapl", "msft", "googl"], period="1y")
        assert req.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_weights_not_summing_to_one_fails(self):
        """Test that weights not summing to 1.0 fail."""
        with pytest.raises(ValidationError):
            PortfolioRequest(
                tickers=["AAPL", "MSFT", "GOOGL"],
                period="1y",
                weights={"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.3},  # Sum = 0.9
            )

    def test_negative_weights_fail(self):
        """Test that negative weights fail."""
        with pytest.raises(ValidationError):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": -0.3, "MSFT": 1.3},
            )

    def test_weights_exceeding_one_fail(self):
        """Test that individual weights > 1.0 fail."""
        with pytest.raises(ValidationError):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 1.5},
            )

    def test_missing_weight_for_ticker_fails(self):
        """Test that missing weight for ticker fails."""
        with pytest.raises(ValidationError):
            PortfolioRequest(
                tickers=["AAPL", "MSFT", "GOOGL"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.5},  # Missing GOOGL
            )

    def test_extra_weight_for_missing_ticker_fails(self):
        """Test that extra weight for non-existent ticker fails."""
        with pytest.raises(ValidationError):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2},  # Extra GOOGL
            )

    def test_weights_within_tolerance_pass(self):
        """Test that weights within tolerance (0.999) pass."""
        req = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.333, "MSFT": 0.333, "GOOGL": 0.333},  # Sum = 0.999
        )
        assert abs(sum(req.weights.values()) - 1.0) < 0.01


class TestNaturalLanguageRequestValidation:
    """Test NaturalLanguageRequest Pydantic validation."""

    def test_valid_request(self):
        """Test valid natural language request."""
        req = NaturalLanguageRequest(query="Compare AAPL and MSFT over the past year")
        assert len(req.query) > 0

    def test_query_too_short_fails(self):
        """Test that query too short fails."""
        with pytest.raises(ValidationError):
            NaturalLanguageRequest(query="Hi")

    def test_query_too_long_fails(self):
        """Test that query too long fails."""
        with pytest.raises(ValidationError):
            NaturalLanguageRequest(query="x" * 501)

    def test_empty_query_fails(self):
        """Test that empty query (whitespace only) fails."""
        with pytest.raises(ValidationError):
            NaturalLanguageRequest(query="     ")

    def test_query_normalization(self):
        """Test query normalization (trim whitespace)."""
        req = NaturalLanguageRequest(query="  Analyze AAPL  ")
        assert req.query == "Analyze AAPL"

    def test_custom_output_directory(self):
        """Test custom output directory."""
        req = NaturalLanguageRequest(
            query="Analyze AAPL", output_dir="./custom_reports"
        )
        assert req.output_dir == "./custom_reports"


class TestValidationErrorMessages:
    """Test validation error message quality."""

    def test_invalid_ticker_error_message(self):
        """Test invalid ticker error message is clear."""
        try:
            TickerRequest(ticker="AAP$L", period="1y")
        except ValidationError as e:
            error_msg = str(e.errors()[0]["ctx"]["error"])
            assert "Invalid characters" in error_msg

    def test_invalid_period_error_message(self):
        """Test invalid period error includes suggestions."""
        try:
            TickerRequest(ticker="AAPL", period="1week")
        except ValidationError as e:
            error_msg = str(e.errors()[0]["ctx"]["error"])
            assert "Must be one of" in error_msg

    def test_weight_sum_error_shows_actual_value(self):
        """Test weight sum error shows actual sum value."""
        try:
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.3, "MSFT": 0.3},
            )
        except ValidationError as e:
            error_msg = str(e.errors()[0]["ctx"]["error"])
            assert "got 0." in error_msg  # Shows actual sum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
