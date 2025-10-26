"""
Comprehensive tests for Pydantic model validation.

Tests ensure that all request models (TickerRequest, PortfolioRequest, NaturalLanguageRequest)
are properly validated with clear error messages for invalid inputs.

Coverage includes:
- TickerRequest validation (ticker format, period validation, normalization)
- PortfolioRequest validation (tickers list, weights, bounds checking)
- NaturalLanguageRequest validation (query length, normalization)
- Edge cases and boundary conditions
- Error message quality and clarity
- Model serialization/deserialization (JSON, dict)
"""

import pytest
from pydantic import ValidationError
from models import TickerRequest, PortfolioRequest, NaturalLanguageRequest


class TestTickerRequestValidation:
    """Test TickerRequest Pydantic model validation."""

    def test_valid_ticker_request(self):
        """Valid ticker request should be accepted."""
        request = TickerRequest(ticker="AAPL", period="1y")
        assert request.ticker == "AAPL"
        assert request.period == "1y"

    def test_ticker_uppercase_normalization(self):
        """Tickers should be normalized to uppercase."""
        request = TickerRequest(ticker="aapl", period="1y")
        assert request.ticker == "AAPL"

    def test_ticker_whitespace_stripped(self):
        """Whitespace should be stripped from tickers."""
        request = TickerRequest(ticker="  AAPL  ", period="1y")
        assert request.ticker == "AAPL"

    def test_empty_ticker_fails(self):
        """Empty ticker should be rejected."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="", period="1y")

    def test_invalid_ticker_characters(self):
        """Invalid characters in ticker should be rejected."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            TickerRequest(ticker="AAPL@@@", period="1y")

    def test_invalid_ticker_special_chars(self):
        """Special characters like $ should be rejected."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            TickerRequest(ticker="AAP$L", period="1y")

    def test_suspicious_ticker_names(self):
        """Suspicious ticker names should be rejected."""
        suspicious_names = ["test", "null", "none", "undefined"]
        for name in suspicious_names:
            with pytest.raises(ValidationError, match="Suspicious ticker name"):
                TickerRequest(ticker=name, period="1y")

    def test_ticker_too_long(self):
        """Tickers longer than 10 characters should be rejected."""
        with pytest.raises(ValidationError):
            TickerRequest(ticker="A" * 11, period="1y")

        # Also test with longer string
        with pytest.raises(ValidationError):
            TickerRequest(ticker="THISISTOOLONG", period="1y")

    def test_invalid_period(self):
        """Invalid periods should be rejected."""
        with pytest.raises(ValidationError, match="Invalid period"):
            TickerRequest(ticker="AAPL", period="invalid")

    def test_invalid_period_1w(self):
        """Invalid period '1w' should be rejected."""
        with pytest.raises(ValidationError, match="Invalid period"):
            TickerRequest(ticker="AAPL", period="1w")

    def test_invalid_period_1week(self):
        """Invalid period '1week' should be rejected."""
        with pytest.raises(ValidationError, match="Invalid period"):
            TickerRequest(ticker="AAPL", period="1week")

    @pytest.mark.parametrize(
        "period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    )
    def test_all_valid_periods_pass(self, period):
        """All valid yfinance periods should be accepted."""
        request = TickerRequest(ticker="AAPL", period=period)
        assert request.period == period


class TestTickerRequestErrorMessages:
    """Test TickerRequest error message quality."""

    def test_invalid_ticker_error_message(self):
        """Invalid ticker error message should be clear."""
        try:
            TickerRequest(ticker="AAP$L", period="1y")
        except ValidationError as e:
            error_msg = str(e.errors()[0]["ctx"]["error"])
            assert "Invalid characters" in error_msg

    def test_invalid_period_error_message(self):
        """Invalid period error should include suggestions."""
        try:
            TickerRequest(ticker="AAPL", period="1week")
        except ValidationError as e:
            error_msg = str(e.errors()[0]["ctx"]["error"])
            assert "Must be one of" in error_msg or "Invalid period" in error_msg


class TestPortfolioRequestValidation:
    """Test PortfolioRequest Pydantic model validation."""

    def test_valid_portfolio_request_no_weights(self):
        """Valid portfolio request without weights should be accepted."""
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y"
        )
        assert request.tickers == ["AAPL", "MSFT", "GOOGL"]
        assert request.period == "1y"
        assert request.weights is None

    def test_valid_portfolio_request_with_weights(self):
        """Valid portfolio request with weights should be accepted."""
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
        )
        assert request.weights == {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

    def test_tickers_normalized_to_uppercase(self):
        """Tickers should be normalized to uppercase."""
        request = PortfolioRequest(
            tickers=["aapl", "msft", "googl"],
            period="1y"
        )
        assert request.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_tickers_whitespace_stripped(self):
        """Whitespace should be stripped from tickers."""
        request = PortfolioRequest(
            tickers=["  AAPL  ", " MSFT ", "GOOGL  "],
            period="1y"
        )
        assert request.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_empty_tickers_list_rejected(self):
        """Empty tickers list should be rejected."""
        with pytest.raises(ValidationError, match="at least 1 item"):
            PortfolioRequest(tickers=[], period="1y")

    def test_too_many_tickers_rejected(self):
        """More than MAX_TICKERS_ALLOWED (20) should be rejected."""
        with pytest.raises(ValidationError, match="at most 20 items"):
            PortfolioRequest(tickers=[f"TICK{i}" for i in range(21)], period="1y")

        # Also test with 25 tickers
        many_tickers = [f"TICK{i}" for i in range(25)]
        with pytest.raises(ValidationError, match="at most 20 items"):
            PortfolioRequest(tickers=many_tickers, period="1y")

    def test_invalid_ticker_in_list(self):
        """Invalid ticker in list should be rejected."""
        with pytest.raises(ValidationError, match="Invalid characters"):
            PortfolioRequest(tickers=["AAPL", "MSFT@@@", "GOOGL"], period="1y")

    def test_suspicious_ticker_in_list(self):
        """Suspicious ticker in list should be rejected."""
        with pytest.raises(ValidationError, match="Suspicious ticker name"):
            PortfolioRequest(tickers=["AAPL", "test", "GOOGL"], period="1y")

    def test_weights_sum_validation_too_low(self):
        """Weights summing below 1.0 (outside tolerance) should be rejected."""
        # Sum = 0.9 (too low)
        with pytest.raises(ValidationError, match="Weights must sum to 1.0"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT", "GOOGL"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.2, "GOOGL": 0.2}
            )

        # Sum = 0.6 (too low)
        with pytest.raises(ValidationError, match="Weights must sum to 1.0"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.3, "MSFT": 0.3}
            )

    def test_weights_sum_validation_too_high(self):
        """Weights summing above 1.0 (outside tolerance) should be rejected."""
        # Sum = 1.1 (too high)
        with pytest.raises(ValidationError, match="Weights must sum to 1.0"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT", "GOOGL"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.3}
            )

    def test_weights_within_tolerance_accepted(self):
        """Weights summing to 1.0 within tolerance should be accepted."""
        # Sum = 1.009 (within tolerance)
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.503, "MSFT": 0.303, "GOOGL": 0.203}
        )
        assert request.weights is not None

        # Sum = 0.999 (within tolerance)
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.333, "MSFT": 0.333, "GOOGL": 0.333}
        )
        assert abs(sum(request.weights.values()) - 1.0) < 0.01

    def test_negative_weight_rejected(self):
        """Negative weights should be rejected."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": -0.5}
            )

        # Also test with different negative value
        with pytest.raises(ValidationError, match="cannot be negative"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": -0.3, "MSFT": 1.3}
            )

    def test_weight_exceeds_one_rejected(self):
        """Weights exceeding 1.0 should be rejected."""
        with pytest.raises(ValidationError, match="cannot exceed 1.0"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 1.5, "MSFT": -0.5}
            )

        # Also test with different value > 1.0
        with pytest.raises(ValidationError, match="cannot exceed 1.0"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 1.5}
            )

    def test_weights_missing_ticker(self):
        """Missing weights for tickers should be rejected."""
        with pytest.raises(ValidationError, match="Missing weights for"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT", "GOOGL"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.5}  # Missing GOOGL
            )

    def test_weights_extra_ticker(self):
        """Extra weights for non-existent tickers should be rejected."""
        with pytest.raises(ValidationError, match="Extra weights for"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.5, "GOOGL": 0.0}  # Extra GOOGL
            )

        # Also test with different extra ticker
        with pytest.raises(ValidationError, match="Extra weights for"):
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}
            )


class TestPortfolioRequestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_ticker_portfolio(self):
        """Portfolio with single ticker should be accepted."""
        request = PortfolioRequest(tickers=["AAPL"], period="1y")
        assert len(request.tickers) == 1

    def test_single_ticker_with_weight_one(self):
        """Single ticker with weight 1.0 should be accepted."""
        request = PortfolioRequest(
            tickers=["AAPL"],
            period="1y",
            weights={"AAPL": 1.0}
        )
        assert request.weights == {"AAPL": 1.0}

    def test_maximum_tickers_allowed(self):
        """Exactly 20 tickers should be accepted."""
        tickers = [f"TICK{i:02d}" for i in range(20)]
        request = PortfolioRequest(tickers=tickers, period="1y")
        assert len(request.tickers) == 20

    def test_equal_weights_portfolio(self):
        """Equal weights portfolio should be accepted."""
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 1/3, "MSFT": 1/3, "GOOGL": 1/3}
        )
        # 1/3 * 3 = 0.9999... which is within tolerance
        assert request.weights is not None

    def test_ticker_with_dot(self):
        """Ticker with dot (valid character) should be accepted."""
        request = PortfolioRequest(tickers=["BRK.B"], period="1y")
        assert request.tickers == ["BRK.B"]

    def test_ticker_with_hyphen(self):
        """Ticker with hyphen (valid character) should be accepted."""
        request = PortfolioRequest(tickers=["BF-B"], period="1y")
        assert request.tickers == ["BF-B"]


class TestPortfolioRequestErrorMessages:
    """Test that error messages are clear and helpful."""

    def test_empty_ticker_error_message(self):
        """Error message for empty ticker should be clear."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioRequest(tickers=[], period="1y")

        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert "tickers" in str(errors[0]["loc"])

    def test_invalid_period_error_message(self):
        """Error message for invalid period should be clear."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioRequest(tickers=["AAPL"], period="invalid_period")

        errors = exc_info.value.errors()
        assert any("Invalid period" in str(error["msg"]) for error in errors)

    def test_invalid_weights_error_message(self):
        """Error message for invalid weights should be clear."""
        with pytest.raises(ValidationError) as exc_info:
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.3, "MSFT": 0.3}  # Sum = 0.6, not 1.0
            )

        errors = exc_info.value.errors()
        assert any("Weights must sum to 1.0" in str(error["msg"]) for error in errors)

    def test_weight_sum_error_shows_actual_value(self):
        """Weight sum error should show actual sum value."""
        try:
            PortfolioRequest(
                tickers=["AAPL", "MSFT"],
                period="1y",
                weights={"AAPL": 0.3, "MSFT": 0.3},
            )
        except ValidationError as e:
            error_msg = str(e.errors()[0]["ctx"]["error"])
            assert "got 0." in error_msg  # Shows actual sum


class TestNaturalLanguageRequestValidation:
    """Test NaturalLanguageRequest Pydantic validation."""

    def test_valid_request(self):
        """Valid natural language request should be accepted."""
        req = NaturalLanguageRequest(query="Compare AAPL and MSFT over the past year")
        assert len(req.query) > 0
        assert req.query == "Compare AAPL and MSFT over the past year"

    def test_query_too_short_fails(self):
        """Query too short should be rejected."""
        with pytest.raises(ValidationError):
            NaturalLanguageRequest(query="Hi")

    def test_query_too_long_fails(self):
        """Query too long should be rejected."""
        with pytest.raises(ValidationError):
            NaturalLanguageRequest(query="x" * 501)

    def test_empty_query_fails(self):
        """Empty query (whitespace only) should be rejected."""
        with pytest.raises(ValidationError):
            NaturalLanguageRequest(query="     ")

    def test_query_normalization(self):
        """Query should be normalized (trim whitespace)."""
        req = NaturalLanguageRequest(query="  Analyze AAPL  ")
        assert req.query == "Analyze AAPL"

    def test_custom_output_directory(self):
        """Custom output directory should be accepted."""
        req = NaturalLanguageRequest(
            query="Analyze AAPL", output_dir="./custom_reports"
        )
        assert req.output_dir == "./custom_reports"


class TestPydanticModelSerialization:
    """Test that Pydantic models can be serialized/deserialized."""

    def test_ticker_request_to_dict(self):
        """TickerRequest should serialize to dict."""
        request = TickerRequest(ticker="AAPL", period="1y")
        data = request.model_dump()

        assert data["ticker"] == "AAPL"
        assert data["period"] == "1y"

    def test_ticker_request_from_dict(self):
        """TickerRequest should deserialize from dict."""
        data = {"ticker": "AAPL", "period": "1y"}
        request = TickerRequest(**data)

        assert request.ticker == "AAPL"
        assert request.period == "1y"

    def test_portfolio_request_to_dict(self):
        """PortfolioRequest should serialize to dict."""
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT"],
            period="1y",
            weights={"AAPL": 0.6, "MSFT": 0.4}
        )
        data = request.model_dump()

        assert data["tickers"] == ["AAPL", "MSFT"]
        assert data["period"] == "1y"
        assert data["weights"] == {"AAPL": 0.6, "MSFT": 0.4}

    def test_portfolio_request_from_dict(self):
        """PortfolioRequest should deserialize from dict."""
        data = {
            "tickers": ["AAPL", "MSFT"],
            "period": "1y",
            "weights": {"AAPL": 0.6, "MSFT": 0.4}
        }
        request = PortfolioRequest(**data)

        assert request.tickers == ["AAPL", "MSFT"]
        assert request.period == "1y"
        assert request.weights == {"AAPL": 0.6, "MSFT": 0.4}

    def test_portfolio_request_json_serialization(self):
        """PortfolioRequest should serialize to JSON."""
        request = PortfolioRequest(
            tickers=["AAPL", "MSFT"],
            period="1y",
            weights={"AAPL": 0.6, "MSFT": 0.4}
        )
        json_str = request.model_dump_json()

        assert "AAPL" in json_str
        assert "MSFT" in json_str
        assert "0.6" in json_str

    def test_portfolio_request_json_deserialization(self):
        """PortfolioRequest should deserialize from JSON."""
        json_str = '{"tickers": ["AAPL", "MSFT"], "period": "1y", "weights": {"AAPL": 0.6, "MSFT": 0.4}}'
        request = PortfolioRequest.model_validate_json(json_str)

        assert request.tickers == ["AAPL", "MSFT"]
        assert request.period == "1y"
        assert request.weights == {"AAPL": 0.6, "MSFT": 0.4}

    def test_natural_language_request_to_dict(self):
        """NaturalLanguageRequest should serialize to dict."""
        request = NaturalLanguageRequest(query="Analyze AAPL", output_dir="./reports")
        data = request.model_dump()

        assert data["query"] == "Analyze AAPL"
        assert data["output_dir"] == "./reports"

    def test_natural_language_request_from_dict(self):
        """NaturalLanguageRequest should deserialize from dict."""
        data = {"query": "Analyze AAPL", "output_dir": "./reports"}
        request = NaturalLanguageRequest(**data)

        assert request.query == "Analyze AAPL"
        assert request.output_dir == "./reports"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
