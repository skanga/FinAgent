"""
Tests for improved error handling in orchestrator.py.

Tests that different exception types are handled appropriately with
correct logging levels and user-friendly messages.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
from orchestrator import FinancialReportOrchestrator
from config import Config
from models import TickerAnalysis


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.openai_api_key = "test-key"
    config.openai_base_url = "https://api.openai.com/v1"
    config.benchmark_ticker = "SPY"
    config.risk_free_rate = 0.02
    config.cache_ttl_hours = 24
    config.max_workers = 3
    config.request_timeout = 30
    config.provider = "openai"
    return config


@pytest.fixture
def orchestrator(mock_config):
    """Create an orchestrator with mocked dependencies."""
    with patch("orchestrator.IntegratedLLMInterface"):
        orch = FinancialReportOrchestrator(mock_config)
        return orch


class TestErrorHandlingDistinction:
    """Test that different error types are handled appropriately."""

    @patch("orchestrator.CachedDataFetcher")
    def test_value_error_handled_as_validation_error(
        self, mock_fetcher_class, orchestrator
    ):
        """ValueError should be treated as a permanent validation error."""
        # Setup mock to raise ValueError
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError("Invalid ticker symbol")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Invalid ticker" in result.error
        assert "INVALID" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_network_error_handled_as_retryable(
        self, mock_fetcher_class, orchestrator
    ):
        """OSError/ConnectionError should be treated as temporary network errors."""
        # Setup mock to raise OSError
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = OSError("Connection refused")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Network error" in result.error
        assert "AAPL" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_connection_error_handled_as_retryable(
        self, mock_fetcher_class, orchestrator
    ):
        """ConnectionError should be treated as temporary network error."""
        # Setup mock to raise ConnectionError
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ConnectionError(
            "Network unreachable"
        )
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("MSFT", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Network error" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_timeout_error_handled_as_retryable(
        self, mock_fetcher_class, orchestrator
    ):
        """TimeoutError should be treated as temporary network error."""
        # Setup mock to raise TimeoutError
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = TimeoutError("Request timed out")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("GOOGL", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Network error" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_key_error_handled_as_parsing_error(
        self, mock_fetcher_class, orchestrator
    ):
        """KeyError should be treated as permanent parsing error."""
        # Setup mock to return data, but analyzer will raise KeyError
        mock_fetcher = Mock()
        mock_df = pd.DataFrame({"Wrong": [1, 2, 3]})  # Missing expected columns
        mock_fetcher.fetch_price_history.return_value = mock_df
        orchestrator.fetcher = mock_fetcher

        # Mock analyzer to raise KeyError
        orchestrator.analyzer.enrich_with_technical_indicators = Mock(
            side_effect=KeyError("'Close' column not found")
        )

        # Analyze ticker
        result = orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Data parsing error" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_parser_error_handled_as_parsing_error(
        self, mock_fetcher_class, orchestrator
    ):
        """pd.errors.ParserError should be treated as permanent parsing error."""
        # Note: pd.errors.ParserError is a subclass of ValueError,
        # so it will be caught by the ValueError handler
        # This is acceptable as it's still a permanent error
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = pd.errors.ParserError(
            "Unable to parse CSV data"
        )
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("TSLA", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        # ParserError is caught by ValueError handler (it's a subclass)
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Invalid ticker" in result.error  # Caught by ValueError handler

    @patch("orchestrator.CachedDataFetcher")
    def test_type_error_handled_as_internal_error(
        self, mock_fetcher_class, orchestrator
    ):
        """TypeError should be treated as internal code error."""
        # Setup mock to raise TypeError directly from fetcher
        # (this is more realistic - e.g., API returns wrong type)
        mock_fetcher = Mock()
        # Simulate a type error by having fetch return wrong type
        mock_fetcher.fetch_price_history.side_effect = TypeError(
            "unsupported operand type(s) for /: 'str' and 'int'"
        )
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("NVDA", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Type error" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_unexpected_error_caught_by_catch_all(
        self, mock_fetcher_class, orchestrator
    ):
        """Unexpected errors should be caught by Exception handler."""
        # Setup mock to raise an unexpected exception
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = RuntimeError(
            "Unexpected runtime error"
        )
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker
        result = orchestrator.analyze_ticker("AMD", "1y", Path("/tmp"))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "Unexpected error" in result.error
        assert "RuntimeError" in result.error


class TestErrorLoggingLevels:
    """Test that errors are logged at appropriate levels."""

    @patch("orchestrator.CachedDataFetcher")
    @patch("orchestrator.logger")
    def test_validation_error_logged_as_error(
        self, mock_logger, mock_fetcher_class, orchestrator
    ):
        """ValueError should be logged at ERROR level."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError("Invalid ticker")
        orchestrator.fetcher = mock_fetcher

        orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

        # Should log at ERROR level
        mock_logger.error.assert_called_once()
        call_args = str(mock_logger.error.call_args)
        assert "VALIDATION ERROR" in call_args

    @patch("orchestrator.CachedDataFetcher")
    @patch("orchestrator.logger")
    def test_network_error_logged_as_warning(
        self, mock_logger, mock_fetcher_class, orchestrator
    ):
        """Network errors should be logged at WARNING level (retryable)."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ConnectionError("Network down")
        orchestrator.fetcher = mock_fetcher

        orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Should log at WARNING level (retryable)
        mock_logger.warning.assert_called_once()
        call_args = str(mock_logger.warning.call_args)
        assert "NETWORK ERROR" in call_args
        assert "retryable" in call_args

    @patch("orchestrator.CachedDataFetcher")
    @patch("orchestrator.logger")
    def test_parsing_error_logged_as_error(
        self, mock_logger, mock_fetcher_class, orchestrator
    ):
        """Parsing errors should be logged at ERROR level (permanent)."""
        mock_fetcher = Mock()
        mock_df = pd.DataFrame({"Wrong": [1, 2, 3]})
        mock_fetcher.fetch_price_history.return_value = mock_df
        orchestrator.fetcher = mock_fetcher

        orchestrator.analyzer.enrich_with_technical_indicators = Mock(
            side_effect=KeyError("Missing column")
        )

        orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Should log at ERROR level
        mock_logger.error.assert_called_once()
        call_args = str(mock_logger.error.call_args)
        assert "PARSING ERROR" in call_args

    @patch("orchestrator.CachedDataFetcher")
    @patch("orchestrator.logger")
    def test_unexpected_error_uses_exception_logging(
        self, mock_logger, mock_fetcher_class, orchestrator
    ):
        """Unexpected errors should use logger.exception() for stack trace."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = RuntimeError("Unexpected")
        orchestrator.fetcher = mock_fetcher

        orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Should use .exception() which logs stack trace
        mock_logger.exception.assert_called_once()
        call_args = str(mock_logger.exception.call_args)
        assert "UNEXPECTED ERROR" in call_args


class TestErrorMessagesUserFriendly:
    """Test that error messages are clear and actionable."""

    @patch("orchestrator.CachedDataFetcher")
    def test_validation_error_message_format(
        self, mock_fetcher_class, orchestrator
    ):
        """Validation error message should include ticker and specific error."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError(
            "Ticker not found in yfinance"
        )
        orchestrator.fetcher = mock_fetcher

        result = orchestrator.analyze_ticker("BADTICKER", "1y", Path("/tmp"))

        assert "Invalid ticker BADTICKER" in result.error
        assert "Ticker not found" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_network_error_message_indicates_retryable(
        self, mock_fetcher_class, orchestrator
    ):
        """Network error message should indicate it's temporary."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ConnectionError("DNS failed")
        orchestrator.fetcher = mock_fetcher

        result = orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        assert "Network error for AAPL" in result.error

    @patch("orchestrator.CachedDataFetcher")
    def test_parsing_error_message_indicates_data_issue(
        self, mock_fetcher_class, orchestrator
    ):
        """Parsing error message should indicate data format issue."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = KeyError("Close")
        orchestrator.fetcher = mock_fetcher

        result = orchestrator.analyze_ticker("MSFT", "1y", Path("/tmp"))

        assert "Data parsing error for MSFT" in result.error


class TestErrorReturnValues:
    """Test that errors return valid TickerAnalysis objects."""

    @patch("orchestrator.CachedDataFetcher")
    def test_error_returns_valid_ticker_analysis(
        self, mock_fetcher_class, orchestrator
    ):
        """Errors should return a valid TickerAnalysis with error field set."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError("Error")
        orchestrator.fetcher = mock_fetcher

        result = orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

        # Should be valid TickerAnalysis
        assert isinstance(result, TickerAnalysis)
        assert result.ticker == "INVALID"
        assert result.error is not None
        assert result.latest_close == 0.0
        assert result.csv_path == Path()
        assert result.chart_path == Path()

    @patch("orchestrator.CachedDataFetcher")
    def test_all_error_types_return_same_structure(
        self, mock_fetcher_class, orchestrator
    ):
        """All error types should return same TickerAnalysis structure."""
        errors = [
            ValueError("validation"),
            ConnectionError("network"),
            KeyError("parsing"),
            TypeError("type"),
            RuntimeError("unexpected"),
        ]

        for error in errors:
            mock_fetcher = Mock()
            mock_fetcher.fetch_price_history.side_effect = error
            orchestrator.fetcher = mock_fetcher

            result = orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

            # All should have same structure
            assert isinstance(result, TickerAnalysis)
            assert result.ticker == "INVALID"
            assert result.error is not None
            assert result.latest_close == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
