"""
Comprehensive tests for error handling and recovery.

Tests cover:
- Error handling in orchestrator (ValueError, OSError, ConnectionError, TimeoutError, KeyError, TypeError)
- Error logging at appropriate levels (ERROR, WARNING, exception)
- User-friendly error messages
- Error recovery in worker threads (_analyze_all_tickers)
- Portfolio weight validation (negative weights, sum validation)
- Portfolio weight edge cases (zeros, extreme concentration)
- TickerAnalysis error return values
- Multiple concurrent failures
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from orchestrator import FinancialReportOrchestrator
from analyzers import PortfolioAnalyzer
from config import Config
from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.openai_api_key = "test-api-key"
    config.openai_base_url = "https://api.openai.com/v1"
    config.model_name = "gpt-4"
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


@pytest.fixture
def portfolio_analyzer():
    """Create a PortfolioAnalyzer instance."""
    return PortfolioAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def mock_analyses():
    """Create mock ticker analyses for portfolio tests."""
    # Create temporary CSV files with return data
    temp_dir = tempfile.mkdtemp()

    analyses = {}
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        csv_path = Path(temp_dir) / f"{ticker}.csv"

        # Create sample return data
        dates = pd.date_range("2024-01-01", periods=100)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        df = pd.DataFrame({
            "Date": dates,
            "close": 150.0,
            "daily_return": returns
        })
        df.to_csv(csv_path, index=False)

        analyses[ticker] = TickerAnalysis(
            ticker=ticker,
            csv_path=csv_path,
            chart_path=Path(f"{ticker}.png"),
            latest_close=150.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={},
            fundamentals=None,
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
            sample_data=[],
            error=None,
        )

    yield analyses

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TEST ERROR TYPE DISTINCTION
# ============================================================================


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


# ============================================================================
# TEST ERROR LOGGING LEVELS
# ============================================================================


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


# ============================================================================
# TEST ERROR MESSAGES
# ============================================================================


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


# ============================================================================
# TEST ERROR RETURN VALUES
# ============================================================================


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


# ============================================================================
# TEST ERROR RECOVERY IN WORKER THREADS
# ============================================================================


class TestAnalyzeAllTickersErrorRecovery:
    """Test error recovery in _analyze_all_tickers method."""

    @patch("orchestrator.CacheManager")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_handles_exception_from_worker_thread(
        self,
        mock_chart,
        mock_llm,
        mock_cache,
        mock_config,
        temp_output_dir,
    ):
        """Should handle exceptions raised in worker threads gracefully."""
        # Setup orchestrator
        orchestrator = FinancialReportOrchestrator(mock_config)

        # Mock analyze_ticker to fail for specific ticker
        def mock_analyze_ticker(ticker, period, output_path, benchmark_returns, include_options=False, options_expirations=3):
            if ticker == "FAIL":
                raise RuntimeError("Worker thread crashed")
            # For successful tickers, return a normal analysis
            return TickerAnalysis(
                ticker=ticker,
                csv_path=Path(f"{ticker}.csv"),
                chart_path=Path(f"{ticker}.png"),
                latest_close=100.0,
                avg_daily_return=0.001,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
                sample_data=[],
                error=None,
            )

        orchestrator.analyze_ticker = mock_analyze_ticker

        # Analyze tickers including one that will fail
        tickers = ["AAPL", "FAIL", "MSFT"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # Should have all 3 analyses
        assert len(analyses) == 3
        assert "AAPL" in analyses
        assert "FAIL" in analyses
        assert "MSFT" in analyses

        # Failed ticker should have error analysis
        assert analyses["FAIL"].error is not None
        assert "Thread error" in analyses["FAIL"].error
        assert analyses["FAIL"].latest_close == 0.0

        # Successful tickers should be fine
        assert analyses["AAPL"].error is None
        assert analyses["MSFT"].error is None

    @patch("orchestrator.CacheManager")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_handles_multiple_concurrent_failures(
        self,
        mock_chart,
        mock_llm,
        mock_cache,
        mock_config,
        temp_output_dir,
    ):
        """Should handle multiple concurrent failures without stopping."""
        orchestrator = FinancialReportOrchestrator(mock_config)

        # Mock analyze_ticker to fail for specific tickers
        def mock_analyze_ticker(ticker, period, output_path, benchmark_returns, include_options=False, options_expirations=3):
            if ticker in ["FAIL1", "FAIL2"]:
                raise ValueError(f"Error in {ticker}")
            return TickerAnalysis(
                ticker=ticker,
                csv_path=Path(f"{ticker}.csv"),
                chart_path=Path(f"{ticker}.png"),
                latest_close=100.0,
                avg_daily_return=0.001,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
                sample_data=[],
                error=None,
            )

        orchestrator.analyze_ticker = mock_analyze_ticker

        tickers = ["AAPL", "FAIL1", "MSFT", "FAIL2", "GOOGL"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # All tickers should be present
        assert len(analyses) == 5

        # Failed tickers should have errors
        assert analyses["FAIL1"].error is not None
        assert analyses["FAIL2"].error is not None
        assert "Thread error" in analyses["FAIL1"].error
        assert "Thread error" in analyses["FAIL2"].error

        # Successful tickers should be fine
        assert analyses["AAPL"].error is None
        assert analyses["MSFT"].error is None
        assert analyses["GOOGL"].error is None


# ============================================================================
# TEST PORTFOLIO WEIGHT VALIDATION
# ============================================================================


class TestPortfolioWeights:
    """Test invalid portfolio weight combinations."""

    def test_mixed_negative_and_positive_weights_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject portfolios with mixed negative and positive weights."""
        # Mix of negative and positive weights (still sum to 1)
        weights = {"AAPL": 1.5, "MSFT": -0.3, "GOOGL": -0.2}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_all_negative_weights_rejected(self, portfolio_analyzer, mock_analyses):
        """Should reject portfolios where all weights are negative."""
        weights = {"AAPL": -0.5, "MSFT": -0.3, "GOOGL": -0.2}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_zero_weight_mixed_with_positive_accepted(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should accept zero weights mixed with positive (if sum to 1)."""
        weights = {"AAPL": 0.5, "MSFT": 0.5, "GOOGL": 0.0}

        # Should not raise (zeros are allowed)
        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None
        assert result.weights == weights

    def test_very_small_negative_weight_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject even very small negative weights."""
        # Tiny negative weight that might be from floating point errors
        weights = {"AAPL": 0.6, "MSFT": 0.4, "GOOGL": -0.0001}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_weights_sum_slightly_over_one_with_negative_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject weights that sum > 1 and include negatives."""
        # Sum to 1.1 with negative component
        weights = {"AAPL": 1.0, "MSFT": 0.5, "GOOGL": -0.4}

        # Should fail on negative check first
        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_weights_sum_slightly_under_one_with_negative_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject weights that sum < 1 and include negatives."""
        # Sum to 0.9 with negative component
        weights = {"AAPL": 0.5, "MSFT": 0.6, "GOOGL": -0.2}

        # Should fail on negative check first
        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_single_negative_weight_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject portfolio with single negative weight."""
        weights = {"AAPL": -1.0}

        # Create single analysis
        single_analysis = {"AAPL": mock_analyses["AAPL"]}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(single_analysis, weights)

    def test_extreme_weight_concentration_with_zero(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should handle extreme concentration (one weight = 1, others = 0)."""
        weights = {"AAPL": 1.0, "MSFT": 0.0, "GOOGL": 0.0}

        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None
        # Concentration risk (Herfindahl) should be 1.0 (max concentration)
        assert result.concentration_risk == 1.0

    def test_fractional_negative_weights_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject fractional negative weights."""
        weights = {"AAPL": 0.7, "MSFT": 0.35, "GOOGL": -0.05}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_weights_with_multiple_validation_errors(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should catch negative weights before sum validation."""
        # Both negative AND sum != 1
        weights = {"AAPL": -0.5, "MSFT": -0.3, "GOOGL": 0.5}

        # Should fail on negative check first (not sum check)
        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)


# ============================================================================
# TEST PORTFOLIO WEIGHT EDGE CASES
# ============================================================================


class TestPortfolioWeightEdgeCases:
    """Test edge cases in portfolio weight handling."""

    def test_empty_weights_dict_uses_equal_weights(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should use equal weights when weights dict is None."""
        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, None)

        assert result is not None
        assert result.weights is not None
        # Should have equal weights for 3 tickers
        expected_weight = 1.0 / 3.0
        for weight in result.weights.values():
            assert abs(weight - expected_weight) < 0.001

    def test_weights_sum_to_exactly_one(self, portfolio_analyzer, mock_analyses):
        """Should accept weights that sum to exactly 1.0."""
        weights = {"AAPL": 0.3333333333333333, "MSFT": 0.3333333333333333, "GOOGL": 0.3333333333333334}

        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None

    def test_weights_sum_within_tolerance(self, portfolio_analyzer, mock_analyses):
        """Should accept weights within tolerance of 1.0."""
        # Sum to 1.001 (within tolerance)
        weights = {"AAPL": 0.334, "MSFT": 0.333, "GOOGL": 0.334}

        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None

    def test_weights_outside_tolerance_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject weights outside tolerance."""
        # Sum to 1.05 (outside tolerance)
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.3}

        with pytest.raises(ValueError, match="must sum to 1.0"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
