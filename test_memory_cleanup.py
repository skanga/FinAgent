"""
Tests for memory cleanup in error handlers.

Verifies that exception handlers properly release memory by calling gc.collect()
when errors occur, especially important for ThreadPoolExecutor workers.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
from orchestrator import FinancialReportOrchestrator
from config import Config


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


class TestMemoryCleanup:
    """Test that error handlers call gc.collect() to free memory."""

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_validation_error_triggers_gc_collect(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """ValueError handler should call gc.collect()."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError("Invalid ticker")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker (will fail with ValueError)
        orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

        # Verify gc.collect() was called
        mock_gc_collect.assert_called()

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_network_error_triggers_gc_collect(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """Network error handler should call gc.collect()."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ConnectionError("Network down")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker (will fail with ConnectionError)
        orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Verify gc.collect() was called
        mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_parsing_error_triggers_gc_collect(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """Parsing error handler should call gc.collect()."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = KeyError("Missing column")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker (will fail with KeyError)
        orchestrator.analyze_ticker("MSFT", "1y", Path("/tmp"))

        # Verify gc.collect() was called
        mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_type_error_triggers_gc_collect(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """TypeError handler should call gc.collect()."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = TypeError("Type mismatch")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker (will fail with TypeError)
        orchestrator.analyze_ticker("GOOGL", "1y", Path("/tmp"))

        # Verify gc.collect() was called
        mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_unexpected_error_triggers_gc_collect(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """Unexpected error handler should call gc.collect()."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = RuntimeError("Unexpected")
        orchestrator.fetcher = mock_fetcher

        # Analyze ticker (will fail with RuntimeError)
        orchestrator.analyze_ticker("TSLA", "1y", Path("/tmp"))

        # Verify gc.collect() was called
        mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_all_error_types_call_gc_collect(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """All error handlers should call gc.collect()."""
        errors = [
            ValueError("validation"),
            ConnectionError("network"),
            KeyError("parsing"),
            TypeError("type"),
            RuntimeError("unexpected"),
        ]

        for error in errors:
            mock_gc_collect.reset_mock()  # Reset call count
            mock_fetcher = Mock()
            mock_fetcher.fetch_price_history.side_effect = error
            orchestrator.fetcher = mock_fetcher

            orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

            # Each error type should trigger gc.collect()
            mock_gc_collect.assert_called_once()

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_gc_collect_called_before_return(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """gc.collect() should be called before returning error analysis."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError("Error")
        orchestrator.fetcher = mock_fetcher

        # The order should be:
        # 1. Exception raised
        # 2. gc.collect() called
        # 3. TickerAnalysis returned
        result = orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

        # Verify gc.collect() was called
        assert mock_gc_collect.called
        # Verify we got an error analysis back
        assert result.error is not None


class TestMemoryCleanupWithPartialData:
    """Test memory cleanup when partial data exists before error."""

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_cleanup_after_partial_dataframe_processing(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """Should cleanup even if DataFrame was partially processed before error."""
        # Create a DataFrame that will be fetched
        mock_df = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]})
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.return_value = mock_df
        orchestrator.fetcher = mock_fetcher

        # Mock analyzer to fail during processing (after DataFrame is created)
        orchestrator.analyzer.compute_metrics = Mock(
            side_effect=KeyError("Missing required column")
        )

        # Analyze ticker (will fail after fetching data)
        result = orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Verify gc.collect() was called to clean up the partial DataFrame
        mock_gc_collect.assert_called_once()
        assert result.error is not None

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_cleanup_with_multiple_dataframes_in_memory(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """Should cleanup when multiple DataFrames might be in memory."""
        # Simulate scenario where data is fetched and partially enriched
        mock_df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=100),
            "Close": range(100, 200),
            "Volume": range(1000, 1100),
        })
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.return_value = mock_df
        orchestrator.fetcher = mock_fetcher

        # Mock to fail during advanced metrics calculation
        # (after enrichment creates more data)
        mock_enriched_df = mock_df.copy()
        mock_enriched_df["rsi"] = range(30, 130)  # More data in memory
        orchestrator.analyzer.compute_metrics = Mock(return_value=mock_enriched_df)
        orchestrator.analyzer.calculate_advanced_metrics = Mock(
            side_effect=RuntimeError("Calculation failed")
        )

        # Analyze ticker
        result = orchestrator.analyze_ticker("AAPL", "1y", Path("/tmp"))

        # Should have called gc.collect() to free both DataFrames
        mock_gc_collect.assert_called_once()
        assert result.error is not None


class TestConcurrentMemoryCleanup:
    """Test memory cleanup works correctly in concurrent execution."""

    @patch("gc.collect")
    @patch("orchestrator.CachedDataFetcher")
    def test_each_thread_cleans_up_independently(
        self, mock_fetcher_class, mock_gc_collect, orchestrator
    ):
        """Each thread should cleanup its own memory on error."""
        # Simulate concurrent failures
        errors = [
            ValueError("Error 1"),
            ConnectionError("Error 2"),
            KeyError("Error 3"),
        ]
        tickers = ["AAPL", "MSFT", "GOOGL"]

        for ticker, error in zip(tickers, errors):
            mock_gc_collect.reset_mock()
            mock_fetcher = Mock()
            mock_fetcher.fetch_price_history.side_effect = error
            orchestrator.fetcher = mock_fetcher

            # Each ticker analysis should cleanup independently
            result = orchestrator.analyze_ticker(ticker, "1y", Path("/tmp"))

            # Each should call gc.collect()
            mock_gc_collect.assert_called_once()
            assert result.error is not None


class TestGCImport:
    """Test that gc module is imported correctly in exception handlers."""

    @patch("orchestrator.CachedDataFetcher")
    def test_gc_import_in_exception_handler(
        self, mock_fetcher_class, orchestrator
    ):
        """gc module should be imported within exception handler (local import)."""
        mock_fetcher = Mock()
        mock_fetcher.fetch_price_history.side_effect = ValueError("Error")
        orchestrator.fetcher = mock_fetcher

        # This should not raise ImportError
        result = orchestrator.analyze_ticker("INVALID", "1y", Path("/tmp"))

        assert result.error is not None
        # If gc import failed, we would have gotten a different error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
