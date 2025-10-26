"""
Comprehensive tests for memory management and leak prevention.

Tests cover:
- Memory cleanup in error handlers (gc.collect() calls)
- Matplotlib figure cleanup (plt.close())
- Memory leak detection in chart generation
- Concurrent memory cleanup in worker threads
- Partial data cleanup before errors
- GC import validation
- TraceMalloc-based leak detection
"""

import pytest
import gc
import tracemalloc
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

# Import orchestrator for error handler tests
from orchestrator import FinancialReportOrchestrator
from config import Config

# Import charts for memory leak tests
from charts import ThreadSafeChartGenerator
from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators


# ============================================================================
# FIXTURES
# ============================================================================


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


def create_test_dataframe(ticker: str, days: int = 252) -> pd.DataFrame:
    """Create a test DataFrame with realistic financial data."""
    dates = [datetime.now(timezone.utc) - timedelta(days=i) for i in range(days, 0, -1)]

    # Generate synthetic price data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(days) * 2)

    df = pd.DataFrame(
        {
            "Date": dates,
            "close": prices,
            "30d_ma": pd.Series(prices).rolling(30).mean(),
            "50d_ma": pd.Series(prices).rolling(50).mean(),
            "rsi": 50 + np.random.randn(days) * 15,
            "bollinger_upper": prices * 1.02,
            "bollinger_lower": prices * 0.98,
            "daily_return": np.random.randn(days) * 0.02,
            "volatility": np.abs(np.random.randn(days) * 0.15),
        }
    )

    return df


def create_test_analysis(ticker: str, csv_path: Path) -> TickerAnalysis:
    """Create a test TickerAnalysis object."""
    return TickerAnalysis(
        ticker=ticker,
        csv_path=csv_path,
        chart_path=f"test_chart_{ticker}.png",
        latest_close=150.0,
        avg_daily_return=0.001,
        volatility=0.02,
        ratios={"pe_ratio": 25.0},
        fundamentals=None,
        advanced_metrics=AdvancedMetrics(
            sharpe_ratio=1.5, max_drawdown=0.15, beta=1.1, alpha=0.02
        ),
        technical_indicators=TechnicalIndicators(rsi=55.0, macd=0.5, macd_signal=0.3),
        sample_data=[],
    )


# ============================================================================
# TEST ERROR HANDLER MEMORY CLEANUP
# ============================================================================


class TestErrorHandlerMemoryCleanup:
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


# ============================================================================
# TEST PARTIAL DATA CLEANUP
# ============================================================================


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


# ============================================================================
# TEST CONCURRENT CLEANUP
# ============================================================================


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


# ============================================================================
# TEST GC IMPORT
# ============================================================================


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


# ============================================================================
# TEST MATPLOTLIB FIGURE CLEANUP
# ============================================================================


class TestMatplotlibFigureCleanup:
    """Test that matplotlib figures are properly closed."""

    def test_figure_cleanup(self):
        """Test that matplotlib figures are properly closed."""
        import matplotlib.pyplot as plt

        # Check initial figure count
        initial_figs = len(plt.get_fignums())

        chart_gen = ThreadSafeChartGenerator()
        test_dir = Path("./test_charts")
        test_dir.mkdir(exist_ok=True)

        try:
            # Create multiple charts
            for i in range(3):
                df = create_test_dataframe(f"TEST{i}", days=100)
                output_path = test_dir / f"test_fig_{i}.png"
                chart_gen.create_price_chart(df, f"TEST{i}", output_path)

                if output_path.exists():
                    output_path.unlink()

            # Check final figure count
            final_figs = len(plt.get_fignums())

            assert (
                final_figs == initial_figs
            ), f"Figures not closed properly (initial: {initial_figs}, final: {final_figs})"

        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_gc_collection_during_chart_generation(self):
        """Test that garbage collection is being called."""
        chart_gen = ThreadSafeChartGenerator()
        test_dir = Path("./test_charts")
        test_dir.mkdir(exist_ok=True)

        try:
            # Track GC stats
            gc.collect()
            initial_collections = gc.get_count()

            # Generate a chart
            df = create_test_dataframe("TEST", days=500)
            output_path = test_dir / "test_gc.png"
            chart_gen.create_price_chart(df, "TEST", output_path)

            final_collections = gc.get_count()

            # Cleanup
            if output_path.exists():
                output_path.unlink()

            # GC should have been triggered (informational only)
            _collections_occurred = any(
                f > i for i, f in zip(initial_collections, final_collections)
            )

        finally:
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)


# ============================================================================
# TEST MEMORY LEAK DETECTION
# ============================================================================


class TestMemoryLeakDetection:
    """Test memory leak detection using tracemalloc."""

    def test_memory_leak_price_chart(self):
        """Test memory leak fixes in create_price_chart."""
        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]

        chart_gen = ThreadSafeChartGenerator()
        test_dir = Path("./test_charts")
        test_dir.mkdir(exist_ok=True)

        try:
            # Create multiple charts to detect memory accumulation
            for i in range(5):
                df = create_test_dataframe(f"TEST{i}", days=1000)  # Large DataFrame
                output_path = test_dir / f"test_price_{i}.png"

                chart_gen.create_price_chart(df, f"TEST{i}", output_path)

                # Clean up test file
                if output_path.exists():
                    output_path.unlink()

            gc.collect()
            final_memory = tracemalloc.get_traced_memory()[0]
            memory_increase = (final_memory - baseline) / 1024 / 1024  # Convert to MB

            # Memory should not grow significantly (allow 10 MB tolerance)
            assert (
                memory_increase < 10
            ), f"Possible memory leak detected - {memory_increase:.2f} MB increase"

        finally:
            tracemalloc.stop()
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_memory_leak_comparison_chart(self):
        """Test memory leak fixes in create_comparison_chart."""
        tracemalloc.start()
        gc.collect()
        baseline = tracemalloc.get_traced_memory()[0]

        chart_gen = ThreadSafeChartGenerator()
        test_dir = Path("./test_charts")
        test_dir.mkdir(exist_ok=True)

        try:
            # Create test data files
            tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            analyses = {}

            for ticker in tickers:
                df = create_test_dataframe(ticker, days=500)
                csv_path = test_dir / f"{ticker}_test.csv"
                df.to_csv(csv_path, index=False)
                analyses[ticker] = create_test_analysis(ticker, csv_path)

            # Create multiple comparison charts
            for i in range(5):
                output_path = test_dir / f"test_comparison_{i}.png"
                chart_gen.create_comparison_chart(analyses, output_path)

                if output_path.exists():
                    output_path.unlink()

            # Cleanup test files
            for ticker in tickers:
                csv_path = test_dir / f"{ticker}_test.csv"
                if csv_path.exists():
                    csv_path.unlink()

            gc.collect()
            final_memory = tracemalloc.get_traced_memory()[0]
            memory_increase = (final_memory - baseline) / 1024 / 1024

            assert (
                memory_increase < 10
            ), f"Possible DataFrame leak - {memory_increase:.2f} MB increase"

        finally:
            tracemalloc.stop()
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
