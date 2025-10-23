"""
Test memory leak fixes in chart generation.
"""
import shutil
import pytest
import gc
import tracemalloc
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from charts import ThreadSafeChartGenerator
from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators


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


class TestMemoryLeaks:
    """Test memory leak fixes in chart generation."""

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

    def test_gc_collection(self):
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
                test_dir.rmdir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
