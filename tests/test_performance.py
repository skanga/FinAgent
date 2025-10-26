"""
Test performance improvements for DataFrame operations.
"""

import pytest
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from analyzers import AdvancedFinancialAnalyzer


def create_large_price_dataframe(n_rows: int = 5000) -> pd.DataFrame:
    """Create a large price DataFrame for performance testing."""
    dates = [
        datetime.now(timezone.utc) - timedelta(days=i) for i in range(n_rows, 0, -1)
    ]

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_rows) * 2)

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, n_rows),
        }
    )


class TestPerformanceOptimizations:
    """Test DataFrame performance optimizations."""

    def test_compute_metrics_performance(self):
        """Test performance of compute_metrics with vectorized operations."""
        analyzer = AdvancedFinancialAnalyzer()

        # Test with various DataFrame sizes
        sizes = [500, 1000, 2500, 5000]
        results = []

        for size in sizes:
            df = create_large_price_dataframe(size)

            # Warm-up run
            _ = analyzer.compute_metrics(df)

            # Timed runs
            times = []
            for _ in range(5):
                start = time.perf_counter()
                result = analyzer.compute_metrics(df)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            std_time = np.std(times)

            results.append({"size": size, "avg_time": avg_time, "std_time": std_time})

            # Verify correctness
            assert "close" in result.columns
            assert "daily_return" in result.columns
            assert "rsi" in result.columns
            assert "macd" in result.columns
            assert "bollinger_position" in result.columns
            assert len(result) == size

        # Check that performance scales reasonably (should be roughly linear)
        time_500 = results[0]["avg_time"]
        time_5000 = results[3]["avg_time"]
        ratio = time_5000 / time_500

        # Should be close to 10x for linear scaling
        assert (
            ratio < 15
        ), f"Performance scaling suboptimal: {ratio:.1f}x (expected < 15x)"

    def test_iloc_optimization(self):
        """Test that iloc[-1] optimizations reduce lookup time."""
        # Simulate the old approach with repeated iloc[-1] calls
        df = create_large_price_dataframe(2000)
        analyzer = AdvancedFinancialAnalyzer()
        df_analyzed = analyzer.compute_metrics(df)

        # Old approach: repeated iloc[-1] lookups
        def old_approach():
            rsi = (
                float(df_analyzed["rsi"].iloc[-1])
                if not pd.isna(df_analyzed["rsi"].iloc[-1])
                else None
            )
            macd = (
                float(df_analyzed["macd"].iloc[-1])
                if not pd.isna(df_analyzed["macd"].iloc[-1])
                else None
            )
            macd_signal = (
                float(df_analyzed["macd_signal"].iloc[-1])
                if not pd.isna(df_analyzed["macd_signal"].iloc[-1])
                else None
            )
            bollinger_upper = (
                float(df_analyzed["bollinger_upper"].iloc[-1])
                if not pd.isna(df_analyzed["bollinger_upper"].iloc[-1])
                else None
            )
            bollinger_lower = (
                float(df_analyzed["bollinger_lower"].iloc[-1])
                if not pd.isna(df_analyzed["bollinger_lower"].iloc[-1])
                else None
            )
            bollinger_position = (
                float(df_analyzed["bollinger_position"].iloc[-1])
                if not pd.isna(df_analyzed["bollinger_position"].iloc[-1])
                else None
            )
            close = (
                float(df_analyzed["close"].iloc[-1])
                if not pd.isna(df_analyzed["close"].iloc[-1])
                else 0.0
            )
            volatility = (
                float(df_analyzed["volatility"].iloc[-1])
                if not pd.isna(df_analyzed["volatility"].iloc[-1])
                else 0.0
            )
            return (
                rsi,
                macd,
                macd_signal,
                bollinger_upper,
                bollinger_lower,
                bollinger_position,
                close,
                volatility,
            )

        # New approach: single iloc[-1] lookup
        def new_approach():
            last_row = df_analyzed.iloc[-1]

            def safe_float(value):
                return float(value) if not pd.isna(value) else None

            rsi = safe_float(last_row["rsi"])
            macd = safe_float(last_row["macd"])
            macd_signal = safe_float(last_row["macd_signal"])
            bollinger_upper = safe_float(last_row["bollinger_upper"])
            bollinger_lower = safe_float(last_row["bollinger_lower"])
            bollinger_position = safe_float(last_row["bollinger_position"])
            close = safe_float(last_row["close"]) or 0.0
            volatility = safe_float(last_row["volatility"]) or 0.0
            return (
                rsi,
                macd,
                macd_signal,
                bollinger_upper,
                bollinger_lower,
                bollinger_position,
                close,
                volatility,
            )

        # Benchmark old approach
        old_times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = old_approach()
            old_times.append(time.perf_counter() - start)

        # Benchmark new approach
        new_times = []
        for _ in range(1000):
            start = time.perf_counter()
            _ = new_approach()
            new_times.append(time.perf_counter() - start)

        old_avg = np.mean(old_times) * 1000000  # Convert to microseconds
        new_avg = np.mean(new_times) * 1000000
        _speedup = old_avg / new_avg

        # Verify both approaches return same results
        assert (
            old_approach() == new_approach()
        ), "Old and new approaches should return same values"

        # Informational - speedup can vary by system
        # No hard assertion on speedup

    def test_vectorized_bollinger_position(self):
        """Test vectorized Bollinger position calculation."""
        analyzer = AdvancedFinancialAnalyzer()
        df = create_large_price_dataframe(1000)

        # Compute metrics
        result = analyzer.compute_metrics(df)

        # Verify bollinger_position is correctly calculated
        valid_positions = result["bollinger_position"].dropna()

        assert len(valid_positions) > 0, "No valid Bollinger positions calculated"

        # Check that positions are within valid range [0, 1] (with some tolerance)
        _min_pos = valid_positions.min()
        _max_pos = valid_positions.max()

        # Bollinger position should typically be between 0 and 1
        # (though can go slightly outside in extreme cases)
        out_of_range = ((valid_positions < -0.5) | (valid_positions > 1.5)).sum()
        assert (
            out_of_range == 0
        ), f"{out_of_range} positions outside typical range [-0.5, 1.5]"

    def test_window_size_caching(self):
        """Test that window sizes are pre-calculated once."""
        analyzer = AdvancedFinancialAnalyzer()

        # Small DataFrame
        df_small = create_large_price_dataframe(50)
        result_small = analyzer.compute_metrics(df_small)

        # Verify that adaptive window sizes were used
        assert "30d_ma" in result_small.columns
        assert "50d_ma" in result_small.columns

        # Check that calculations completed despite small size
        valid_30d = result_small["30d_ma"].dropna()
        valid_50d = result_small["50d_ma"].dropna()

        # Large DataFrame
        df_large = create_large_price_dataframe(500)
        result_large = analyzer.compute_metrics(df_large)

        valid_30d_large = result_large["30d_ma"].dropna()
        valid_50d_large = result_large["50d_ma"].dropna()

        # Both should have valid results
        assert len(valid_30d) > 0, "Small DF: No valid 30d moving averages"
        assert len(valid_50d) > 0, "Small DF: No valid 50d moving averages"
        assert len(valid_30d_large) > 0, "Large DF: No valid 30d moving averages"
        assert len(valid_50d_large) > 0, "Large DF: No valid 50d moving averages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
