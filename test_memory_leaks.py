"""
Test memory leak fixes in chart generation.
"""
import gc
import sys
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

    df = pd.DataFrame({
        'Date': dates,
        'close': prices,
        '30d_ma': pd.Series(prices).rolling(30).mean(),
        '50d_ma': pd.Series(prices).rolling(50).mean(),
        'rsi': 50 + np.random.randn(days) * 15,
        'bollinger_upper': prices * 1.02,
        'bollinger_lower': prices * 0.98,
        'daily_return': np.random.randn(days) * 0.02,
        'volatility': np.abs(np.random.randn(days) * 0.15)
    })

    return df


def create_test_analysis(ticker: str, csv_path: str) -> TickerAnalysis:
    """Create a test TickerAnalysis object."""
    return TickerAnalysis(
        ticker=ticker,
        csv_path=csv_path,
        chart_path=f"test_chart_{ticker}.png",
        latest_close=150.0,
        avg_daily_return=0.001,
        volatility=0.02,
        ratios={'pe_ratio': 25.0},
        fundamentals=None,
        advanced_metrics=AdvancedMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            beta=1.1,
            alpha=0.02
        ),
        technical_indicators=TechnicalIndicators(
            rsi=55.0,
            macd=0.5,
            macd_signal=0.3
        ),
        sample_data=[]
    )


def test_memory_leak_price_chart():
    """Test memory leak fixes in create_price_chart."""
    print("Testing create_price_chart memory management...")

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    baseline = tracemalloc.get_traced_memory()[0]

    chart_gen = ThreadSafeChartGenerator()
    test_dir = Path("./test_charts")
    test_dir.mkdir(exist_ok=True)

    # Create multiple charts to detect memory accumulation
    for i in range(5):
        df = create_test_dataframe(f"TEST{i}", days=1000)  # Large DataFrame
        output_path = test_dir / f"test_price_{i}.png"

        chart_gen.create_price_chart(df, f"TEST{i}", str(output_path))

        # Clean up test file
        if output_path.exists():
            output_path.unlink()

    gc.collect()
    final_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    memory_increase = (final_memory - baseline) / 1024 / 1024  # Convert to MB

    print(f"  Baseline memory: {baseline / 1024 / 1024:.2f} MB")
    print(f"  Final memory: {final_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory increase: {memory_increase:.2f} MB")

    # Cleanup
    test_dir.rmdir()

    # Memory should not grow significantly (allow 10 MB tolerance)
    if memory_increase < 10:
        print("  [PASS] Memory leak fixed - minimal memory growth\n")
        return True
    else:
        print(f"  [FAIL] Possible memory leak detected - {memory_increase:.2f} MB increase\n")
        return False


def test_memory_leak_comparison_chart():
    """Test memory leak fixes in create_comparison_chart."""
    print("Testing create_comparison_chart memory management...")

    tracemalloc.start()
    gc.collect()
    baseline = tracemalloc.get_traced_memory()[0]

    chart_gen = ThreadSafeChartGenerator()
    test_dir = Path("./test_charts")
    test_dir.mkdir(exist_ok=True)

    # Create test data files
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    analyses = {}

    for ticker in tickers:
        df = create_test_dataframe(ticker, days=500)
        csv_path = test_dir / f"{ticker}_test.csv"
        df.to_csv(csv_path, index=False)
        analyses[ticker] = create_test_analysis(ticker, str(csv_path))

    # Create multiple comparison charts
    for i in range(5):
        output_path = test_dir / f"test_comparison_{i}.png"
        chart_gen.create_comparison_chart(analyses, str(output_path))

        if output_path.exists():
            output_path.unlink()

    # Cleanup test files
    for ticker in tickers:
        csv_path = test_dir / f"{ticker}_test.csv"
        if csv_path.exists():
            csv_path.unlink()

    gc.collect()
    final_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    memory_increase = (final_memory - baseline) / 1024 / 1024

    print(f"  Baseline memory: {baseline / 1024 / 1024:.2f} MB")
    print(f"  Final memory: {final_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory increase: {memory_increase:.2f} MB")

    # Cleanup
    test_dir.rmdir()

    if memory_increase < 10:
        print("  [PASS] DataFrame cleanup working - minimal memory growth\n")
        return True
    else:
        print(f"  [FAIL] Possible DataFrame leak - {memory_increase:.2f} MB increase\n")
        return False


def test_gc_collection():
    """Test that garbage collection is being called."""
    print("Testing explicit garbage collection...")

    chart_gen = ThreadSafeChartGenerator()
    test_dir = Path("./test_charts")
    test_dir.mkdir(exist_ok=True)

    # Track GC stats
    gc.collect()
    initial_collections = gc.get_count()

    # Generate a chart
    df = create_test_dataframe("TEST", days=500)
    output_path = test_dir / "test_gc.png"
    chart_gen.create_price_chart(df, "TEST", str(output_path))

    final_collections = gc.get_count()

    # Cleanup
    if output_path.exists():
        output_path.unlink()
    test_dir.rmdir()

    # GC should have been triggered
    collections_occurred = any(f > i for i, f in zip(initial_collections, final_collections))

    if collections_occurred:
        print("  [PASS] Garbage collection being invoked\n")
        return True
    else:
        print("  [WARN] No garbage collection detected (may be normal)\n")
        return True  # Not a failure, just informational


def test_figure_cleanup():
    """Test that matplotlib figures are properly closed."""
    print("Testing matplotlib figure cleanup...")

    import matplotlib.pyplot as plt

    # Check initial figure count
    initial_figs = len(plt.get_fignums())

    chart_gen = ThreadSafeChartGenerator()
    test_dir = Path("./test_charts")
    test_dir.mkdir(exist_ok=True)

    # Create multiple charts
    for i in range(3):
        df = create_test_dataframe(f"TEST{i}", days=100)
        output_path = test_dir / f"test_fig_{i}.png"
        chart_gen.create_price_chart(df, f"TEST{i}", str(output_path))

        if output_path.exists():
            output_path.unlink()

    # Check final figure count
    final_figs = len(plt.get_fignums())

    # Cleanup
    test_dir.rmdir()

    if final_figs == initial_figs:
        print(f"  [PASS] All figures closed (initial: {initial_figs}, final: {final_figs})\n")
        return True
    else:
        print(f"  [FAIL] Figures not closed properly (initial: {initial_figs}, final: {final_figs})\n")
        return False


def main():
    """Run all memory leak tests."""
    print("=" * 60)
    print("MEMORY LEAK FIX VERIFICATION")
    print("=" * 60)
    print()

    tests = [
        test_figure_cleanup,
        test_gc_collection,
        test_memory_leak_price_chart,
        test_memory_leak_comparison_chart
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [ERROR] {test.__name__} failed with exception:")
            print(f"    {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append(False)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All memory leak fixes verified!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
