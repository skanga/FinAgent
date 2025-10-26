"""
Performance benchmarks for optimization analysis.
"""

import gc
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

def benchmark_gc_collect_in_charts():
    """Benchmark the impact of gc.collect() after chart creation."""
    print("\n" + "="*80)
    print("BENCHMARK: gc.collect() impact on chart creation")
    print("="*80)

    # Create sample data
    dates = pd.date_range("2024-01-01", periods=252)
    prices = 100 + np.cumsum(np.random.randn(252) * 2)

    temp_dir = tempfile.mkdtemp()

    try:
        # Test WITHOUT gc.collect()
        times_without_gc = []
        for i in range(10):
            start = time.perf_counter()

            fig, ax = plt.subplots(figsize=(14, 10))
            ax.plot(dates, prices)
            ax.set_title(f"Test Chart {i}")
            fig.savefig(Path(temp_dir) / f"chart_no_gc_{i}.png", dpi=150)
            plt.close(fig)
            plt.close("all")

            elapsed = time.perf_counter() - start
            times_without_gc.append(elapsed)

        # Test WITH gc.collect()
        times_with_gc = []
        for i in range(10):
            start = time.perf_counter()

            fig, ax = plt.subplots(figsize=(14, 10))
            ax.plot(dates, prices)
            ax.set_title(f"Test Chart {i}")
            fig.savefig(Path(temp_dir) / f"chart_with_gc_{i}.png", dpi=150)
            plt.close(fig)
            plt.close("all")
            gc.collect()  # This is what we're testing

            elapsed = time.perf_counter() - start
            times_with_gc.append(elapsed)

        avg_without = np.mean(times_without_gc) * 1000
        avg_with = np.mean(times_with_gc) * 1000
        overhead = ((avg_with - avg_without) / avg_without) * 100

        print(f"\nWithout gc.collect(): {avg_without:.2f}ms (avg)")
        print(f"With gc.collect():    {avg_with:.2f}ms (avg)")
        print(f"Overhead:             {overhead:.1f}%")

        if overhead > 10:
            print(f"\nWARNING: gc.collect() adds {overhead:.1f}% overhead - SHOULD OPTIMIZE")
        else:
            print(f"\nOK: gc.collect() overhead is negligible ({overhead:.1f}%) - OK to keep")

        return overhead

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_cache_serialization():
    """Benchmark pickle vs parquet for DataFrame caching."""
    print("\n" + "="*80)
    print("BENCHMARK: Cache serialization (pickle vs parquet)")
    print("="*80)

    # Create sample DataFrame (typical price data size)
    dates = pd.date_range("2024-01-01", periods=252)
    df = pd.DataFrame({
        "Date": dates,
        "Open": 100 + np.random.randn(252) * 10,
        "High": 105 + np.random.randn(252) * 10,
        "Low": 95 + np.random.randn(252) * 10,
        "Close": 100 + np.random.randn(252) * 10,
        "Volume": np.random.randint(1000000, 10000000, 252),
        "daily_return": np.random.randn(252) * 0.02,
    })

    temp_dir = tempfile.mkdtemp()

    try:
        # Test pickle
        pickle_times_write = []
        pickle_times_read = []
        for i in range(20):
            # Write
            start = time.perf_counter()
            df.to_pickle(Path(temp_dir) / "test.pkl")
            pickle_times_write.append(time.perf_counter() - start)

            # Read
            start = time.perf_counter()
            _ = pd.read_pickle(Path(temp_dir) / "test.pkl")  # Load data to measure read time
            pickle_times_read.append(time.perf_counter() - start)

        # Test parquet
        parquet_times_write = []
        parquet_times_read = []
        for i in range(20):
            # Write
            start = time.perf_counter()
            df.to_parquet(Path(temp_dir) / "test.parquet")
            parquet_times_write.append(time.perf_counter() - start)

            # Read
            start = time.perf_counter()
            _ = pd.read_parquet(Path(temp_dir) / "test.parquet")  # Load data to measure read time
            parquet_times_read.append(time.perf_counter() - start)

        # Get file sizes
        pickle_size = (Path(temp_dir) / "test.pkl").stat().st_size
        parquet_size = (Path(temp_dir) / "test.parquet").stat().st_size

        print("\nPickle:")
        print(f"  Write: {np.mean(pickle_times_write)*1000:.2f}ms")
        print(f"  Read:  {np.mean(pickle_times_read)*1000:.2f}ms")
        print(f"  Size:  {pickle_size:,} bytes")

        print("\nParquet:")
        print(f"  Write: {np.mean(parquet_times_write)*1000:.2f}ms")
        print(f"  Read:  {np.mean(parquet_times_read)*1000:.2f}ms")
        print(f"  Size:  {parquet_size:,} bytes")

        write_improvement = ((np.mean(pickle_times_write) - np.mean(parquet_times_write)) / np.mean(pickle_times_write)) * 100
        read_improvement = ((np.mean(pickle_times_read) - np.mean(parquet_times_read)) / np.mean(pickle_times_read)) * 100
        size_improvement = ((pickle_size - parquet_size) / pickle_size) * 100

        print("\nParquet vs Pickle:")
        print(f"  Write: {write_improvement:+.1f}% {'faster' if write_improvement > 0 else 'slower'}")
        print(f"  Read:  {read_improvement:+.1f}% {'faster' if read_improvement > 0 else 'slower'}")
        print(f"  Size:  {size_improvement:+.1f}% {'smaller' if size_improvement > 0 else 'larger'}")

        if read_improvement > 20 or size_improvement > 20:
            print("\nOK: Parquet is significantly better - SHOULD SWITCH")
        else:
            print("\nWARNING: Parquet not significantly better - pickle is OK")

        return read_improvement, size_improvement

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def benchmark_dataframe_reuse():
    """Benchmark creating new DataFrame vs reusing existing."""
    print("\n" + "="*80)
    print("BENCHMARK: DataFrame creation vs reuse")
    print("="*80)

    # Simulate portfolio metrics calculation
    tickers = ["AAPL", "MSFT", "GOOGL"]

    # Create sample return data
    returns_data = {}
    for ticker in tickers:
        returns_data[ticker] = pd.Series(np.random.randn(252) * 0.02)

    # Method 1: Create new DataFrame (current approach)
    times_create_new = []
    for i in range(100):
        start = time.perf_counter()

        returns_df = pd.DataFrame(returns_data).dropna()
        weights_array = np.array([0.33, 0.33, 0.34])
        _ = returns_df.values @ weights_array  # Calculate returns to measure performance

        elapsed = time.perf_counter() - start
        times_create_new.append(elapsed)

    # Method 2: Reuse existing data structures
    times_reuse = []
    # Pre-create aligned data
    returns_df_cached = pd.DataFrame(returns_data).dropna()
    weights_array_cached = np.array([0.33, 0.33, 0.34])

    for i in range(100):
        start = time.perf_counter()

        # Reuse pre-created structures
        _ = returns_df_cached.values @ weights_array_cached  # Calculate returns to measure performance

        elapsed = time.perf_counter() - start
        times_reuse.append(elapsed)

    avg_create = np.mean(times_create_new) * 1000000  # microseconds
    avg_reuse = np.mean(times_reuse) * 1000000
    improvement = ((avg_create - avg_reuse) / avg_create) * 100

    print(f"\nCreate new DataFrame: {avg_create:.1f}µs")
    print(f"Reuse DataFrame:      {avg_reuse:.1f}µs")
    print(f"Improvement:          {improvement:.1f}%")

    if improvement > 20:
        print(f"\nOK: Reusing DataFrames saves {improvement:.1f}% - SHOULD OPTIMIZE")
    else:
        print(f"\nWARNING: Improvement only {improvement:.1f}% - not worth complexity")

    return improvement


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZATION BENCHMARKS")
    print("="*80)

    # Run all benchmarks
    gc_overhead = benchmark_gc_collect_in_charts()
    read_imp, size_imp = benchmark_cache_serialization()
    df_improvement = benchmark_dataframe_reuse()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    print("\n1. gc.collect() in charts.py:")
    if gc_overhead > 10:
        print(f"   WARNING: Remove gc.collect() - adds {gc_overhead:.1f}% overhead")
    else:
        print(f"   OK: Keep gc.collect() - only {gc_overhead:.1f}% overhead, helps with memory")

    print("\n2. Cache serialization:")
    if read_imp > 20 or size_imp > 20:
        print(f"   OK: Switch to parquet - {read_imp:.1f}% faster reads, {size_imp:.1f}% smaller")
    else:
        print("   WARNING: Keep pickle - parquet not significantly better")

    print("\n3. DataFrame reuse in portfolio metrics:")
    if df_improvement > 20:
        print(f"   OK: Optimize to reuse DataFrames - {df_improvement:.1f}% faster")
    else:
        print(f"   WARNING: Current approach OK - only {df_improvement:.1f}% improvement possible")

    print("\n" + "="*80)
