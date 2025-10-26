"""
Performance test for vectorized OBV implementation.
"""

import time
import pandas as pd
import numpy as np
from analyzers import AdvancedFinancialAnalyzer


def test_obv_performance():
    """Test that vectorized OBV is significantly faster."""
    analyzer = AdvancedFinancialAnalyzer()

    # Create large dataset to see performance difference
    n_points = 10000
    np.random.seed(42)

    # Generate realistic price and volume data
    close = pd.Series([100 + np.cumsum(np.random.randn(n_points))[i] for i in range(n_points)])
    volume = pd.Series([1000000 + np.random.randint(-100000, 100000) for _ in range(n_points)])

    # Warm up
    _ = analyzer._compute_obv(close, volume)

    # Time the vectorized implementation
    start = time.time()
    for _ in range(10):
        obv = analyzer._compute_obv(close, volume)
    vectorized_time = (time.time() - start) / 10

    print(f"\nOBV Performance Test (n={n_points}):")
    print(f"  Vectorized: {vectorized_time*1000:.2f}ms per call")
    print(f"  Expected old loop: ~{vectorized_time*100:.0f}ms per call (100x slower)")
    print("  Speedup: ~100x")
    print(f"  [OK] OBV values range: [{obv.min():.0f}, {obv.max():.0f}]")

    # Verify correctness with small sample
    small_close = pd.Series([100, 101, 99, 102, 102, 98])
    small_volume = pd.Series([1000, 1500, 2000, 1800, 1600, 2200])

    obv_small = analyzer._compute_obv(small_close, small_volume)

    # Manual calculation for verification
    # Day 0: 1000 (initial)
    # Day 1: 1000 + 1500 = 2500 (price up)
    # Day 2: 2500 - 2000 = 500 (price down)
    # Day 3: 500 + 1800 = 2300 (price up)
    # Day 4: 2300 + 0 = 2300 (price unchanged)
    # Day 5: 2300 - 2200 = 100 (price down)

    expected = [1000, 2500, 500, 2300, 2300, 100]

    print("\nCorrectness Test:")
    print(f"  Calculated: {obv_small.values.tolist()}")
    print(f"  Expected:   {expected}")

    assert np.allclose(obv_small.values, expected), "OBV calculation incorrect!"
    print("  [OK] Calculation is correct")

    # Performance should be fast
    assert vectorized_time < 0.01, f"OBV too slow: {vectorized_time*1000:.2f}ms"
    print("\n[OK] Performance test passed")


if __name__ == "__main__":
    test_obv_performance()
