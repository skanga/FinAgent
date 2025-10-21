"""
Test YoY growth calculation refactoring.
"""
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from analyzers import AdvancedFinancialAnalyzer
from models import FundamentalData


def create_mock_income_statement() -> pd.DataFrame:
    """Create a mock income statement DataFrame with quarterly data."""
    # Simulate 5 quarters of data (Q0 = most recent, Q4 = 1 year ago)
    quarters = [
        datetime(2024, 12, 31, tzinfo=timezone.utc),  # Q0 - Most recent
        datetime(2024, 9, 30, tzinfo=timezone.utc),   # Q1
        datetime(2024, 6, 30, tzinfo=timezone.utc),   # Q2
        datetime(2024, 3, 31, tzinfo=timezone.utc),   # Q3
        datetime(2023, 12, 31, tzinfo=timezone.utc),  # Q4 - Year ago
    ]

    data = {
        quarters[0]: {
            'Total Revenue': 120_000_000,      # Current: $120M
            'Net Income': 24_000_000,           # Current: $24M
            'Gross Profit': 60_000_000,
            'Operating Income': 30_000_000,
            'EBITDA': 35_000_000,
        },
        quarters[1]: {
            'Total Revenue': 115_000_000,
            'Net Income': 22_000_000,
            'Gross Profit': 57_500_000,
            'Operating Income': 28_750_000,
            'EBITDA': 33_250_000,
        },
        quarters[2]: {
            'Total Revenue': 110_000_000,
            'Net Income': 20_000_000,
            'Gross Profit': 55_000_000,
            'Operating Income': 27_500_000,
            'EBITDA': 31_500_000,
        },
        quarters[3]: {
            'Total Revenue': 105_000_000,
            'Net Income': 18_000_000,
            'Gross Profit': 52_500_000,
            'Operating Income': 26_250_000,
            'EBITDA': 29_750_000,
        },
        quarters[4]: {
            'Total Revenue': 100_000_000,      # Year ago: $100M
            'Net Income': 20_000_000,           # Year ago: $20M
            'Gross Profit': 50_000_000,
            'Operating Income': 25_000_000,
            'EBITDA': 28_000_000,
        },
    }

    return pd.DataFrame(data)


def create_insufficient_data_income_statement() -> pd.DataFrame:
    """Create income statement with insufficient columns for YoY calculation."""
    quarters = [
        datetime(2024, 12, 31, tzinfo=timezone.utc),
        datetime(2024, 9, 30, tzinfo=timezone.utc),
    ]

    data = {
        quarters[0]: {
            'Total Revenue': 120_000_000,
            'Net Income': 24_000_000,
        },
        quarters[1]: {
            'Total Revenue': 115_000_000,
            'Net Income': 22_000_000,
        },
    }

    return pd.DataFrame(data)


def create_zero_base_income_statement() -> pd.DataFrame:
    """Create income statement with zero year-ago value (division by zero test)."""
    quarters = [datetime(2024, 12, 31, tzinfo=timezone.utc) for _ in range(5)]

    data = {
        quarters[0]: {'Total Revenue': 120_000_000, 'Net Income': 24_000_000},
        quarters[1]: {'Total Revenue': 100_000_000, 'Net Income': 20_000_000},
        quarters[2]: {'Total Revenue': 80_000_000, 'Net Income': 16_000_000},
        quarters[3]: {'Total Revenue': 50_000_000, 'Net Income': 10_000_000},
        quarters[4]: {'Total Revenue': 0, 'Net Income': 0},  # Zero base
    }

    return pd.DataFrame(data)


def create_missing_data_income_statement() -> pd.DataFrame:
    """Create income statement with missing/NaN values."""
    quarters = [datetime(2024, 12, 31, tzinfo=timezone.utc) for _ in range(5)]

    data = {
        quarters[0]: {'Total Revenue': 120_000_000, 'Net Income': 24_000_000},
        quarters[1]: {'Total Revenue': 115_000_000, 'Net Income': 22_000_000},
        quarters[2]: {'Total Revenue': 110_000_000, 'Net Income': 20_000_000},
        quarters[3]: {'Total Revenue': 105_000_000, 'Net Income': 18_000_000},
        quarters[4]: {'Total Revenue': np.nan, 'Net Income': np.nan},  # Missing data
    }

    return pd.DataFrame(data)


def test_yoy_growth_normal_case():
    """Test YoY growth calculation with normal data."""
    print("Testing YoY growth calculation - normal case...")

    analyzer = AdvancedFinancialAnalyzer()
    income_stmt = create_mock_income_statement()

    # Calculate YoY growth
    revenue_growth = analyzer._calculate_yoy_growth(income_stmt, 'Total Revenue')
    earnings_growth = analyzer._calculate_yoy_growth(income_stmt, 'Net Income')

    # Expected: (120M - 100M) / 100M = 0.20 (20% growth)
    expected_revenue = 0.20
    # Expected: (24M - 20M) / 20M = 0.20 (20% growth)
    expected_earnings = 0.20

    print(f"  Revenue growth: {revenue_growth:.2%} (expected: {expected_revenue:.2%})")
    print(f"  Earnings growth: {earnings_growth:.2%} (expected: {expected_earnings:.2%})")

    if revenue_growth is not None and abs(revenue_growth - expected_revenue) < 0.001:
        print("  [PASS] Revenue growth calculated correctly")
        result1 = True
    else:
        print(f"  [FAIL] Revenue growth incorrect: {revenue_growth} vs {expected_revenue}")
        result1 = False

    if earnings_growth is not None and abs(earnings_growth - expected_earnings) < 0.001:
        print("  [PASS] Earnings growth calculated correctly\n")
        result2 = True
    else:
        print(f"  [FAIL] Earnings growth incorrect: {earnings_growth} vs {expected_earnings}\n")
        result2 = False

    return result1 and result2


def test_yoy_growth_insufficient_data():
    """Test YoY growth calculation with insufficient columns."""
    print("Testing YoY growth - insufficient data...")

    analyzer = AdvancedFinancialAnalyzer()
    income_stmt = create_insufficient_data_income_statement()

    revenue_growth = analyzer._calculate_yoy_growth(income_stmt, 'Total Revenue')
    earnings_growth = analyzer._calculate_yoy_growth(income_stmt, 'Net Income')

    print(f"  Revenue growth: {revenue_growth}")
    print(f"  Earnings growth: {earnings_growth}")

    if revenue_growth is None and earnings_growth is None:
        print("  [PASS] Correctly returns None for insufficient data\n")
        return True
    else:
        print("  [FAIL] Should return None when insufficient columns\n")
        return False


def test_yoy_growth_zero_base():
    """Test YoY growth calculation with zero base (division by zero)."""
    print("Testing YoY growth - zero base value...")

    analyzer = AdvancedFinancialAnalyzer()
    income_stmt = create_zero_base_income_statement()

    revenue_growth = analyzer._calculate_yoy_growth(income_stmt, 'Total Revenue')
    earnings_growth = analyzer._calculate_yoy_growth(income_stmt, 'Net Income')

    print(f"  Revenue growth: {revenue_growth}")
    print(f"  Earnings growth: {earnings_growth}")

    if revenue_growth is None and earnings_growth is None:
        print("  [PASS] Correctly handles zero base (no division by zero)\n")
        return True
    else:
        print("  [FAIL] Should return None to avoid division by zero\n")
        return False


def test_yoy_growth_missing_data():
    """Test YoY growth calculation with NaN values."""
    print("Testing YoY growth - missing/NaN data...")

    analyzer = AdvancedFinancialAnalyzer()
    income_stmt = create_missing_data_income_statement()

    revenue_growth = analyzer._calculate_yoy_growth(income_stmt, 'Total Revenue')
    earnings_growth = analyzer._calculate_yoy_growth(income_stmt, 'Net Income')

    print(f"  Revenue growth: {revenue_growth}")
    print(f"  Earnings growth: {earnings_growth}")

    if revenue_growth is None and earnings_growth is None:
        print("  [PASS] Correctly handles NaN values\n")
        return True
    else:
        print("  [FAIL] Should return None for NaN data\n")
        return False


def test_yoy_growth_custom_columns():
    """Test YoY growth with custom column indices."""
    print("Testing YoY growth - custom column indices...")

    analyzer = AdvancedFinancialAnalyzer()
    income_stmt = create_mock_income_statement()

    # Calculate QoQ growth (Q0 vs Q1) instead of YoY
    qoq_revenue = analyzer._calculate_yoy_growth(income_stmt, 'Total Revenue',
                                                  current_col=0, yoy_col=1)

    # Expected: (120M - 115M) / 115M ≈ 0.0435 (4.35% growth)
    expected_qoq = (120_000_000 - 115_000_000) / 115_000_000

    print(f"  QoQ revenue growth: {qoq_revenue:.2%} (expected: {expected_qoq:.2%})")

    if qoq_revenue is not None and abs(qoq_revenue - expected_qoq) < 0.001:
        print("  [PASS] Custom column indices work correctly\n")
        return True
    else:
        print(f"  [FAIL] QoQ growth incorrect: {qoq_revenue} vs {expected_qoq}\n")
        return False


def test_code_deduplication():
    """Test that the refactored code reduces duplication."""
    print("Testing code deduplication...")

    # Read the analyzers.py file
    with open('analyzers.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Check that the helper method exists
    has_helper = '_calculate_yoy_growth' in content

    # Count occurrences of the old duplicate pattern
    duplicate_pattern = 'current_revenue = self._safe_get_value(income_stmt.iloc[:, 0]'
    duplicate_count = content.count(duplicate_pattern)

    print(f"  Helper method exists: {has_helper}")
    print(f"  Duplicate pattern occurrences: {duplicate_count}")

    if has_helper and duplicate_count == 0:
        print("  [PASS] Code successfully deduplicated\n")
        return True
    else:
        print("  [FAIL] Duplication still exists or helper not found\n")
        return False


def test_backward_compatibility():
    """Test that refactored code produces same results as original."""
    print("Testing backward compatibility...")

    analyzer = AdvancedFinancialAnalyzer()
    income_stmt = create_mock_income_statement()

    # New approach using helper
    new_revenue_growth = analyzer._calculate_yoy_growth(income_stmt, 'Total Revenue')
    new_earnings_growth = analyzer._calculate_yoy_growth(income_stmt, 'Net Income')

    # Old approach (manual calculation)
    current_revenue = analyzer._safe_get_value(income_stmt.iloc[:, 0], 'Total Revenue')
    yoy_revenue = analyzer._safe_get_value(income_stmt.iloc[:, 4], 'Total Revenue')
    old_revenue_growth = (current_revenue - yoy_revenue) / yoy_revenue if current_revenue and yoy_revenue and yoy_revenue != 0 else None

    current_income = analyzer._safe_get_value(income_stmt.iloc[:, 0], 'Net Income')
    yoy_income = analyzer._safe_get_value(income_stmt.iloc[:, 4], 'Net Income')
    old_earnings_growth = (current_income - yoy_income) / yoy_income if current_income and yoy_income and yoy_income != 0 else None

    print(f"  New revenue growth: {new_revenue_growth:.4f}")
    print(f"  Old revenue growth: {old_revenue_growth:.4f}")
    print(f"  New earnings growth: {new_earnings_growth:.4f}")
    print(f"  Old earnings growth: {old_earnings_growth:.4f}")

    revenue_match = abs(new_revenue_growth - old_revenue_growth) < 0.0001
    earnings_match = abs(new_earnings_growth - old_earnings_growth) < 0.0001

    if revenue_match and earnings_match:
        print("  [PASS] Results match original implementation\n")
        return True
    else:
        print("  [FAIL] Results differ from original\n")
        return False


def main():
    """Run all YoY growth refactoring tests."""
    print("=" * 60)
    print("YoY GROWTH CALCULATION REFACTORING TESTS")
    print("=" * 60)
    print()

    tests = [
        test_yoy_growth_normal_case,
        test_yoy_growth_insufficient_data,
        test_yoy_growth_zero_base,
        test_yoy_growth_missing_data,
        test_yoy_growth_custom_columns,
        test_code_deduplication,
        test_backward_compatibility
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  [ERROR] {test.__name__} failed:")
            print(f"    {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All YoY refactoring tests passed!")
        return 0
    else:
        print(f"\n[PARTIAL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
