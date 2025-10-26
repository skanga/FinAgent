"""
Test YoY growth calculation refactoring.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from analyzers import AdvancedFinancialAnalyzer


def create_mock_income_statement() -> pd.DataFrame:
    """Create a mock income statement DataFrame with quarterly data."""
    # Simulate 5 quarters of data (Q0 = most recent, Q4 = 1 year ago)
    quarters = [
        datetime(2024, 12, 31, tzinfo=timezone.utc),  # Q0 - Most recent
        datetime(2024, 9, 30, tzinfo=timezone.utc),  # Q1
        datetime(2024, 6, 30, tzinfo=timezone.utc),  # Q2
        datetime(2024, 3, 31, tzinfo=timezone.utc),  # Q3
        datetime(2023, 12, 31, tzinfo=timezone.utc),  # Q4 - Year ago
    ]

    data = {
        quarters[0]: {
            "Total Revenue": 120_000_000,  # Current: $120M
            "Net Income": 24_000_000,  # Current: $24M
            "Gross Profit": 60_000_000,
            "Operating Income": 30_000_000,
            "EBITDA": 35_000_000,
        },
        quarters[1]: {
            "Total Revenue": 115_000_000,
            "Net Income": 22_000_000,
            "Gross Profit": 57_500_000,
            "Operating Income": 28_750_000,
            "EBITDA": 33_250_000,
        },
        quarters[2]: {
            "Total Revenue": 110_000_000,
            "Net Income": 20_000_000,
            "Gross Profit": 55_000_000,
            "Operating Income": 27_500_000,
            "EBITDA": 31_500_000,
        },
        quarters[3]: {
            "Total Revenue": 105_000_000,
            "Net Income": 18_000_000,
            "Gross Profit": 52_500_000,
            "Operating Income": 26_250_000,
            "EBITDA": 29_750_000,
        },
        quarters[4]: {
            "Total Revenue": 100_000_000,  # Year ago: $100M
            "Net Income": 20_000_000,  # Year ago: $20M
            "Gross Profit": 50_000_000,
            "Operating Income": 25_000_000,
            "EBITDA": 28_000_000,
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
            "Total Revenue": 120_000_000,
            "Net Income": 24_000_000,
        },
        quarters[1]: {
            "Total Revenue": 115_000_000,
            "Net Income": 22_000_000,
        },
    }

    return pd.DataFrame(data)


def create_zero_base_income_statement() -> pd.DataFrame:
    """Create income statement with zero year-ago value (division by zero test)."""
    quarters = [datetime(2024, 12, 31, tzinfo=timezone.utc) for _ in range(5)]

    data = {
        quarters[0]: {"Total Revenue": 120_000_000, "Net Income": 24_000_000},
        quarters[1]: {"Total Revenue": 100_000_000, "Net Income": 20_000_000},
        quarters[2]: {"Total Revenue": 80_000_000, "Net Income": 16_000_000},
        quarters[3]: {"Total Revenue": 50_000_000, "Net Income": 10_000_000},
        quarters[4]: {"Total Revenue": 0, "Net Income": 0},  # Zero base
    }

    return pd.DataFrame(data)


def create_missing_data_income_statement() -> pd.DataFrame:
    """Create income statement with missing/NaN values."""
    quarters = [datetime(2024, 12, 31, tzinfo=timezone.utc) for _ in range(5)]

    data = {
        quarters[0]: {"Total Revenue": 120_000_000, "Net Income": 24_000_000},
        quarters[1]: {"Total Revenue": 115_000_000, "Net Income": 22_000_000},
        quarters[2]: {"Total Revenue": 110_000_000, "Net Income": 20_000_000},
        quarters[3]: {"Total Revenue": 105_000_000, "Net Income": 18_000_000},
        quarters[4]: {"Total Revenue": np.nan, "Net Income": np.nan},  # Missing data
    }

    return pd.DataFrame(data)


class TestYoYGrowthCalculation:
    """Test YoY growth calculation functionality."""

    def test_yoy_growth_normal_case(self):
        """Test YoY growth calculation with normal data."""
        analyzer = AdvancedFinancialAnalyzer()
        income_stmt = create_mock_income_statement()

        # Calculate YoY growth
        revenue_growth = analyzer._calculate_yoy_growth(income_stmt, "Total Revenue")
        earnings_growth = analyzer._calculate_yoy_growth(income_stmt, "Net Income")

        # Expected: (120M - 100M) / 100M = 0.20 (20% growth)
        expected_revenue = 0.20
        # Expected: (24M - 20M) / 20M = 0.20 (20% growth)
        expected_earnings = 0.20

        assert revenue_growth is not None, "Revenue growth should not be None"
        assert (
            abs(revenue_growth - expected_revenue) < 0.001
        ), f"Revenue growth incorrect: {revenue_growth} vs {expected_revenue}"

        assert earnings_growth is not None, "Earnings growth should not be None"
        assert (
            abs(earnings_growth - expected_earnings) < 0.001
        ), f"Earnings growth incorrect: {earnings_growth} vs {expected_earnings}"

    def test_yoy_growth_insufficient_data(self):
        """Test YoY growth calculation with insufficient columns."""
        analyzer = AdvancedFinancialAnalyzer()
        income_stmt = create_insufficient_data_income_statement()

        revenue_growth = analyzer._calculate_yoy_growth(income_stmt, "Total Revenue")
        earnings_growth = analyzer._calculate_yoy_growth(income_stmt, "Net Income")

        assert (
            revenue_growth is None
        ), "Revenue growth should be None for insufficient data"
        assert (
            earnings_growth is None
        ), "Earnings growth should be None for insufficient data"

    def test_yoy_growth_zero_base(self):
        """Test YoY growth calculation with zero base (division by zero)."""
        analyzer = AdvancedFinancialAnalyzer()
        income_stmt = create_zero_base_income_statement()

        revenue_growth = analyzer._calculate_yoy_growth(income_stmt, "Total Revenue")
        earnings_growth = analyzer._calculate_yoy_growth(income_stmt, "Net Income")

        assert revenue_growth is None, "Revenue growth should be None for zero base"
        assert earnings_growth is None, "Earnings growth should be None for zero base"

    def test_yoy_growth_missing_data(self):
        """Test YoY growth calculation with NaN values."""
        analyzer = AdvancedFinancialAnalyzer()
        income_stmt = create_missing_data_income_statement()

        revenue_growth = analyzer._calculate_yoy_growth(income_stmt, "Total Revenue")
        earnings_growth = analyzer._calculate_yoy_growth(income_stmt, "Net Income")

        assert revenue_growth is None, "Revenue growth should be None for NaN data"
        assert earnings_growth is None, "Earnings growth should be None for NaN data"

    def test_yoy_growth_custom_columns(self):
        """Test YoY growth with custom column indices."""
        analyzer = AdvancedFinancialAnalyzer()
        income_stmt = create_mock_income_statement()

        # Calculate QoQ growth (Q0 vs Q1) instead of YoY
        qoq_revenue = analyzer._calculate_yoy_growth(
            income_stmt, "Total Revenue", current_col=0, yoy_col=1
        )

        # Expected: (120M - 115M) / 115M ≈ 0.0435 (4.35% growth)
        expected_qoq = (120_000_000 - 115_000_000) / 115_000_000

        assert qoq_revenue is not None, "QoQ revenue growth should not be None"
        assert (
            abs(qoq_revenue - expected_qoq) < 0.001
        ), f"QoQ growth incorrect: {qoq_revenue} vs {expected_qoq}"

    def test_code_deduplication(self):
        """Test that the refactored code reduces duplication."""
        # Read the analyzers.py file
        with open(Path("analyzers.py"), "r", encoding="utf-8") as f:
            content = f.read()

        # Check that the helper method exists
        has_helper = "_calculate_yoy_growth" in content

        # Count occurrences of the old duplicate pattern
        duplicate_pattern = (
            "current_revenue = self._safe_get_value(income_stmt.iloc[:, 0]"
        )
        duplicate_count = content.count(duplicate_pattern)

        assert has_helper, "Helper method _calculate_yoy_growth should exist"
        assert (
            duplicate_count == 0
        ), f"Found {duplicate_count} occurrences of duplicate pattern"

    def test_backward_compatibility(self):
        """Test that refactored code produces same results as original."""
        analyzer = AdvancedFinancialAnalyzer()
        income_stmt = create_mock_income_statement()

        # New approach using helper
        new_revenue_growth = analyzer._calculate_yoy_growth(
            income_stmt, "Total Revenue"
        )
        new_earnings_growth = analyzer._calculate_yoy_growth(income_stmt, "Net Income")

        # Old approach (manual calculation)
        current_revenue = analyzer._safe_get_value(
            income_stmt.iloc[:, 0], "Total Revenue"
        )
        yoy_revenue = analyzer._safe_get_value(income_stmt.iloc[:, 4], "Total Revenue")
        old_revenue_growth = (
            (current_revenue - yoy_revenue) / yoy_revenue
            if current_revenue and yoy_revenue and yoy_revenue != 0
            else None
        )

        current_income = analyzer._safe_get_value(income_stmt.iloc[:, 0], "Net Income")
        yoy_income = analyzer._safe_get_value(income_stmt.iloc[:, 4], "Net Income")
        old_earnings_growth = (
            (current_income - yoy_income) / yoy_income
            if current_income and yoy_income and yoy_income != 0
            else None
        )

        assert (
            abs(new_revenue_growth - old_revenue_growth) < 0.0001
        ), f"Revenue growth mismatch: new={new_revenue_growth:.4f} vs old={old_revenue_growth:.4f}"
        assert (
            abs(new_earnings_growth - old_earnings_growth) < 0.0001
        ), f"Earnings growth mismatch: new={new_earnings_growth:.4f} vs old={old_earnings_growth:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
