"""
Unit tests for analyzers.py module.

Tests technical indicator calculations, financial ratio computations,
and fundamental data parsing with mocked external dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import timezone

from analyzers import AdvancedFinancialAnalyzer, PortfolioAnalyzer
from models import FundamentalData, TickerAnalysis


@pytest.fixture
def analyzer():
    """Create an AdvancedFinancialAnalyzer instance."""
    return AdvancedFinancialAnalyzer(risk_free_rate=0.02, benchmark_ticker="SPY")


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz=timezone.utc)
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    return pd.DataFrame(
        {
            "Date": dates,
            "Close": prices,
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Volume": np.random.randint(1000000, 10000000, 100),
        }
    )


class TestComputeMetrics:
    """Test the compute_metrics method."""

    def test_compute_metrics_adds_all_indicators(self, analyzer, sample_price_data):
        """Test that all technical indicators are computed and added to DataFrame."""
        result = analyzer.compute_metrics(sample_price_data)

        # Check that all expected columns are present
        expected_columns = [
            "Date",
            "close",
            "daily_return",
            "30d_ma",
            "50d_ma",
            "volatility",
            "rsi",
            "bollinger_upper",
            "bollinger_lower",
            "bollinger_position",
            "macd",
            "macd_signal",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_metrics_preserves_row_count(self, analyzer, sample_price_data):
        """Test that compute_metrics doesn't drop rows."""
        result = analyzer.compute_metrics(sample_price_data)
        assert len(result) == len(sample_price_data)

    def test_compute_metrics_handles_small_dataset(self, analyzer):
        """Test that compute_metrics works with minimal data."""
        small_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=10, tz=timezone.utc),
                "Close": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            }
        )

        result = analyzer.compute_metrics(small_data)

        # Should still compute, but some indicators may have NaN values
        assert len(result) == 10
        assert "rsi" in result.columns
        assert "macd" in result.columns

    def test_compute_metrics_handles_flat_prices(self, analyzer):
        """Test behavior with constant prices (no volatility)."""
        flat_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=50, tz=timezone.utc),
                "Close": [100.0] * 50,
            }
        )

        result = analyzer.compute_metrics(flat_data)

        # Volatility should be 0 or very close to 0
        assert result["volatility"].iloc[-1] <= 0.001 or pd.isna(
            result["volatility"].iloc[-1]
        )
        # Daily returns should all be 0
        assert result["daily_return"].iloc[1:].abs().max() < 0.001


class TestRSIComputation:
    """Test RSI (Relative Strength Index) calculation."""

    def test_compute_rsi_range(self, analyzer, sample_price_data):
        """Test that RSI values are in valid range [0, 100]."""
        result = analyzer.compute_metrics(sample_price_data)

        rsi_values = result["rsi"].dropna()
        assert (rsi_values >= 0).all(), "RSI values below 0"
        assert (rsi_values <= 100).all(), "RSI values above 100"

    def test_compute_rsi_trending_up(self, analyzer):
        """Test RSI with consistently rising prices (should trend toward overbought)."""
        rising_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=50, tz=timezone.utc),
                "Close": [100 + i for i in range(50)],  # Consistent upward trend
            }
        )

        result = analyzer.compute_metrics(rising_data)

        # RSI should be relatively high for strong uptrend
        final_rsi = result["rsi"].iloc[-1]
        assert final_rsi > 50, f"Expected RSI > 50 for uptrend, got {final_rsi}"

    def test_compute_rsi_trending_down(self, analyzer):
        """Test RSI with consistently falling prices (should trend toward oversold)."""
        falling_data = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=50, tz=timezone.utc),
                "Close": [100 - i for i in range(50)],  # Consistent downward trend
            }
        )

        result = analyzer.compute_metrics(falling_data)

        # RSI should be relatively low for strong downtrend
        final_rsi = result["rsi"].iloc[-1]
        assert final_rsi < 50, f"Expected RSI < 50 for downtrend, got {final_rsi}"


class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_bollinger_bands_relationship(self, analyzer, sample_price_data):
        """Test that upper band > price > lower band (generally)."""
        result = analyzer.compute_metrics(sample_price_data)

        # Drop NaN values from early periods
        valid_data = result.dropna(subset=["bollinger_upper", "bollinger_lower"])

        # Upper band should be above lower band
        assert (valid_data["bollinger_upper"] > valid_data["bollinger_lower"]).all()

    def test_bollinger_position_range(self, analyzer, sample_price_data):
        """Test that Bollinger position is typically in [0, 1] range."""
        result = analyzer.compute_metrics(sample_price_data)

        positions = result["bollinger_position"].dropna()

        # Most positions should be in [0, 1] range (price within bands)
        # Some outliers are acceptable
        in_range = ((positions >= 0) & (positions <= 1)).sum()
        assert (
            in_range / len(positions) > 0.7
        ), "Too many prices outside Bollinger Bands"


class TestMACDComputation:
    """Test MACD indicator calculation."""

    def test_macd_components_exist(self, analyzer, sample_price_data):
        """Test that both MACD and signal line are computed."""
        result = analyzer.compute_metrics(sample_price_data)

        assert "macd" in result.columns
        assert "macd_signal" in result.columns

        # Should have non-NaN values after sufficient data
        assert result["macd"].notna().sum() > 0
        assert result["macd_signal"].notna().sum() > 0


class TestAdvancedMetrics:
    """Test advanced performance metrics calculation."""

    def test_calculate_advanced_metrics_with_positive_returns(self, analyzer):
        """Test metrics calculation with profitable returns."""
        # Simulate positive daily returns
        returns = pd.Series([0.01, 0.005, 0.015, -0.002, 0.008] * 20)  # 100 days
        benchmark_returns = pd.Series([0.008] * 100)

        metrics = analyzer.calculate_advanced_metrics(returns, benchmark_returns)

        assert metrics.sharpe_ratio is not None
        assert metrics.sortino_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.var_95 is not None

    def test_calculate_advanced_metrics_sharpe_ratio(self, analyzer):
        """Test Sharpe ratio calculation."""
        # Consistent positive returns with low volatility should give high Sharpe
        returns = pd.Series([0.01] * 100)  # 1% daily return consistently

        metrics = analyzer.calculate_advanced_metrics(returns, None)

        # Should have positive Sharpe ratio
        assert metrics.sharpe_ratio > 0

    def test_calculate_advanced_metrics_max_drawdown(self, analyzer):
        """Test maximum drawdown calculation."""
        # Create returns that result in a known drawdown - use more data points
        # Start at 100, rise to 110, fall to 95 (13.6% drawdown from peak)
        prices = [
            100,
            102,
            105,
            108,
            110,
            108,
            105,
            100,
            97,
            95,
        ] * 3  # Repeat to have enough data
        returns = pd.Series(
            [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]
        )

        metrics = analyzer.calculate_advanced_metrics(returns, None)

        # Should detect the drawdown
        if metrics.max_drawdown is not None:
            assert metrics.max_drawdown < 0  # Drawdowns are negative
            assert metrics.max_drawdown <= -0.10  # At least 10% drawdown

    def test_calculate_advanced_metrics_with_all_negative_returns(self, analyzer):
        """Test metrics with consistently losing returns."""
        returns = pd.Series([-0.01] * 100)  # Consistent losses

        metrics = analyzer.calculate_advanced_metrics(returns, None)

        # Sharpe ratio should be negative
        assert metrics.sharpe_ratio < 0
        # Max drawdown should be substantial
        assert metrics.max_drawdown < -0.50


class TestComputeRatios:
    """Test financial ratios computation with mocked yfinance."""

    @patch("analyzers.yf.Ticker")
    def test_compute_ratios_with_valid_data(self, mock_ticker, analyzer):
        """Test ratio computation with complete yfinance data."""
        # Mock the ticker info
        mock_ticker_obj = Mock()
        mock_ticker_obj.info = {
            "trailingPE": 25.5,
            "forwardPE": 22.0,
            "pegRatio": 1.8,
            "priceToBook": 3.5,
            "debtToEquity": 45.2,
            "currentRatio": 2.1,
            "returnOnEquity": 0.18,
            "profitMargins": 0.15,
            "beta": 1.2,
        }
        mock_ticker.return_value = mock_ticker_obj

        ratios = analyzer.compute_ratios("AAPL")

        assert ratios["pe_ratio"] == 25.5
        assert ratios["forward_pe"] == 22.0
        assert ratios["peg_ratio"] == 1.8
        assert ratios["price_to_book"] == 3.5
        assert ratios["beta"] == 1.2

    @patch("analyzers.yf.Ticker")
    def test_compute_ratios_with_missing_data(self, mock_ticker, analyzer):
        """Test ratio computation when some data is missing."""
        mock_ticker_obj = Mock()
        mock_ticker_obj.info = {
            "trailingPE": 25.5,
            # Other ratios missing
        }
        mock_ticker.return_value = mock_ticker_obj

        ratios = analyzer.compute_ratios("AAPL")

        # Should have PE ratio
        assert ratios["pe_ratio"] == 25.5
        # Others should be None
        assert ratios["forward_pe"] is None
        assert ratios["peg_ratio"] is None

    @patch("analyzers.yf.Ticker")
    def test_compute_ratios_with_empty_info(self, mock_ticker, analyzer):
        """Test ratio computation when yfinance returns empty info."""
        mock_ticker_obj = Mock()
        mock_ticker_obj.info = {}
        mock_ticker.return_value = mock_ticker_obj

        ratios = analyzer.compute_ratios("INVALID")

        # All ratios should be None
        assert all(v is None for v in ratios.values())

    @patch("analyzers.yf.Ticker")
    def test_compute_ratios_handles_exception(self, mock_ticker, analyzer):
        """Test that compute_ratios handles exceptions gracefully."""
        mock_ticker.side_effect = Exception("Network error")

        ratios = analyzer.compute_ratios("AAPL")

        # Should return dict with None values, not raise exception
        assert all(v is None for v in ratios.values())


class TestParseFundamentals:
    """Test fundamental data parsing with mocked yfinance."""

    @patch("analyzers.yf.Ticker")
    def test_parse_fundamentals_with_complete_data(self, mock_ticker, analyzer):
        """Test parsing complete fundamental data."""
        mock_ticker_obj = Mock()
        # Data should be in columns (each column is a quarter)
        mock_income = pd.DataFrame(
            {
                "Q1": [100e9, 20e9],
                "Q2": [95e9, 18e9],
                "Q3": [90e9, 17e9],
                "Q4": [85e9, 16e9],
                "Q5": [80e9, 15e9],
            },
            index=["Total Revenue", "Net Income"],
        )

        mock_cashflow = pd.DataFrame(
            {"Q1": [25e9], "Q2": [23e9], "Q3": [22e9], "Q4": [20e9], "Q5": [19e9]},
            index=["Free Cash Flow"],
        )

        mock_ticker_obj.quarterly_income_stmt = mock_income
        mock_ticker_obj.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker_obj.quarterly_cashflow = mock_cashflow
        mock_ticker.return_value = mock_ticker_obj

        fundamentals = analyzer.parse_fundamentals("AAPL")

        assert fundamentals.revenue == 100e9
        assert fundamentals.net_income == 20e9
        assert fundamentals.free_cash_flow == 25e9
        # YoY growth should be calculated
        assert fundamentals.revenue_growth is not None
        assert fundamentals.earnings_growth is not None

    @patch("analyzers.yf.Ticker")
    def test_parse_fundamentals_calculates_yoy_growth(self, mock_ticker, analyzer):
        """Test year-over-year growth calculation."""
        mock_ticker_obj = Mock()
        # Revenue growing from 80B to 100B (25% growth)
        mock_income = pd.DataFrame(
            {
                "Q1": [100e9, 20e9],
                "Q2": [95e9, 18e9],
                "Q3": [90e9, 17e9],
                "Q4": [85e9, 16e9],
                "Q5": [80e9, 15e9],
            },
            index=["Total Revenue", "Net Income"],
        )

        mock_cashflow = pd.DataFrame(
            {"Q1": [25e9], "Q2": [25e9], "Q3": [25e9], "Q4": [25e9], "Q5": [25e9]},
            index=["Free Cash Flow"],
        )

        mock_ticker_obj.quarterly_income_stmt = mock_income
        mock_ticker_obj.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker_obj.quarterly_cashflow = mock_cashflow
        mock_ticker.return_value = mock_ticker_obj

        fundamentals = analyzer.parse_fundamentals("AAPL")

        # Revenue growth should be positive (100B vs 80B = 25% growth)
        assert fundamentals.revenue_growth > 0.20
        assert fundamentals.revenue_growth < 0.30

    @patch("analyzers.yf.Ticker")
    def test_parse_fundamentals_with_insufficient_data(self, mock_ticker, analyzer):
        """Test parsing when insufficient historical data available."""
        mock_ticker_obj = Mock()
        # Only 3 quarters of data (need 5 for YoY)
        mock_income = pd.DataFrame(
            {"Q1": [100e9, 20e9], "Q2": [95e9, 18e9], "Q3": [90e9, 17e9]},
            index=["Total Revenue", "Net Income"],
        )

        mock_cashflow = pd.DataFrame(
            {"Q1": [25e9], "Q2": [23e9], "Q3": [22e9]}, index=["Free Cash Flow"]
        )

        mock_ticker_obj.quarterly_income_stmt = mock_income
        mock_ticker_obj.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker_obj.quarterly_cashflow = mock_cashflow
        mock_ticker.return_value = mock_ticker_obj

        fundamentals = analyzer.parse_fundamentals("AAPL")

        # Current values should be present
        assert fundamentals.revenue == 100e9
        # But growth should be None (insufficient data)
        assert fundamentals.revenue_growth is None
        assert fundamentals.earnings_growth is None

    @patch("analyzers.yf.Ticker")
    def test_parse_fundamentals_handles_exception(self, mock_ticker, analyzer):
        """Test that parse_fundamentals handles exceptions gracefully."""
        mock_ticker.side_effect = Exception("API error")

        fundamentals = analyzer.parse_fundamentals("AAPL")

        # Should return FundamentalData with None values
        assert isinstance(fundamentals, FundamentalData)
        assert fundamentals.revenue is None
        assert fundamentals.net_income is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_metrics_with_empty_dataframe(self, analyzer):
        """Test behavior with empty DataFrame."""
        empty_data = pd.DataFrame(columns=["Date", "Close"])

        # Should handle gracefully, not crash
        result = analyzer.compute_metrics(empty_data)
        assert len(result) == 0

    def test_compute_metrics_with_nan_values(self, analyzer):
        """Test handling of NaN values in input data."""
        data_with_nans = pd.DataFrame(
            {
                "Date": pd.date_range(start="2024-01-01", periods=50, tz=timezone.utc),
                "Close": [100 + i if i % 5 != 0 else np.nan for i in range(50)],
            }
        )

        # Should handle NaN values without crashing
        result = analyzer.compute_metrics(data_with_nans)
        assert len(result) == 50

    def test_calculate_advanced_metrics_with_empty_returns(self, analyzer):
        """Test advanced metrics with no returns data."""
        empty_returns = pd.Series([], dtype=float)

        metrics = analyzer.calculate_advanced_metrics(empty_returns, None)

        # Should return metrics object with None values
        assert metrics.sharpe_ratio is None
        assert metrics.max_drawdown is None


@pytest.fixture
def portfolio_analyzer():
    """Create a PortfolioAnalyzer instance."""
    return PortfolioAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def mock_analyses():
    """Create mock ticker analyses for portfolio testing."""
    # Create just enough data for the portfolio analyzer to work
    return {
        "AAPL": TickerAnalysis(
            ticker="AAPL",
            csv_path="AAPL.csv",
            chart_path="AAPL.png",
            latest_close=150.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={},
            fundamentals=None,
            advanced_metrics=None,
            technical_indicators=None,
            sample_data=[],
            error=None,
        ),
        "MSFT": TickerAnalysis(
            ticker="MSFT",
            csv_path="MSFT.csv",
            chart_path="MSFT.png",
            latest_close=300.0,
            avg_daily_return=0.0008,
            volatility=0.018,
            ratios={},
            fundamentals=None,
            advanced_metrics=None,
            technical_indicators=None,
            sample_data=[],
            error=None,
        ),
    }


class TestPortfolioAnalyzer:
    """Test portfolio analysis edge cases."""

    def test_calculate_portfolio_metrics_with_negative_weights(self, portfolio_analyzer, mock_analyses):
        """Test that negative weights raise a ValueError."""
        weights = {"AAPL": 1.2, "MSFT": -0.2}  # Sums to 1.0 but contains negative weight

        with pytest.raises(ValueError, match="Portfolio weights cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_calculate_portfolio_metrics_with_valid_weights(self, portfolio_analyzer, mock_analyses):
        """Test that valid weights (including zero) work correctly."""
        weights = {"AAPL": 1.0, "MSFT": 0.0}

        # This should not raise an exception
        try:
            # We need to mock the pd.read_csv call inside the method
            with patch("pandas.read_csv") as mock_read_csv:
                # Return a dummy dataframe with the required column
                mock_read_csv.return_value = pd.DataFrame({"daily_return": [0.1, 0.2]})
                portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        except ValueError:
            pytest.fail("Valid weights (including zero) raised an unexpected ValueError.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])