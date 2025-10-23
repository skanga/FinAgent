"""
Tests for comparative analysis functionality.

Verifies that ComparativeAnalysis dataclass is properly populated with
benchmark comparison metrics.
"""

import pytest
import pandas as pd
import numpy as np
from analyzers import AdvancedFinancialAnalyzer
from models import ComparativeAnalysis


@pytest.fixture
def analyzer():
    """Create an analyzer with default settings."""
    return AdvancedFinancialAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def ticker_returns():
    """Create sample ticker returns."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    return returns


@pytest.fixture
def benchmark_returns():
    """Create sample benchmark returns."""
    np.random.seed(43)
    dates = pd.date_range("2024-01-01", periods=100)
    returns = pd.Series(np.random.normal(0.0008, 0.015, 100), index=dates)
    return returns


class TestCACreation:
    """Test that comparative analysis is created correctly."""

    def test_create_comparative_analysis_returns_object(
        self, analyzer, ticker_returns, benchmark_returns
    ):
        """Should return a ComparativeAnalysis object."""
        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert isinstance(result, ComparativeAnalysis)

    def test_create_comparative_analysis_without_benchmark(
        self, analyzer, ticker_returns
    ):
        """Should return None when benchmark is not available."""
        result = analyzer.create_comparative_analysis(ticker_returns, None)

        assert result is None

    def test_create_comparative_analysis_with_empty_benchmark(
        self, analyzer, ticker_returns
    ):
        """Should return None when benchmark is empty."""
        empty_benchmark = pd.Series([], dtype=float)
        result = analyzer.create_comparative_analysis(ticker_returns, empty_benchmark)

        assert result is None

    def test_create_comparative_analysis_with_insufficient_data(self, analyzer):
        """Should return None when insufficient data points."""
        short_ticker = pd.Series([0.01, 0.02])
        short_benchmark = pd.Series([0.01, 0.015])

        result = analyzer.create_comparative_analysis(short_ticker, short_benchmark)

        assert result is None


class TestCAMetrics:
    """Test that all comparative metrics are calculated."""

    def test_all_metrics_are_present(
        self, analyzer, ticker_returns, benchmark_returns
    ):
        """All ComparativeAnalysis fields should be populated."""
        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert result.outperformance is not None
        assert result.correlation is not None
        assert result.tracking_error is not None
        assert result.information_ratio is not None
        assert result.beta_vs_benchmark is not None
        assert result.alpha_vs_benchmark is not None
        assert result.relative_volatility is not None

    def test_beta_is_calculated(self, analyzer, ticker_returns, benchmark_returns):
        """Beta should be calculated and be a reasonable value."""
        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result.beta_vs_benchmark is not None
        # Beta should typically be between -2 and 3 for most stocks
        assert -2.0 <= result.beta_vs_benchmark <= 3.0

    def test_correlation_is_in_valid_range(
        self, analyzer, ticker_returns, benchmark_returns
    ):
        """Correlation should be between -1 and 1."""
        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result.correlation is not None
        assert -1.0 <= result.correlation <= 1.0

    def test_tracking_error_is_positive(
        self, analyzer, ticker_returns, benchmark_returns
    ):
        """Tracking error should be a positive value."""
        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result.tracking_error is not None
        assert result.tracking_error >= 0

    def test_relative_volatility_is_positive(
        self, analyzer, ticker_returns, benchmark_returns
    ):
        """Relative volatility should be a positive value."""
        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result.relative_volatility is not None
        assert result.relative_volatility > 0


class TestCAEdgeCases:
    """Test edge cases for comparative analysis."""

    def test_handles_identical_returns(self, analyzer):
        """Should handle case where ticker matches benchmark exactly."""
        dates = pd.date_range("2024-01-01", periods=50)
        identical_returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates)

        result = analyzer.create_comparative_analysis(
            identical_returns, identical_returns.copy()
        )

        assert result is not None
        # Beta should be close to 1.0 for identical returns
        assert result.beta_vs_benchmark is not None
        assert abs(result.beta_vs_benchmark - 1.0) < 0.1
        # Correlation should be 1.0 for identical returns
        assert result.correlation is not None
        assert abs(result.correlation - 1.0) < 0.01
        # Tracking error should be None or near zero (identical returns have zero tracking error)
        if result.tracking_error is not None:
            assert result.tracking_error < 0.01
        # Information ratio may be None when tracking error is zero
        # Outperformance should be near zero
        assert abs(result.outperformance) < 0.001

    def test_handles_negative_correlation(self, analyzer):
        """Should handle negatively correlated returns."""
        dates = pd.date_range("2024-01-01", periods=50)
        ticker_returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates)
        # Create negatively correlated benchmark
        benchmark_returns = -ticker_returns + pd.Series(
            np.random.normal(0, 0.005, 50), index=dates
        )

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert result.correlation is not None
        # Should be negatively correlated
        assert result.correlation < 0

    def test_handles_high_volatility_ticker(self, analyzer):
        """Should handle ticker with much higher volatility than benchmark."""
        dates = pd.date_range("2024-01-01", periods=50)
        ticker_returns = pd.Series(np.random.normal(0.001, 0.05, 50), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.001, 0.01, 50), index=dates)

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert result.relative_volatility is not None
        # Ticker volatility is 5x benchmark volatility
        assert result.relative_volatility > 1.0

    def test_handles_misaligned_dates(self, analyzer):
        """Should align returns to common dates."""
        ticker_dates = pd.date_range("2024-01-01", periods=60)
        benchmark_dates = pd.date_range("2024-01-10", periods=60)

        ticker_returns = pd.Series(
            np.random.normal(0.001, 0.02, 60), index=ticker_dates
        )
        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.015, 60), index=benchmark_dates
        )

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        # Should work despite date misalignment
        assert result is not None
        assert result.beta_vs_benchmark is not None

    def test_handles_nan_values_in_returns(self, analyzer):
        """Should handle NaN values in returns."""
        dates = pd.date_range("2024-01-01", periods=50)
        ticker_returns = pd.Series(np.random.normal(0.001, 0.02, 50), index=dates)
        ticker_returns.iloc[10:15] = np.nan  # Add some NaN values

        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.015, 50), index=dates)

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        # Should still work after dropping NaN
        assert result is not None

    def test_returns_none_with_too_few_aligned_points(self, analyzer):
        """Should return None when too few data points after alignment."""
        ticker_dates = pd.date_range("2024-01-01", periods=5)
        benchmark_dates = pd.date_range("2024-02-01", periods=5)

        ticker_returns = pd.Series(
            np.random.normal(0.001, 0.02, 5), index=ticker_dates
        )
        benchmark_returns = pd.Series(
            np.random.normal(0.0008, 0.015, 5), index=benchmark_dates
        )

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        # No overlapping dates
        assert result is None


class TestCAIntegration:
    """Test integration with TickerAnalysis."""

    def test_comparative_analysis_included_in_ticker_analysis(self):
        """Comparative analysis should be included in TickerAnalysis when benchmark available."""
        from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators

        # This is tested implicitly by test_orchestrator.py
        # Just verify the field exists in TickerAnalysis
        analysis = TickerAnalysis(
            ticker="TEST",
            csv_path="test.csv",
            chart_path="test.png",
            latest_close=100.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={},
            fundamentals=None,
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
            comparative_analysis=ComparativeAnalysis(
                outperformance=0.05,
                correlation=0.8,
                tracking_error=0.03,
                information_ratio=1.5,
                beta_vs_benchmark=1.2,
                alpha_vs_benchmark=0.02,
                relative_volatility=1.1,
            ),
        )

        assert analysis.comparative_analysis is not None
        assert isinstance(analysis.comparative_analysis, ComparativeAnalysis)
        assert analysis.comparative_analysis.beta_vs_benchmark == 1.2


class TestCAMetricsValues:
    """Test specific metric calculations."""

    def test_outperformance_positive_when_ticker_outperforms(self, analyzer):
        """Outperformance should be positive when ticker beats benchmark."""
        dates = pd.date_range("2024-01-01", periods=50)
        # Ticker with higher returns
        ticker_returns = pd.Series(np.full(50, 0.002), index=dates)
        # Benchmark with lower returns
        benchmark_returns = pd.Series(np.full(50, 0.001), index=dates)

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert result.outperformance > 0

    def test_outperformance_negative_when_ticker_underperforms(self, analyzer):
        """Outperformance should be negative when ticker underperforms benchmark."""
        dates = pd.date_range("2024-01-01", periods=50)
        # Ticker with lower returns
        ticker_returns = pd.Series(np.full(50, 0.001), index=dates)
        # Benchmark with higher returns
        benchmark_returns = pd.Series(np.full(50, 0.002), index=dates)

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert result.outperformance < 0

    def test_beta_near_one_for_similar_volatility(self, analyzer):
        """Beta should be near 1.0 when ticker moves with benchmark."""
        np.random.seed(44)
        dates = pd.date_range("2024-01-01", periods=100)
        benchmark_returns = pd.Series(
            np.random.normal(0.001, 0.02, 100), index=dates
        )
        # Ticker moves with benchmark plus small noise
        ticker_returns = (
            benchmark_returns * 0.95
            + pd.Series(np.random.normal(0, 0.003, 100), index=dates)
        )

        result = analyzer.create_comparative_analysis(ticker_returns, benchmark_returns)

        assert result is not None
        assert result.beta_vs_benchmark is not None
        # Beta should be close to 1.0
        assert 0.7 <= result.beta_vs_benchmark <= 1.3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
