"""
Tests for error recovery paths and edge cases.

Covers:
- Error recovery in orchestrator._analyze_all_tickers
- Invalid portfolio weight combinations
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from orchestrator import FinancialReportOrchestrator
from analyzers import PortfolioAnalyzer
from config import Config
from models import TickerAnalysis, AdvancedMetrics, TechnicalIndicators


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = Mock(spec=Config)
    config.openai_api_key = "test-api-key"
    config.openai_base_url = "https://api.openai.com/v1"
    config.model_name = "gpt-4"
    config.benchmark_ticker = "SPY"
    config.risk_free_rate = 0.02
    config.cache_ttl_hours = 24
    config.max_workers = 3
    config.request_timeout = 30
    config.provider = "openai"
    return config


@pytest.fixture
def portfolio_analyzer():
    """Create a PortfolioAnalyzer instance."""
    return PortfolioAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def mock_analyses():
    """Create mock ticker analyses for portfolio tests."""
    # Create temporary CSV files with return data
    temp_dir = tempfile.mkdtemp()

    analyses = {}
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        csv_path = Path(temp_dir) / f"{ticker}.csv"

        # Create sample return data
        dates = pd.date_range("2024-01-01", periods=100)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        df = pd.DataFrame({
            "Date": dates,
            "close": 150.0,
            "daily_return": returns
        })
        df.to_csv(csv_path, index=False)

        analyses[ticker] = TickerAnalysis(
            ticker=ticker,
            csv_path=csv_path,
            chart_path=Path(f"{ticker}.png"),
            latest_close=150.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={},
            fundamentals=None,
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
            sample_data=[],
            error=None,
        )

    yield analyses

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestAnalyzeAllTickersErrorRecovery:
    """Test error recovery in _analyze_all_tickers method."""

    @patch("orchestrator.CacheManager")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_handles_exception_from_worker_thread(
        self,
        mock_chart,
        mock_llm,
        mock_cache,
        mock_config,
        temp_output_dir,
    ):
        """Should handle exceptions raised in worker threads gracefully."""
        # Setup orchestrator
        orchestrator = FinancialReportOrchestrator(mock_config)

        # Mock analyze_ticker to fail for specific ticker
        #original_analyze = orchestrator.analyze_ticker

        def mock_analyze_ticker(ticker, period, output_path, benchmark_returns):
            if ticker == "FAIL":
                raise RuntimeError("Worker thread crashed")
            # For successful tickers, return a normal analysis
            return TickerAnalysis(
                ticker=ticker,
                csv_path=Path(f"{ticker}.csv"),
                chart_path=Path(f"{ticker}.png"),
                latest_close=100.0,
                avg_daily_return=0.001,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
                sample_data=[],
                error=None,
            )

        orchestrator.analyze_ticker = mock_analyze_ticker

        # Analyze tickers including one that will fail
        tickers = ["AAPL", "FAIL", "MSFT"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # Should have all 3 analyses
        assert len(analyses) == 3
        assert "AAPL" in analyses
        assert "FAIL" in analyses
        assert "MSFT" in analyses

        # Failed ticker should have error analysis
        assert analyses["FAIL"].error is not None
        assert "Thread error" in analyses["FAIL"].error
        assert analyses["FAIL"].latest_close == 0.0

        # Successful tickers should be fine
        assert analyses["AAPL"].error is None
        assert analyses["MSFT"].error is None

    @patch("orchestrator.CacheManager")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_handles_multiple_concurrent_failures(
        self,
        mock_chart,
        mock_llm,
        mock_cache,
        mock_config,
        temp_output_dir,
    ):
        """Should handle multiple concurrent failures without stopping."""
        orchestrator = FinancialReportOrchestrator(mock_config)

        # Mock analyze_ticker to fail for specific tickers
        def mock_analyze_ticker(ticker, period, output_path, benchmark_returns):
            if ticker in ["FAIL1", "FAIL2"]:
                raise ValueError(f"Error in {ticker}")
            return TickerAnalysis(
                ticker=ticker,
                csv_path=Path(f"{ticker}.csv"),
                chart_path=Path(f"{ticker}.png"),
                latest_close=100.0,
                avg_daily_return=0.001,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
                sample_data=[],
                error=None,
            )

        orchestrator.analyze_ticker = mock_analyze_ticker

        tickers = ["AAPL", "FAIL1", "MSFT", "FAIL2", "GOOGL"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # All tickers should be present
        assert len(analyses) == 5

        # Failed tickers should have errors
        assert analyses["FAIL1"].error is not None
        assert analyses["FAIL2"].error is not None
        assert "Thread error" in analyses["FAIL1"].error
        assert "Thread error" in analyses["FAIL2"].error

        # Successful tickers should be fine
        assert analyses["AAPL"].error is None
        assert analyses["MSFT"].error is None
        assert analyses["GOOGL"].error is None


class TestPortfolioWeights:
    """Test invalid portfolio weight combinations."""

    def test_mixed_negative_and_positive_weights_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject portfolios with mixed negative and positive weights."""
        # Mix of negative and positive weights (still sum to 1)
        weights = {"AAPL": 1.5, "MSFT": -0.3, "GOOGL": -0.2}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_all_negative_weights_rejected(self, portfolio_analyzer, mock_analyses):
        """Should reject portfolios where all weights are negative."""
        weights = {"AAPL": -0.5, "MSFT": -0.3, "GOOGL": -0.2}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_zero_weight_mixed_with_positive_accepted(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should accept zero weights mixed with positive (if sum to 1)."""
        weights = {"AAPL": 0.5, "MSFT": 0.5, "GOOGL": 0.0}

        # Should not raise (zeros are allowed)
        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None
        assert result.weights == weights

    def test_very_small_negative_weight_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject even very small negative weights."""
        # Tiny negative weight that might be from floating point errors
        weights = {"AAPL": 0.6, "MSFT": 0.4, "GOOGL": -0.0001}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_weights_sum_slightly_over_one_with_negative_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject weights that sum > 1 and include negatives."""
        # Sum to 1.1 with negative component
        weights = {"AAPL": 1.0, "MSFT": 0.5, "GOOGL": -0.4}

        # Should fail on negative check first
        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_weights_sum_slightly_under_one_with_negative_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject weights that sum < 1 and include negatives."""
        # Sum to 0.9 with negative component
        weights = {"AAPL": 0.5, "MSFT": 0.6, "GOOGL": -0.2}

        # Should fail on negative check first
        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_single_negative_weight_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject portfolio with single negative weight."""
        weights = {"AAPL": -1.0}

        # Create single analysis
        single_analysis = {"AAPL": mock_analyses["AAPL"]}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(single_analysis, weights)

    def test_extreme_weight_concentration_with_zero(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should handle extreme concentration (one weight = 1, others = 0)."""
        weights = {"AAPL": 1.0, "MSFT": 0.0, "GOOGL": 0.0}

        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None
        # Concentration risk (Herfindahl) should be 1.0 (max concentration)
        assert result.concentration_risk == 1.0

    def test_fractional_negative_weights_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject fractional negative weights."""
        weights = {"AAPL": 0.7, "MSFT": 0.35, "GOOGL": -0.05}

        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)

    def test_weights_with_multiple_validation_errors(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should catch negative weights before sum validation."""
        # Both negative AND sum != 1
        weights = {"AAPL": -0.5, "MSFT": -0.3, "GOOGL": 0.5}

        # Should fail on negative check first (not sum check)
        with pytest.raises(ValueError, match="cannot be negative"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)


class TestPortfolioWeightEdgeCases:
    """Test edge cases in portfolio weight handling."""

    def test_empty_weights_dict_uses_equal_weights(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should use equal weights when weights dict is None."""
        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, None)

        assert result is not None
        assert result.weights is not None
        # Should have equal weights for 3 tickers
        expected_weight = 1.0 / 3.0
        for weight in result.weights.values():
            assert abs(weight - expected_weight) < 0.001

    def test_weights_sum_to_exactly_one(self, portfolio_analyzer, mock_analyses):
        """Should accept weights that sum to exactly 1.0."""
        weights = {"AAPL": 0.3333333333333333, "MSFT": 0.3333333333333333, "GOOGL": 0.3333333333333334}

        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None

    def test_weights_sum_within_tolerance(self, portfolio_analyzer, mock_analyses):
        """Should accept weights within tolerance of 1.0."""
        # Sum to 1.001 (within tolerance)
        weights = {"AAPL": 0.334, "MSFT": 0.333, "GOOGL": 0.334}

        result = portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)
        assert result is not None

    def test_weights_outside_tolerance_rejected(
        self, portfolio_analyzer, mock_analyses
    ):
        """Should reject weights outside tolerance."""
        # Sum to 1.05 (outside tolerance)
        weights = {"AAPL": 0.4, "MSFT": 0.35, "GOOGL": 0.3}

        with pytest.raises(ValueError, match="must sum to 1.0"):
            portfolio_analyzer.calculate_portfolio_metrics(mock_analyses, weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
