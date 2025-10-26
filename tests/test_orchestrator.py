"""
Integration tests for orchestrator.py module.

Tests the full orchestration workflow with mocked external APIs (yfinance, OpenAI).
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import timezone

from orchestrator import FinancialReportOrchestrator
from config import Config
from models import TickerAnalysis, PortfolioMetrics, PortfolioRequest


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
    config.generate_html = True
    config.embed_images_in_html = False
    config.open_in_browser = False
    return config


@pytest.fixture
def mock_price_data():
    """Create mock price data from yfinance."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D", tz=timezone.utc)
    np.random.seed(42)
    base_price = 150
    prices = base_price + np.cumsum(np.random.randn(100) * 2)

    # yfinance returns data with Date as index
    df = pd.DataFrame(
        {
            "Close": prices,
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Volume": np.random.randint(1000000, 10000000, 100),
        },
        index=dates
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def mock_ticker_info():
    """Create mock ticker info from yfinance."""
    return {
        "trailingPE": 28.5,
        "forwardPE": 25.0,
        "priceToBook": 35.0,
        "debtToEquity": 150.0,
        "profitMargins": 0.25,
        "returnOnEquity": 0.95,
        "beta": 1.3,
    }


@pytest.fixture
def mock_income_stmt():
    """Create mock income statement from yfinance."""
    return pd.DataFrame(
        {
            "TotalRevenue": [394e9, 383e9, 365e9, 347e9, 327e9],
            "NetIncome": [97e9, 95e9, 90e9, 86e9, 81e9],
        }
    )


@pytest.fixture
def mock_cashflow():
    """Create mock cash flow statement from yfinance."""
    return pd.DataFrame({"FreeCashFlow": [99e9, 96e9, 93e9, 89e9, 85e9]})


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_orchestrator_initializes_with_config(self, mock_config):
        """Test that orchestrator initializes successfully with config."""
        with patch("orchestrator.IntegratedLLMInterface"), patch(
            "orchestrator.CacheManager"
        ), patch("orchestrator.ThreadSafeChartGenerator"):

            orchestrator = FinancialReportOrchestrator(mock_config)

            assert orchestrator.config == mock_config
            assert orchestrator.analyzer is not None
            assert orchestrator.alert_system is not None

    def test_orchestrator_creates_required_components(self, mock_config):
        """Test that orchestrator creates all required components."""
        with patch("orchestrator.IntegratedLLMInterface") as mock_llm, patch(
            "orchestrator.CacheManager"
        ) as mock_cache, patch("orchestrator.ThreadSafeChartGenerator") as mock_chart:

            _orchestrator = FinancialReportOrchestrator(mock_config)

            # Verify all components were created
            mock_llm.assert_called_once()
            mock_cache.assert_called_once()
            mock_chart.assert_called_once()


class TestAnalyzeTicker:
    """Test single ticker analysis."""

    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_analyze_ticker_returns_ticker_analysis(
        self,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_config,
        mock_price_data,
        mock_ticker_info,
        temp_output_dir,
    ):
        """Test that analyze_ticker returns a TickerAnalysis object."""
        # Setup mocks
        mock_ticker_obj = Mock()
        mock_ticker_obj.history.return_value = mock_price_data.copy()
        mock_ticker_obj.info = mock_ticker_info
        mock_ticker_obj.income_stmt = pd.DataFrame({"TotalRevenue": [100e9] * 5})
        mock_ticker_obj.cashflow = pd.DataFrame({"FreeCashFlow": [20e9] * 5})
        mock_yf.return_value = mock_ticker_obj

        # Cache returns None (no cache hit)
        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Run analysis
        result = orchestrator.analyze_ticker("AAPL", "1y", Path(temp_output_dir))

        # Verify result
        assert isinstance(result, TickerAnalysis)
        assert result.ticker == "AAPL"
        assert result.error is None
        assert result.latest_close > 0
        assert result.csv_path is not None
        assert result.chart_path is not None

    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_analyze_ticker_handles_errors_gracefully(
        self,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_config,
        temp_output_dir,
    ):
        """Test that analyze_ticker handles errors without crashing."""
        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Mock the fetcher to raise a ValueError (which is caught by analyze_ticker)
        orchestrator.fetcher.fetch_price_history = Mock(
            side_effect=ValueError("API Error")
        )

        # Run analysis - should not raise exception
        result = orchestrator.analyze_ticker("INVALID", "1y", Path(temp_output_dir))

        # Should return TickerAnalysis with error
        assert isinstance(result, TickerAnalysis)
        assert result.error is not None
        assert "INVALID" in result.ticker


class TestGenerateReport:
    """Test full report generation."""

    @patch("builtins.print")  # Suppress print to avoid Unicode errors in tests
    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    @patch("orchestrator.HTMLGenerator")
    def test_generate_report_ticker_mode(
        self,
        mock_html_gen,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_print,
        mock_config,
        mock_price_data,
        mock_ticker_info,
        mock_income_stmt,
        mock_cashflow,
        temp_output_dir,
    ):
        """Test report generation in ticker mode."""
        # Setup mocks - use side_effect to return different mock for each ticker
        def create_mock_ticker(ticker_symbol, **kwargs):  # Accept any kwargs
            mock_obj = Mock()
            # Mock history to be a Mock that returns data when called
            mock_obj.history.return_value = mock_price_data.copy()
            mock_obj.info = mock_ticker_info if ticker_symbol == "AAPL" else {"trailingPE": 20.0}
            mock_obj.income_stmt = mock_income_stmt if ticker_symbol == "AAPL" else pd.DataFrame({"TotalRevenue": [100e9] * 5})
            mock_obj.cashflow = mock_cashflow if ticker_symbol == "AAPL" else pd.DataFrame({"FreeCashFlow": [20e9] * 5})
            return mock_obj

        mock_yf.side_effect = create_mock_ticker

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        mock_llm_instance = Mock()
        mock_llm_instance.generate_detailed_report.return_value = "# Test Report\n\nThis is a test report."
        mock_llm_instance.review_report.return_value = ([], 8.5, {"quality_score": 8.5, "issues": [], "suggestions": []})
        mock_llm.return_value = mock_llm_instance

        mock_html_gen_instance = Mock()
        mock_html_gen_instance.generate_html_report.return_value = Path(temp_output_dir) / "test_report.html"
        mock_html_gen.return_value = mock_html_gen_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Generate report with Pydantic request
        portfolio_request = PortfolioRequest(tickers=["AAPL"], period="1y")
        report_metadata = orchestrator.run(
            request=portfolio_request, output_dir=str(temp_output_dir)
        )

        # Verify report was created
        assert report_metadata is not None
        assert report_metadata.final_markdown_path is not None
        assert Path(report_metadata.final_markdown_path).exists()
        assert Path(report_metadata.final_markdown_path).suffix == ".md"

    @patch("builtins.print")  # Suppress print to avoid Unicode errors in tests
    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    @patch("orchestrator.HTMLGenerator")
    def test_generate_report_portfolio_mode(
        self,
        mock_html_gen,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_print,
        mock_config,
        mock_price_data,
        temp_output_dir,
    ):
        """Test report generation in portfolio mode with weights."""

        # Setup mocks for multiple tickers
        def create_ticker_mock(ticker_symbol, **kwargs):  # Accept any kwargs
            mock_obj = Mock()
            # Mock history to return data when called
            mock_obj.history.return_value = mock_price_data.copy()
            mock_obj.info = {"trailingPE": 20.0, "beta": 1.0}
            mock_obj.income_stmt = pd.DataFrame({"TotalRevenue": [100e9] * 5})
            mock_obj.cashflow = pd.DataFrame({"FreeCashFlow": [20e9] * 5})
            return mock_obj

        mock_yf.side_effect = create_ticker_mock

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        mock_llm_instance = Mock()
        mock_llm_instance.generate_detailed_report.return_value = "# Portfolio Report\n\nPortfolio analysis."
        mock_llm_instance.review_report.return_value = ([], 8.5, {"quality_score": 8.5, "issues": [], "suggestions": []})
        mock_llm.return_value = mock_llm_instance

        mock_html_gen_instance = Mock()
        mock_html_gen_instance.generate_html_report.return_value = Path(temp_output_dir) / "test_report.html"
        mock_html_gen.return_value = mock_html_gen_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Generate portfolio report with Pydantic request
        portfolio_request = PortfolioRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            period="1y",
            weights={"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
        )
        report_metadata = orchestrator.run(
            request=portfolio_request, output_dir=str(temp_output_dir)
        )

        # Verify report was created
        assert report_metadata is not None
        assert report_metadata.final_markdown_path is not None
        assert Path(report_metadata.final_markdown_path).exists()


class TestAnalyzeAllTickers:
    """Test concurrent ticker analysis."""

    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_analyze_all_tickers_concurrent(
        self,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_config,
        mock_price_data,
        temp_output_dir,
    ):
        """Test that multiple tickers are analyzed concurrently."""

        # Setup mocks
        def create_ticker_mock(ticker_symbol):
            mock_obj = Mock()
            # Don't add ticker column - it's added by the fetcher
            mock_obj.history.return_value = mock_price_data.copy()
            mock_obj.info = {"trailingPE": 20.0}
            mock_obj.income_stmt = pd.DataFrame({"TotalRevenue": [100e9] * 5})
            mock_obj.cashflow = pd.DataFrame({"FreeCashFlow": [20e9] * 5})
            return mock_obj

        mock_yf.side_effect = lambda t: create_ticker_mock(t)

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Analyze multiple tickers
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # Verify all tickers were analyzed
        assert len(analyses) == 4
        assert all(ticker in analyses for ticker in tickers)
        assert all(not analysis.error for analysis in analyses.values())


class TestPortfolioMetrics:
    """Test portfolio metrics calculation."""

    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_calculate_portfolio_metrics(
        self,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_config,
        mock_price_data,
        temp_output_dir,
    ):
        """Test portfolio metrics calculation with weights."""

        # Setup mocks
        def create_ticker_mock(ticker_symbol):
            mock_obj = Mock()
            data = mock_price_data.copy()
            # Vary prices slightly for each ticker
            data["Close"] = data["Close"] * (1 + np.random.rand() * 0.1)
            # Don't add ticker column - it's added by the fetcher
            mock_obj.history.return_value = data
            mock_obj.info = {"trailingPE": 20.0}
            mock_obj.income_stmt = pd.DataFrame({"TotalRevenue": [100e9] * 5})
            mock_obj.cashflow = pd.DataFrame({"FreeCashFlow": [20e9] * 5})
            return mock_obj

        mock_yf.side_effect = lambda t: create_ticker_mock(t)

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Analyze tickers
        tickers = ["AAPL", "MSFT"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # Calculate portfolio metrics
        weights = {"AAPL": 0.6, "MSFT": 0.4}
        portfolio_metrics = orchestrator._calculate_portfolio_metrics_if_needed(
            analyses, weights
        )

        # Verify metrics were calculated
        assert portfolio_metrics is not None
        assert isinstance(portfolio_metrics, PortfolioMetrics)
        assert portfolio_metrics.portfolio_return is not None
        assert portfolio_metrics.portfolio_volatility is not None
        assert portfolio_metrics.portfolio_sharpe is not None


class TestBenchmarkFetching:
    """Test benchmark data fetching."""

    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_fetch_benchmark(
        self,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_config,
        mock_price_data,
        temp_output_dir,
    ):
        """Test fetching benchmark data."""
        # Setup mocks
        mock_ticker_obj = Mock()
        mock_ticker_obj.history.return_value = mock_price_data.copy()
        mock_ticker_obj.info = {}
        mock_yf.return_value = mock_ticker_obj

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Fetch benchmark
        benchmark_analysis, benchmark_returns = orchestrator._fetch_benchmark(
            "1y", Path(temp_output_dir)
        )

        # Verify benchmark was fetched
        assert benchmark_analysis is not None
        assert benchmark_returns is None or isinstance(benchmark_returns, pd.Series)


class TestErrorHandling:
    """Test error handling in orchestrator."""

    @patch("fetcher.yf.Ticker")
    @patch("orchestrator.IntegratedLLMInterface")
    @patch("orchestrator.CacheManager")
    @patch("orchestrator.ThreadSafeChartGenerator")
    def test_handles_all_failed_analyses(
        self,
        mock_chart,
        mock_cache,
        mock_llm,
        mock_yf,
        mock_config,
        temp_output_dir,
    ):
        """Test handling when all ticker analyses fail."""
        # Setup mock to always fail
        mock_yf.side_effect = Exception("Network error")

        mock_cache_instance = Mock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        orchestrator = FinancialReportOrchestrator(mock_config)
        orchestrator.fetcher.cache = mock_cache_instance

        # Analyze tickers - should all fail
        tickers = ["AAPL", "MSFT"]
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", Path(temp_output_dir), None
        )

        # Process results - should raise ValueError
        with pytest.raises(ValueError, match="All analyses failed"):
            orchestrator._process_analysis_results(analyses)


class TestAnalyzeAllTickersExtended:
    """Test concurrent ticker analysis with failures."""

    @patch("orchestrator.FinancialReportOrchestrator.analyze_ticker")
    def test_analyze_all_tickers_handles_partial_failures(
        self,
        mock_analyze_ticker,
        mock_config,
    ):
        """Test that _analyze_all_tickers continues if one ticker fails."""
        # Setup mocks
        orchestrator = FinancialReportOrchestrator(mock_config)

        # Mock analyze_ticker to succeed for some and fail for others
        def side_effect(ticker, period, output_dir, benchmark_returns, include_options=False, options_expirations=3):
            if ticker == "FAIL":
                raise Exception("Simulated analysis failure")
            else:
                return TickerAnalysis(
                    ticker=ticker,
                    csv_path=f"{ticker}.csv",
                    chart_path=f"{ticker}.png",
                    latest_close=100.0,
                    avg_daily_return=0.001,
                    volatility=0.02,
                    ratios={},
                    fundamentals=None,
                    advanced_metrics=None,
                    technical_indicators=None,
                    sample_data=[],
                    error=None,
                )

        mock_analyze_ticker.side_effect = side_effect

        tickers = ["AAPL", "MSFT", "FAIL", "GOOGL"]
        output_path = Path("/tmp")
        analyses = orchestrator._analyze_all_tickers(
            tickers, "1y", output_path, None
        )

        # Verify results
        assert len(analyses) == 4
        assert "AAPL" in analyses and analyses["AAPL"].error is None
        assert "MSFT" in analyses and analyses["MSFT"].error is None
        assert "GOOGL" in analyses and analyses["GOOGL"].error is None
        assert "FAIL" in analyses and "Thread error" in analyses["FAIL"].error

        # Verify that the successful analyses have data
        assert analyses["AAPL"].latest_close == 100.0
        assert analyses["MSFT"].csv_path == "MSFT.csv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
