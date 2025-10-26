"""
Test suite for long methods refactoring.

Tests that the refactored orchestrator.py and llm_interface.py methods:
1. Maintain the same functionality as before
2. Follow single responsibility principle
3. Have proper error handling
4. Generate correct outputs
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    TickerAnalysis,
    AdvancedMetrics,
    TechnicalIndicators,
    FundamentalData,
    PortfolioMetrics,
)
from config import Config
from orchestrator import FinancialReportOrchestrator
from llm_interface import IntegratedLLMInterface


class TestOrchestrator:
    """Test refactored orchestrator methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir) / "test_output"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create mock config
        self.config = Mock(spec=Config)
        self.config.benchmark_ticker = "SPY"
        self.config.openai_api_key = "test_key"
        self.config.openai_base_url = "https://api.openai.com/v1"
        self.config.cache_ttl_hours = 24
        self.config.max_workers = 4  # Set max_workers for ThreadPoolExecutor
        self.config.request_timeout = 30
        self.config.model_name = "gpt-4o"
        self.config.risk_free_rate = 0.02
        self.config.generate_html = True
        self.config.embed_images_in_html = False
        self.config.open_in_browser = False

        # Create orchestrator
        with patch("orchestrator.IntegratedLLMInterface"), patch(
            "orchestrator.CacheManager"
        ), patch("orchestrator.ThreadSafeChartGenerator"), patch(
            "orchestrator.HTMLGenerator"
        ):
            self.orchestrator = FinancialReportOrchestrator(self.config)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_mock_analysis(
        self, ticker: str, has_error: bool = False
    ) -> TickerAnalysis:
        """Create a mock TickerAnalysis object."""
        if has_error:
            return TickerAnalysis(
                ticker=ticker,
                csv_path=str(self.output_path / f"{ticker}.csv"),
                chart_path=str(self.output_path / f"{ticker}.png"),
                latest_close=100.0,
                avg_daily_return=0.001,
                volatility=0.02,
                ratios={},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
                error="Test error",
            )

        return TickerAnalysis(
            ticker=ticker,
            csv_path=str(self.output_path / f"{ticker}.csv"),
            chart_path=str(self.output_path / f"{ticker}.png"),
            latest_close=100.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={"P/E": 15.0, "ROE": 0.12},
            fundamentals=FundamentalData(revenue=1000000.0),
            advanced_metrics=AdvancedMetrics(sharpe_ratio=1.5, max_drawdown=-0.10),
            technical_indicators=TechnicalIndicators(rsi=55.0),
            alerts=["Test alert"],
        )

    def test_fetch_benchmark_success(self):
        """Test successful benchmark fetching."""
        # Mock analyze_ticker to return valid analysis
        mock_analysis = self.create_mock_analysis("SPY")

        # Create mock CSV with benchmark data
        csv_path = Path(mock_analysis.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=10),
                "close": [100 + i for i in range(10)],
                "daily_return": [0.01] * 10,
            }
        )
        df.to_csv(csv_path, index=False)

        with patch.object(
            self.orchestrator, "analyze_ticker", return_value=mock_analysis
        ), patch(
            "builtins.print"
        ):  # Suppress print to avoid Unicode errors
            benchmark_analysis, benchmark_returns = self.orchestrator._fetch_benchmark(
                "1y", self.output_path
            )

        assert benchmark_analysis is not None
        assert benchmark_analysis.ticker == "SPY"
        assert benchmark_returns is not None
        assert len(benchmark_returns) == 10

    def test_fetch_benchmark_failure(self):
        """Test benchmark fetching with errors."""
        mock_analysis = self.create_mock_analysis("SPY", has_error=True)

        with patch.object(
            self.orchestrator, "analyze_ticker", return_value=mock_analysis
        ), patch(
            "builtins.print"
        ):  # Suppress print
            benchmark_analysis, benchmark_returns = self.orchestrator._fetch_benchmark(
                "1y", self.output_path
            )

        assert benchmark_analysis is not None
        assert benchmark_returns is None

    def test_analyze_all_tickers(self):
        """Test analyzing multiple tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        mock_analyses = {t: self.create_mock_analysis(t) for t in tickers}

        def mock_analyze(ticker, period, output_path, benchmark_returns=None, include_options=False, options_expirations=3):
            return mock_analyses[ticker]

        with patch.object(
            self.orchestrator, "analyze_ticker", side_effect=mock_analyze
        ), patch(
            "builtins.print"
        ):  # Suppress print
            analyses = self.orchestrator._analyze_all_tickers(
                tickers, "1y", self.output_path, None
            )

        assert len(analyses) == 3
        assert all(ticker in analyses for ticker in tickers)
        assert all(not a.error for a in analyses.values())

    def test_process_analysis_results_success(self):
        """Test processing successful and failed analyses."""
        analyses = {
            "AAPL": self.create_mock_analysis("AAPL"),
            "MSFT": self.create_mock_analysis("MSFT"),
            "FAIL": self.create_mock_analysis("FAIL", has_error=True),
        }

        with patch("builtins.print"):  # Suppress print
            successful, failed = self.orchestrator._process_analysis_results(analyses)

        assert len(successful) == 2
        assert len(failed) == 1
        assert "AAPL" in successful
        assert "MSFT" in successful
        assert "FAIL" in failed

    def test_process_analysis_results_all_failed(self):
        """Test when all analyses fail."""
        analyses = {
            "FAIL1": self.create_mock_analysis("FAIL1", has_error=True),
            "FAIL2": self.create_mock_analysis("FAIL2", has_error=True),
        }

        try:
            self.orchestrator._process_analysis_results(analyses)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "All analyses failed" in str(e)

    def test_calculate_portfolio_metrics_multiple_tickers(self):
        """Test portfolio metrics calculation with 2+ tickers."""
        successful = {
            "AAPL": self.create_mock_analysis("AAPL"),
            "MSFT": self.create_mock_analysis("MSFT"),
        }

        mock_portfolio_metrics = PortfolioMetrics(
            total_value=10000.0,
            portfolio_return=0.15,
            portfolio_volatility=0.18,
            portfolio_sharpe=0.8,
        )

        with patch.object(
            self.orchestrator.portfolio_analyzer,
            "calculate_portfolio_metrics",
            return_value=mock_portfolio_metrics,
        ), patch(
            "builtins.print"
        ):  # Suppress print
            result = self.orchestrator._calculate_portfolio_metrics_if_needed(
                successful, None
            )

        assert result is not None
        assert result.total_value == 10000.0

    def test_calculate_portfolio_metrics_single_ticker(self):
        """Test that portfolio metrics are not calculated for single ticker."""
        successful = {"AAPL": self.create_mock_analysis("AAPL")}

        result = self.orchestrator._calculate_portfolio_metrics_if_needed(
            successful, None
        )

        assert result is None

    def test_generate_comparison_charts(self):
        """Test comparison chart generation."""
        successful = {
            "AAPL": self.create_mock_analysis("AAPL"),
            "MSFT": self.create_mock_analysis("MSFT"),
        }

        with patch.object(
            self.orchestrator.chart_gen, "create_comparison_chart"
        ), patch(
            "builtins.print"
        ):  # Suppress print
            chart_files = self.orchestrator._generate_comparison_charts(
                successful, self.output_path
            )

        # Should include individual charts + comparison chart
        assert len(chart_files) >= 2

    def test_save_report(self):
        """Test report saving."""
        report_content = "# Test Report\n\nThis is a test."

        report_path, timestamp = self.orchestrator._save_report(
            report_content, self.output_path
        )

        assert report_path.exists()
        assert report_path.suffix == ".md"
        assert timestamp.endswith("Z")  # UTC format

        with open(report_path, "r") as f:
            content = f.read()
        assert content == report_content

    def test_collect_performance_metrics(self):
        """Test performance metrics collection."""
        import time

        start_time = time.time() - 10.0  # Simulate 10 seconds

        successful = {
            "AAPL": self.create_mock_analysis("AAPL"),
            "MSFT": self.create_mock_analysis("MSFT"),
        }
        failed = {"FAIL": self.create_mock_analysis("FAIL", has_error=True)}

        metrics = self.orchestrator._collect_performance_metrics(
            start_time=start_time,
            tickers=["AAPL", "MSFT", "FAIL"],
            successful=successful,
            failed=failed,
            chart_files=["chart1.png", "chart2.png"],
            quality_score=8,
            portfolio_metrics=None,
        )

        assert metrics["tickers_analyzed"] == 3
        assert metrics["successful"] == 2
        assert metrics["failed"] == 1
        assert metrics["charts_generated"] == 2
        assert metrics["quality_score"] == 8
        assert metrics["portfolio_analyzed"] is False
        assert metrics["execution_time_seconds"] >= 10.0


class TestLLMInterface:
    """Test refactored LLM interface methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock(spec=Config)
        self.config.openai_api_key = "test_key"
        self.config.openai_base_url = "https://api.openai.com/v1"
        self.config.request_timeout = 30
        self.config.model_name = "gpt-4o"

        with patch("llm_interface.ChatOpenAI"):
            self.llm = IntegratedLLMInterface(self.config)

    def create_mock_analysis(self, ticker: str) -> TickerAnalysis:
        """Create a mock TickerAnalysis object."""
        return TickerAnalysis(
            ticker=ticker,
            csv_path=Path(f"/tmp/{ticker}.csv"),
            chart_path=Path(f"/tmp/{ticker}.png"),
            latest_close=100.0,
            avg_daily_return=0.001,
            volatility=0.02,
            ratios={"P/E": 15.0},
            fundamentals=FundamentalData(revenue=1000000.0),
            advanced_metrics=AdvancedMetrics(sharpe_ratio=1.5),
            technical_indicators=TechnicalIndicators(rsi=55.0),
            alerts=["Test alert"],
        )

    def test_generate_report_header(self):
        """Test report header generation."""
        header = self.llm._generate_report_header()

        assert len(header) == 3
        assert header[0].startswith("# ")
        assert "Financial Analysis Report" in header[0]
        assert "*Generated:" in header[1]
        assert header[2] == ""

    def test_generate_executive_summary_section(self):
        """Test executive summary section generation."""
        analyses = {"AAPL": self.create_mock_analysis("AAPL")}

        with patch.object(
            self.llm, "generate_narrative_summary", return_value="Test narrative"
        ):
            section = self.llm._generate_executive_summary_section(analyses, "1y")

        assert len(section) == 3
        assert "Executive Summary" in section[0]
        assert section[1] == "Test narrative"
        assert section[2] == ""

    def test_generate_portfolio_overview_section_with_metrics(self):
        """Test portfolio overview section with metrics."""
        portfolio_metrics = PortfolioMetrics(
            total_value=10000.0,
            portfolio_return=0.15,
            portfolio_volatility=0.18,
            portfolio_sharpe=0.8,
        )

        with patch.object(
            self.llm, "_generate_portfolio_section", return_value="Portfolio content"
        ):
            section = self.llm._generate_portfolio_overview_section(portfolio_metrics)

        assert len(section) == 3
        assert "Portfolio Overview" in section[0]
        assert section[1] == "Portfolio content"

    def test_generate_portfolio_overview_section_without_metrics(self):
        """Test portfolio overview section without metrics."""
        section = self.llm._generate_portfolio_overview_section(None)
        assert section == []

    def test_generate_alerts_section_with_alerts(self):
        """Test alerts section with active alerts."""
        analyses = {"AAPL": self.create_mock_analysis("AAPL")}
        analyses["AAPL"].alerts = ["Alert 1", "Alert 2"]

        section = self.llm._generate_alerts_section(analyses)

        assert len(section) == 4  # Header + 2 alerts + blank line
        assert "Active Alerts" in section[0]
        assert "- Alert 1" in section[1]
        assert "- Alert 2" in section[2]

    def test_generate_alerts_section_without_alerts(self):
        """Test alerts section without alerts."""
        analyses = {"AAPL": self.create_mock_analysis("AAPL")}
        analyses["AAPL"].alerts = []

        section = self.llm._generate_alerts_section(analyses)
        assert section == []

    def test_generate_failures_section_with_failures(self):
        """Test failures section with errors."""
        analyses = {
            "AAPL": self.create_mock_analysis("AAPL"),
            "FAIL": TickerAnalysis(
                ticker="FAIL",
                csv_path="",
                chart_path="",
                latest_close=0,
                avg_daily_return=0,
                volatility=0,
                ratios={},
                fundamentals=None,
                advanced_metrics=AdvancedMetrics(),
                technical_indicators=TechnicalIndicators(),
                error="Test error",
            ),
        }

        section = self.llm._generate_failures_section(analyses)

        assert len(section) == 3  # Header + 1 failure + blank line
        assert "Data Quality Notes" in section[0]
        assert "FAIL" in section[1]
        assert "Test error" in section[1]

    def test_generate_failures_section_without_failures(self):
        """Test failures section without errors."""
        analyses = {"AAPL": self.create_mock_analysis("AAPL")}

        section = self.llm._generate_failures_section(analyses)
        assert section == []

    def test_generate_detailed_report_structure(self):
        """Test that detailed report has correct structure."""
        analyses = {"AAPL": self.create_mock_analysis("AAPL")}

        portfolio_metrics = PortfolioMetrics(
            total_value=10000.0, portfolio_return=0.15, portfolio_volatility=0.18
        )

        with patch.object(
            self.llm, "generate_narrative_summary", return_value="Narrative"
        ), patch.object(
            self.llm, "_generate_portfolio_section", return_value="Portfolio"
        ), patch.object(
            self.llm, "_generate_metrics_table", return_value="Metrics"
        ), patch.object(
            self.llm, "_generate_fundamental_section", return_value="Fundamentals"
        ), patch.object(
            self.llm, "_generate_individual_analysis", return_value="Individual"
        ), patch.object(
            self.llm, "_generate_risk_analysis", return_value="Risk"
        ), patch.object(
            self.llm, "_generate_recommendations", return_value="Recommendations"
        ):

            report = self.llm.generate_detailed_report(
                analyses, None, portfolio_metrics, "1y"
            )

        # Verify report structure
        assert "# 📊 Financial Analysis Report" in report
        assert "## 🎯 Executive Summary" in report
        assert "## 💼 Portfolio Overview" in report
        assert "## 📈 Key Performance Metrics" in report
        assert "## 📊 Fundamental Analysis" in report
        assert "## 📋 Detailed Stock Analysis" in report
        assert "## ⚠️ Risk Analysis" in report
        assert "## 💡 Investment Recommendations" in report


def run_tests():
    """Run all tests."""
    import pytest

    # Run pytest
    exit_code = pytest.main([__file__, "-v", "-s"])
    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
