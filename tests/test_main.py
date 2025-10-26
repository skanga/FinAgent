"""
Comprehensive end-to-end tests for CLI interface.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from main import setup_cli, parse_weights, main
from models import ReportMetadata, TickerAnalysis, PortfolioMetrics


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator for testing."""
    mock = Mock()

    # Create a mock result
    mock_result = ReportMetadata(
        tickers=["AAPL"],
        period="1y",
        analyses={"AAPL": Mock(spec=TickerAnalysis)},
        final_markdown_path=Path("/tmp/report.md"),
        final_html_path=Path("/tmp/report.html"),
        executive_summary="Test summary",
        review_issues=[],
        review_suggestions=[],
        performance_metrics={
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        },
    )

    mock.run.return_value = mock_result
    mock.run_from_natural_language.return_value = mock_result

    return mock


class TestCLISetup:
    """Test CLI argument parser setup."""

    def test_parser_creation(self):
        """Test that parser is created successfully."""
        parser = setup_cli()

        assert parser is not None
        assert parser.description is not None

    def test_request_argument(self):
        """Test --request argument."""
        parser = setup_cli()

        args = parser.parse_args(["--request", "Analyze AAPL"])

        assert args.request == "Analyze AAPL"

    def test_tickers_argument(self):
        """Test --tickers argument."""
        parser = setup_cli()

        args = parser.parse_args(["--tickers", "AAPL,MSFT"])

        assert args.tickers == "AAPL,MSFT"

    def test_period_argument(self):
        """Test --period argument with valid period."""
        parser = setup_cli()

        args = parser.parse_args(["--period", "6mo"])

        assert args.period == "6mo"

    def test_invalid_period_raises_error(self):
        """Test that invalid period raises error."""
        parser = setup_cli()

        with pytest.raises(SystemExit):
            parser.parse_args(["--period", "invalid"])

    def test_weights_argument(self):
        """Test --weights argument."""
        parser = setup_cli()

        args = parser.parse_args(["--weights", "0.5,0.3,0.2"])

        assert args.weights == "0.5,0.3,0.2"

    def test_output_argument(self):
        """Test --output argument."""
        parser = setup_cli()

        args = parser.parse_args(["--output", "./custom_output"])

        assert args.output == "./custom_output"

    def test_default_output(self):
        """Test default output directory."""
        parser = setup_cli()

        args = parser.parse_args([])

        assert args.output == "./financial_reports"

    def test_clear_cache_flag(self):
        """Test --clear-cache flag."""
        parser = setup_cli()

        args = parser.parse_args(["--clear-cache"])

        assert args.clear_cache is True

    def test_verbose_flag(self):
        """Test --verbose flag."""
        parser = setup_cli()

        args = parser.parse_args(["--verbose"])

        assert args.verbose is True

    def test_no_html_flag(self):
        """Test --no-html flag."""
        parser = setup_cli()

        args = parser.parse_args(["--no-html"])

        assert args.no_html is True

    def test_no_browser_flag(self):
        """Test --no-browser flag."""
        parser = setup_cli()

        args = parser.parse_args(["--no-browser"])

        assert args.no_browser is True

    def test_embed_images_flag(self):
        """Test --embed-images flag."""
        parser = setup_cli()

        args = parser.parse_args(["--embed-images"])

        assert args.embed_images is True

    def test_options_flag(self):
        """Test --options flag."""
        parser = setup_cli()

        args = parser.parse_args(["--options"])

        assert args.options is True

    def test_options_expirations_argument(self):
        """Test --options-expirations argument."""
        parser = setup_cli()

        args = parser.parse_args(["--options-expirations", "5"])

        assert args.options_expirations == 5

    def test_default_options_expirations(self):
        """Test default options expirations value."""
        parser = setup_cli()

        args = parser.parse_args([])

        assert args.options_expirations == 3


class TestWeightsParser:
    """Test weight parsing functionality."""

    def test_parse_valid_weights(self):
        """Test parsing valid weights."""
        weights = parse_weights("0.5,0.3,0.2", ["AAPL", "MSFT", "GOOGL"])

        assert weights == {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

    def test_parse_weights_with_spaces(self):
        """Test parsing weights with spaces."""
        weights = parse_weights(" 0.5 , 0.3 , 0.2 ", ["AAPL", "MSFT", "GOOGL"])

        assert weights == {"AAPL": 0.5, "MSFT": 0.3, "GOOGL": 0.2}

    def test_parse_none_weights(self):
        """Test parsing None weights."""
        weights = parse_weights(None, ["AAPL"])

        assert weights is None

    def test_parse_empty_weights(self):
        """Test parsing empty string weights."""
        weights = parse_weights("", ["AAPL"])

        assert weights is None

    def test_parse_weights_count_mismatch(self):
        """Test error when weights count doesn't match tickers."""
        with pytest.raises(ValueError, match="Number of weights.*must match"):
            parse_weights("0.5,0.5", ["AAPL", "MSFT", "GOOGL"])

    def test_parse_invalid_weight_format(self):
        """Test error for invalid weight format."""
        with pytest.raises(ValueError, match="Invalid weights format"):
            parse_weights("0.5,invalid,0.2", ["AAPL", "MSFT", "GOOGL"])

    def test_parse_single_weight(self):
        """Test parsing single weight."""
        weights = parse_weights("1.0", ["AAPL"])

        assert weights == {"AAPL": 1.0}


class TestMainFunction:
    """Test main function execution."""

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_main_with_default_tickers(self, mock_orch_class, mock_config):
        """Test main with default tickers."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        # Create mock result
        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        # Mock sys.argv
        with patch.object(sys, "argv", ["main.py"]):
            result = main()

        assert result == 0
        mock_orchestrator.run.assert_called_once()

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_main_with_tickers(self, mock_orch_class, mock_config):
        """Test main with specified tickers."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 2,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--tickers", "AAPL,MSFT"]):
            result = main()

        assert result == 0
        mock_orchestrator.run.assert_called_once()

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_main_with_natural_language(self, mock_orch_class, mock_config):
        """Test main with natural language request."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run_from_natural_language.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--request", "Analyze AAPL"]):
            result = main()

        assert result == 0
        mock_orchestrator.run_from_natural_language.assert_called_once()

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_main_with_weights(self, mock_orch_class, mock_config):
        """Test main with portfolio weights."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 2,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--tickers", "AAPL,MSFT", "--weights", "0.6,0.4"]):
            result = main()

        assert result == 0

    @patch("main.Config.from_env")
    @patch("cache.CacheManager")  # Mock from cache module, not main
    @patch("main.FinancialReportOrchestrator")
    def test_main_with_clear_cache(self, mock_orch_class, mock_cache_class, mock_config):
        """Test main with --clear-cache flag."""
        mock_config.return_value = Mock()
        mock_cache = Mock()
        mock_cache.clear_expired.return_value = 5
        mock_cache_class.return_value = mock_cache

        # Setup orchestrator mock
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--clear-cache"]):
            result = main()

        mock_cache.clear_expired.assert_called_once()
        assert result == 0

    @patch("main.Config.from_env")
    def test_main_handles_keyboard_interrupt(self, mock_config):
        """Test that KeyboardInterrupt is handled gracefully."""
        mock_config.side_effect = KeyboardInterrupt()

        with patch.object(sys, "argv", ["main.py"]):
            result = main()

        assert result == 1

    @patch("main.Config.from_env")
    def test_main_handles_exceptions(self, mock_config):
        """Test that exceptions are handled gracefully."""
        mock_config.side_effect = Exception("Test error")

        with patch.object(sys, "argv", ["main.py"]):
            result = main()

        assert result == 1


class TestHTMLOptionsIntegration:
    """Test HTML report options integration."""

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_no_html_flag_disables_html(self, mock_orch_class, mock_config):
        """Test that --no-html flag disables HTML generation."""
        mock_config_instance = Mock()
        mock_config_instance.generate_html = True
        mock_config.return_value = mock_config_instance

        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--no-html"]):
            main()

        assert mock_config_instance.generate_html is False

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_no_browser_flag_disables_browser(self, mock_orch_class, mock_config):
        """Test that --no-browser flag disables browser opening."""
        mock_config_instance = Mock()
        mock_config_instance.open_in_browser = True
        mock_config.return_value = mock_config_instance

        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--no-browser"]):
            main()

        assert mock_config_instance.open_in_browser is False

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_embed_images_flag_enables_embedding(self, mock_orch_class, mock_config):
        """Test that --embed-images flag enables image embedding."""
        mock_config_instance = Mock()
        mock_config_instance.embed_images_in_html = False
        mock_config.return_value = mock_config_instance

        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--embed-images"]):
            main()

        assert mock_config_instance.embed_images_in_html is True


class TestOptionsAnalysisIntegration:
    """Test options analysis integration."""

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_options_flag_enables_options_analysis(self, mock_orch_class, mock_config):
        """Test that --options flag enables options analysis."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--options"]):
            main()

        # Verify run was called with include_options=True
        call_args = mock_orchestrator.run.call_args
        portfolio_request = call_args[0][0]
        assert portfolio_request.include_options is True

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_options_expirations_parameter(self, mock_orch_class, mock_config):
        """Test that --options-expirations parameter is passed."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--options", "--options-expirations", "5"]):
            main()

        call_args = mock_orchestrator.run.call_args
        portfolio_request = call_args[0][0]
        assert portfolio_request.options_expirations == 5

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_options_expirations_capped_at_10(self, mock_orch_class, mock_config):
        """Test that options expirations is capped at 10."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--options", "--options-expirations", "20"]):
            main()

        call_args = mock_orchestrator.run.call_args
        portfolio_request = call_args[0][0]
        assert portfolio_request.options_expirations == 10  # Capped at 10


class TestVerboseLogging:
    """Test verbose logging functionality."""

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    @patch("main.logging.getLogger")
    def test_verbose_flag_sets_debug_level(self, mock_get_logger, mock_orch_class, mock_config):
        """Test that --verbose flag sets DEBUG log level."""
        mock_config.return_value = Mock()
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py", "--verbose"]):
            main()

        # Verify setLevel was called with DEBUG
        mock_logger.setLevel.assert_called()


class TestReportSummaryOutput:
    """Test report summary output."""

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_portfolio_metrics_displayed(self, mock_orch_class, mock_config):
        """Test that portfolio metrics are displayed when present."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 2,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 9,
        }
        mock_result.portfolio_metrics = Mock(spec=PortfolioMetrics)
        mock_result.portfolio_metrics.total_value = 100000.0
        mock_result.portfolio_metrics.portfolio_return = 0.15
        mock_result.portfolio_metrics.portfolio_volatility = 0.20
        mock_result.portfolio_metrics.portfolio_sharpe = 1.5
        mock_result.portfolio_metrics.diversification_ratio = 1.2
        mock_result.review_issues = []
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py"]):
            result = main()

        assert result == 0

    @patch("main.Config.from_env")
    @patch("main.FinancialReportOrchestrator")
    def test_review_issues_displayed(self, mock_orch_class, mock_config):
        """Test that review issues are displayed when present."""
        mock_config.return_value = Mock()
        mock_orchestrator = Mock()
        mock_orch_class.return_value = mock_orchestrator

        mock_result = Mock(spec=ReportMetadata)
        mock_result.final_markdown_path = Path("/tmp/report.md")
        mock_result.final_html_path = None
        mock_result.performance_metrics = {
            "execution_time_seconds": 10.5,
            "successful": 1,
            "failed": 0,
            "charts_generated": 3,
            "quality_score": 7,
        }
        mock_result.portfolio_metrics = None
        mock_result.review_issues = ["Issue 1", "Issue 2"]
        mock_orchestrator.run.return_value = mock_result

        with patch.object(sys, "argv", ["main.py"]):
            result = main()

        assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
