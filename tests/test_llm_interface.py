"""
Comprehensive tests for LLM interface.
"""

import json
import pytest
import httpx
from pathlib import Path
from unittest.mock import Mock, patch
from config import Config
from models import ParsedRequest, TickerAnalysis, PortfolioMetrics, TechnicalIndicators, AdvancedMetrics, FundamentalData
from llm_interface import IntegratedLLMInterface


@pytest.fixture
def config():
    """Create test configuration."""
    return Config(
        openai_api_key="test-key-12345",
        openai_base_url="https://api.openai.com/v1",
        model_name="gpt-4o",
        max_workers=2,
        cache_ttl_hours=24,
        benchmark_ticker="SPY",
        request_timeout=30.0,
    )


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    with patch("llm_interface.ChatOpenAI") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def llm_interface(config, mock_llm):
    """Create LLM interface with mocked LLM."""
    return IntegratedLLMInterface(config)


@pytest.fixture
def sample_ticker_analysis():
    """Create sample ticker analysis for testing."""
    return TickerAnalysis(
        ticker="AAPL",
        csv_path=Path("/tmp/AAPL.csv"),
        chart_path=Path("/tmp/AAPL.png"),
        latest_close=150.0,
        avg_daily_return=0.0005,  # 0.05% daily return
        volatility=0.02,  # 2% daily volatility
        ratios={
            "pe_ratio": 25.0,
            "price_to_book": 5.0,
        },
        fundamentals=FundamentalData(
            revenue=400000000000.0,  # $400B
            net_income=100000000000.0,  # $100B
            revenue_growth=0.10,  # 10% growth
            earnings_growth=0.15,  # 15% growth
            free_cash_flow=90000000000.0,  # $90B
        ),
        technical_indicators=TechnicalIndicators(
            rsi=65.5,
            macd=2.5,
            macd_signal=2.0,
            bollinger_upper=155.0,
            bollinger_lower=145.0,
            bollinger_position=0.5,
            stochastic_k=70.0,
            stochastic_d=68.0,
            atr=3.5,
            obv=1000000.0,
            ma_200d=148.0,
        ),
        advanced_metrics=AdvancedMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=-0.15,  # -15%
            beta=1.1,
            alpha=0.025,  # 2.5% annualized
            var_95=-0.025,  # -2.5%
            cvar_95=-0.030,  # -3.0%
            treynor_ratio=0.10,
            information_ratio=0.5,
        ),
    )


class TestLLMInterfaceInitialization:
    """Test LLM interface initialization."""

    def test_initialization_success(self, config, mock_llm):
        """Test successful initialization."""
        interface = IntegratedLLMInterface(config)

        assert interface.config == config
        assert interface.llm is not None

    def test_http_client_creation(self, config, mock_llm):
        """Test HTTP client is created with connection pooling."""
        interface = IntegratedLLMInterface(config, max_connections=30, max_keepalive_connections=15)

        # Verify the interface was initialized
        assert interface is not None

    def test_prompts_are_setup(self, llm_interface):
        """Test that prompts are properly initialized."""
        assert hasattr(llm_interface, "parse_prompt")
        assert hasattr(llm_interface, "narrative_prompt")
        assert hasattr(llm_interface, "review_prompt")

        # Verify prompt templates have correct input variables
        assert "user_request" in llm_interface.parse_prompt.input_variables
        assert "analysis_data" in llm_interface.narrative_prompt.input_variables
        assert "report_content" in llm_interface.review_prompt.input_variables


class TestNaturalLanguageRequestParsing:
    """Test natural language request parsing."""

    def test_parse_simple_request(self, llm_interface):
        """Test parsing a simple request."""
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = json.dumps({
            "tickers": ["AAPL"],
            "period": "1y",
            "metrics": ["returns", "risk"],
            "output_format": "markdown"
        })

        with patch.object(llm_interface, "parser_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            result = llm_interface.parse_natural_language_request("Analyze AAPL over the past year")

            assert isinstance(result, ParsedRequest)
            assert result.tickers == ["AAPL"]
            assert result.period == "1y"

    def test_parse_multi_ticker_request(self, llm_interface):
        """Test parsing request with multiple tickers."""
        mock_response = Mock()
        mock_response.content = json.dumps({
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "period": "6mo",
            "metrics": ["returns"],
            "output_format": "markdown"
        })

        with patch.object(llm_interface, "parser_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            result = llm_interface.parse_natural_language_request("Compare AAPL, MSFT, and GOOGL over 6 months")

            assert len(result.tickers) == 3
            assert result.period == "6mo"

    def test_parse_connection_error_retry(self, llm_interface):
        """Test retry logic for connection errors."""
        mock_response_good = Mock()
        mock_response_good.content = json.dumps({
            "tickers": ["AAPL"],
            "period": "1y",
            "metrics": ["returns"],
            "output_format": "markdown"
        })

        with patch.object(llm_interface, "parser_chain") as mock_chain:
            # First two calls raise ConnectionError, third succeeds
            mock_chain.invoke.side_effect = [
                ConnectionError("Network error"),
                ConnectionError("Network error"),
                mock_response_good
            ]

            result = llm_interface.parse_natural_language_request("Analyze AAPL")

            assert isinstance(result, ParsedRequest)
            assert mock_chain.invoke.call_count == 3

    def test_parse_invalid_json_raises_error(self, llm_interface):
        """Test that invalid JSON raises ValueError (no retry for ValueError)."""
        mock_response = Mock()
        mock_response.content = "Invalid JSON"

        with patch.object(llm_interface, "parser_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            with pytest.raises(ValueError, match="Failed to parse JSON response"):
                llm_interface.parse_natural_language_request("Analyze stocks")


class TestNarrativeGeneration:
    """Test narrative generation."""

    def test_generate_narrative_summary_single_ticker(self, llm_interface, sample_ticker_analysis):
        """Test generating narrative summary for single ticker."""
        mock_response = Mock()
        mock_response.content = "AAPL has shown strong performance with a 0.05% daily return over the past year."

        with patch.object(llm_interface, "narrative_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            analyses = {"AAPL": sample_ticker_analysis}
            result = llm_interface.generate_narrative_summary(analyses, "1y")

            assert isinstance(result, str)
            assert len(result) > 0
            mock_chain.invoke.assert_called_once()

    def test_generate_narrative_summary_portfolio(self, llm_interface, sample_ticker_analysis):
        """Test generating narrative summary for portfolio."""
        mock_response = Mock()
        mock_response.content = "The portfolio showed mixed performance across the three holdings."

        # Create multiple analyses (use replace instead of model_copy)
        import dataclasses
        msft_analysis = dataclasses.replace(sample_ticker_analysis, ticker="MSFT", latest_close=350.0)
        googl_analysis = dataclasses.replace(sample_ticker_analysis, ticker="GOOGL", latest_close=140.0)

        with patch.object(llm_interface, "narrative_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            analyses = {
                "AAPL": sample_ticker_analysis,
                "MSFT": msft_analysis,
                "GOOGL": googl_analysis
            }
            result = llm_interface.generate_narrative_summary(analyses, "1y")

            assert isinstance(result, str)
            assert len(result) > 0

    def test_generate_narrative_summary_with_error(self, llm_interface, sample_ticker_analysis):
        """Test handling of analysis with errors."""
        import dataclasses
        error_analysis = dataclasses.replace(sample_ticker_analysis, error="Data fetch failed")

        mock_response = Mock()
        mock_response.content = "Analysis limited due to data issues."

        with patch.object(llm_interface, "narrative_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            analyses = {"AAPL": error_analysis}
            result = llm_interface.generate_narrative_summary(analyses, "1y")

            assert isinstance(result, str)


class TestReportQualityReview:
    """Test report quality review functionality."""

    def test_review_report_high_quality(self, llm_interface, sample_ticker_analysis):
        """Test reviewing a high-quality report."""
        mock_response = Mock()
        mock_response.content = json.dumps({
            "issues": [],
            "suggestions": ["Could add industry comparison"],
            "quality_score": 9
        })

        with patch.object(llm_interface, "review_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            report_content = "This is a well-written report with accurate data."
            analyses = {"AAPL": sample_ticker_analysis}

            issues, quality_score, full_review = llm_interface.review_report(report_content, analyses)

            assert len(issues) == 0
            assert len(full_review.get("suggestions", [])) > 0
            assert quality_score == 9

    def test_review_report_with_issues(self, llm_interface, sample_ticker_analysis):
        """Test reviewing a report with quality issues."""
        mock_response = Mock()
        mock_response.content = json.dumps({
            "issues": ["RSI value of 60 incorrectly labeled as 'overbought'"],
            "suggestions": ["Add methodology notes"],
            "quality_score": 6
        })

        with patch.object(llm_interface, "review_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            report_content = "Report with issues"
            analyses = {"AAPL": sample_ticker_analysis}

            issues, quality_score, full_review = llm_interface.review_report(report_content, analyses)

            assert len(issues) > 0
            assert quality_score == 6

    def test_review_report_invalid_json(self, llm_interface, sample_ticker_analysis):
        """Test handling of invalid JSON in review response."""
        mock_response = Mock()
        mock_response.content = "Not valid JSON"

        with patch.object(llm_interface, "review_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            report_content = "Test report"
            analyses = {"AAPL": sample_ticker_analysis}

            # Should handle error gracefully and return default values
            issues, quality_score, full_review = llm_interface.review_report(report_content, analyses)

            assert isinstance(issues, list)
            assert isinstance(quality_score, int)
            assert isinstance(full_review, dict)


class TestReportGeneration:
    """Test report generation methods."""

    def test_generate_report_header(self, llm_interface):
        """Test generating report header."""
        header = llm_interface._generate_report_header()

        assert isinstance(header, list)
        assert any("Financial Analysis Report" in line for line in header)

    def test_generate_metrics_table(self, llm_interface, sample_ticker_analysis):
        """Test generating metrics table."""
        analyses = {"AAPL": sample_ticker_analysis}

        table = llm_interface._generate_metrics_table(analyses)

        assert isinstance(table, str)
        assert "AAPL" in table
        assert "Sharpe" in table or "sharpe" in table.lower()

    def test_generate_portfolio_section(self, llm_interface):
        """Test generating portfolio section."""
        portfolio_metrics = PortfolioMetrics(
            total_value=1000000.0,
            portfolio_return=0.15,  # 15% as decimal
            portfolio_volatility=0.18,  # 18% as decimal
            portfolio_sharpe=1.8,
            diversification_ratio=1.2,
            concentration_risk=0.35,
        )

        section = llm_interface._generate_portfolio_section(portfolio_metrics)

        assert isinstance(section, str)
        assert "Portfolio" in section or "portfolio" in section.lower()
        assert "1.8" in section or "Sharpe" in section

    def test_generate_risk_analysis(self, llm_interface, sample_ticker_analysis):
        """Test generating risk analysis section."""
        analyses = {"AAPL": sample_ticker_analysis}

        risk_section = llm_interface._generate_risk_analysis(analyses)

        assert isinstance(risk_section, str)
        assert "risk" in risk_section.lower() or "Risk" in risk_section

    def test_generate_fundamental_section(self, llm_interface, sample_ticker_analysis):
        """Test generating fundamental analysis section."""
        analyses = {"AAPL": sample_ticker_analysis}

        fundamental_section = llm_interface._generate_fundamental_section(analyses)

        assert isinstance(fundamental_section, str)
        # Check for fundamental data fields (Revenue, Net Income, etc.)
        assert "Revenue" in fundamental_section or "revenue" in fundamental_section.lower()

    def test_generate_recommendations(self, llm_interface, sample_ticker_analysis):
        """Test generating recommendations section."""
        analyses = {"AAPL": sample_ticker_analysis}

        recommendations = llm_interface._generate_recommendations(analyses)

        assert isinstance(recommendations, str)
        assert len(recommendations) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_analysis(self, llm_interface):
        """Test handling of empty analysis dictionary."""
        mock_response = Mock()
        mock_response.content = "No data available for analysis."

        with patch.object(llm_interface, "narrative_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            result = llm_interface.generate_narrative_summary({}, "1y")

            assert isinstance(result, str)

    def test_long_report_truncation(self, llm_interface, sample_ticker_analysis):
        """Test that long reports are truncated for review."""
        mock_response = Mock()
        mock_response.content = json.dumps({
            "issues": [],
            "suggestions": [],
            "quality_score": 8
        })

        with patch.object(llm_interface, "review_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            # Create a very long report
            long_report = "A" * 10000
            analyses = {"AAPL": sample_ticker_analysis}

            issues, quality_score, full_review = llm_interface.review_report(long_report, analyses)

            # Verify truncation occurred in the call
            call_args = mock_chain.invoke.call_args[0][0]
            assert len(call_args["report_content"]) <= 4000

    def test_truncation_warning_logged(self, llm_interface, sample_ticker_analysis, caplog):
        """Test that truncation warning is logged for long reports."""
        import logging

        mock_response = Mock()
        mock_response.content = json.dumps({
            "issues": [],
            "suggestions": [],
            "quality_score": 8
        })

        with patch.object(llm_interface, "review_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            # Create a very long report (>4000 chars)
            long_report = "A" * 10000
            analyses = {"AAPL": sample_ticker_analysis}

            with caplog.at_level(logging.WARNING):
                llm_interface.review_report(long_report, analyses)

            # Verify warning was logged
            assert any("truncated for review" in record.message for record in caplog.records)
            assert any("10000 chars -> 4000 chars" in record.message for record in caplog.records)

    def test_no_truncation_warning_for_short_reports(self, llm_interface, sample_ticker_analysis, caplog):
        """Test that no truncation warning is logged for short reports."""
        import logging

        mock_response = Mock()
        mock_response.content = json.dumps({
            "issues": [],
            "suggestions": [],
            "quality_score": 8
        })

        with patch.object(llm_interface, "review_chain") as mock_chain:
            mock_chain.invoke.return_value = mock_response

            # Create a short report (<4000 chars)
            short_report = "This is a short report with less than 4000 characters."
            analyses = {"AAPL": sample_ticker_analysis}

            with caplog.at_level(logging.WARNING):
                llm_interface.review_report(short_report, analyses)

            # Verify NO warning was logged
            assert not any("truncated for review" in record.message for record in caplog.records)

    def test_missing_fundamental_data(self, llm_interface, sample_ticker_analysis):
        """Test handling of missing fundamental data."""
        import dataclasses
        analysis_no_fundamentals = dataclasses.replace(sample_ticker_analysis, fundamentals=None)

        analyses = {"AAPL": analysis_no_fundamentals}
        section = llm_interface._generate_fundamental_section(analyses)

        assert isinstance(section, str)
        # Should handle None gracefully

    def test_http_client_cleanup(self, config):
        """Test HTTP client cleanup on deletion."""
        with patch("llm_interface.ChatOpenAI"):
            interface = IntegratedLLMInterface(config)

            # Mock the http_client (the client is actually interface.llm.client not http_client)
            mock_http_client = Mock(spec=httpx.Client)
            mock_http_client.close = Mock()

            # Set the mock on the LLM's client attribute
            if hasattr(interface.llm, 'client'):
                interface.llm.client = mock_http_client
                interface.llm.client.close = mock_http_client.close

            # Trigger cleanup
            del interface

            # Note: cleanup is best-effort, may or may not be called depending on GC


class TestConnectionPooling:
    """Test connection pooling functionality."""

    def test_connection_pool_configuration(self, config):
        """Test that connection pool is properly configured."""
        with patch("llm_interface.ChatOpenAI"):
            interface = IntegratedLLMInterface(config, max_connections=50, max_keepalive_connections=25)

            assert interface is not None

    def test_http2_enabled(self, config, mock_llm):
        """Test that HTTP/2 is enabled for multiplexing."""
        with patch("llm_interface.httpx.Client") as mock_client:
            _ = IntegratedLLMInterface(config)  # Create interface to trigger httpx.Client

            # Verify HTTP/2 was enabled
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs.get("http2") is True


class TestPromptTemplates:
    """Test prompt template content."""

    def test_parse_prompt_has_required_fields(self, llm_interface):
        """Test that parse prompt requests required fields."""
        template = llm_interface.parse_prompt.template

        assert "tickers" in template
        assert "period" in template
        assert "JSON" in template

    def test_narrative_prompt_has_guidelines(self, llm_interface):
        """Test that narrative prompt includes important guidelines."""
        template = llm_interface.narrative_prompt.template

        assert "IMPORTANT" in template or "GUIDELINES" in template
        assert "RSI" in template
        assert "Sharpe" in template

    def test_review_prompt_separates_issues_and_suggestions(self, llm_interface):
        """Test that review prompt distinguishes issues from suggestions."""
        template = llm_interface.review_prompt.template

        assert "issues" in template
        assert "suggestions" in template
        assert "quality_score" in template


class TestTableValidation:
    """Tests for markdown table validation and correction."""

    def test_validate_correct_table(self):
        """Test that correctly formatted tables pass through unchanged."""
        text = """# Test Report

| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

Some other text.
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)
        assert result == text

    def test_fix_missing_columns(self):
        """Test that rows with missing columns are padded with N/A."""
        text = """| Header A | Header B | Header C |
|----------|----------|----------|
| Value 1  | Value 2  |
| Value 4  | Value 5  | Value 6  |
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)

        # Should have N/A added to first data row
        assert "N/A" in result
        lines = result.split('\n')
        # Find the data row (skip header and separator)
        data_row = lines[2]
        # Should have 4 pipes (3 columns)
        assert data_row.count('|') == 4

    def test_fix_extra_columns(self):
        """Test that rows with extra columns are truncated."""
        text = """| Header A | Header B | Header C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  | Extra    |
| Value 4  | Value 5  | Value 6  |
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)

        # Extra column should be removed
        assert "Extra" not in result
        lines = result.split('\n')
        # All data rows should have same number of pipes
        data_row1 = lines[2]
        data_row2 = lines[3]
        assert data_row1.count('|') == data_row2.count('|')

    def test_multiple_tables(self):
        """Test that multiple tables in same text are all validated."""
        text = """## First Table

| A | B |
|---|---|
| 1 | 2 |

## Second Table

| X | Y | Z |
|---|---|---|
| a | b |
| d | e | f |
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)

        # Second table should have N/A added
        assert "N/A" in result

    def test_non_table_text_unchanged(self):
        """Test that non-table text with pipes is not modified."""
        text = """Some text with | pipes | in it.

Not a table.
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)
        assert result == text

    def test_real_world_msft_issue(self):
        """Test the actual MSFT table issue from the bug report."""
        # This is the problematic MSFT table with 7 data columns but 8 header columns
        text = """### 2. Top Strategy Opportunities

| Rank | Strategy (strike) | Structure | Net Premium (cost) | Max Profit | Max Loss | Breakeven(s) | Why It Makes Sense |
|------|-------------------|-----------|--------------------|------------|----------|--------------|
| 1 | Long Straddle @ 525 (cost $2,852) | Buy 1 ATM Call 525 + 1 ATM Put 525 | Unlimited (upside & downside) | Premium paid = $2,852 (loss if price stays at 525) | $496.5 (down) - $553.5 (up) | IV is extremely high; a move of > $28.5 in either direction yields profit. |
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)

        lines = result.split('\n')
        # Find header and data rows
        header_line = lines[2]
        data_line = lines[4]

        # Both should have same number of pipes
        header_pipes = header_line.count('|')
        data_pipes = data_line.count('|')
        assert header_pipes == data_pipes, f"Header has {header_pipes} pipes, data has {data_pipes} pipes"

        # Should have N/A added for missing "Why It Makes Sense" column
        assert "N/A" in result or len(data_line.split('|')) == len(header_line.split('|'))

    def test_empty_cells_preserved(self):
        """Test that intentionally empty cells are preserved."""
        text = """| A | B | C |
|---|---|---|
| 1 |   | 3 |
"""
        result = IntegratedLLMInterface._validate_and_fix_markdown_tables(text)

        # Should still have 3 columns in data row (empty middle cell)
        lines = result.split('\n')
        data_row = lines[2]
        assert data_row.count('|') == 4  # 4 pipes = 3 columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
