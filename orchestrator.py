"""
Main orchestrator coordinating all components.
"""

import time
import logging
import pandas as pd
from pathlib import Path
from config import Config
from cache import CacheManager
from alerts import AlertSystem
from utils import ProgressTracker, validate_ticker_symbol
from fetcher import CachedDataFetcher
from datetime import datetime, timezone
from charts import ThreadSafeChartGenerator
from llm_interface import IntegratedLLMInterface
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from analyzers import AdvancedFinancialAnalyzer, PortfolioAnalyzer
from models import (
    TickerAnalysis,
    TechnicalIndicators,
    AdvancedMetrics,
    ReportMetadata,
    PortfolioMetrics,
    PortfolioRequest,
)

logger = logging.getLogger(__name__)


class FinancialReportOrchestrator:
    """Main orchestrator with all features integrated."""

    def __init__(self, config: Config) -> None:
        """
        Initializes the FinancialReportOrchestrator with a configuration object.

        Args:
            config (Config): The configuration object.
        """
        self.config = config
        self.cache = CacheManager(
            cache_dir="./.cache", ttl_hours=config.cache_ttl_hours
        )
        self.fetcher = CachedDataFetcher(self.cache, timeout=config.request_timeout)
        self.analyzer = AdvancedFinancialAnalyzer(
            risk_free_rate=config.risk_free_rate,
            benchmark_ticker=config.benchmark_ticker,
        )
        self.portfolio_analyzer = PortfolioAnalyzer(
            risk_free_rate=config.risk_free_rate
        )
        self.chart_gen = ThreadSafeChartGenerator()
        self.alert_system = AlertSystem()
        self.llm = IntegratedLLMInterface(config)

    def _validate_output_path(self, output_dir: str) -> Path:
        """Validate and sanitize output directory path to prevent path traversal."""
        output_path = Path(output_dir).resolve()

        # Ensure output directory is within current working directory or safe locations
        cwd = Path.cwd().resolve()
        try:
            # Check if output_dir is relative to cwd
            output_path.relative_to(cwd)
            return output_path
        except ValueError:
            # If not relative to cwd, check if it's within home directory
            home_dir = Path.home().resolve()
            try:
                output_path.relative_to(home_dir)
                return output_path
            except ValueError:
                raise ValueError(
                    f"Output directory must be within current working directory or home directory. "
                    f"Got: {output_dir} (resolved to: {output_path})"
                )

    def _create_error_analysis(self, ticker: str, error_msg: str) -> TickerAnalysis:
        """
        Creates a TickerAnalysis object representing an error state.

        Args:
            ticker: The ticker symbol
            error_msg: The error message

        Returns:
            TickerAnalysis with error set and default values
        """
        import gc
        gc.collect()  # Force cleanup of any partial results to free memory

        return TickerAnalysis(
            ticker=ticker,
            csv_path=Path(),
            chart_path=Path(),
            latest_close=0.0,
            avg_daily_return=0.0,
            volatility=0.0,
            ratios={},
            fundamentals=None,
            advanced_metrics=AdvancedMetrics(),
            technical_indicators=TechnicalIndicators(),
            sample_data=[],
            error=error_msg,
        )

    def analyze_ticker(
        self,
        ticker: str,
        period: str,
        output_dir: Path,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> TickerAnalysis:
        """
        Performs a comprehensive analysis of a single ticker.

        Args:
            ticker (str): The ticker symbol to analyze (will be validated and normalized).
            period (str): The analysis period.
            output_dir (Path): The directory to save the output files to.
            benchmark_returns (Optional[pd.Series]): The benchmark returns for comparison.

        Returns:
            TickerAnalysis: An object containing the analysis results.

        Raises:
            ValueError: If the ticker symbol is invalid.
        """
        # Validate and normalize ticker symbol using centralized validation
        ticker = validate_ticker_symbol(ticker)
        logger.info(f"Analyzing {ticker}")

        try:
            # Fetch price data (cached)
            raw_price_history = self.fetcher.fetch_price_history(ticker, period)

            # Compute metrics
            price_data_with_indicators = self.analyzer.compute_metrics(
                raw_price_history
            )

            # Save CSV
            csv_path = output_dir / f"{ticker}_prices.csv"
            price_data_with_indicators.to_csv(csv_path, index=False)

            # Create chart
            chart_path = output_dir / f"{ticker}_technical.png"
            self.chart_gen.create_price_chart(
                price_data_with_indicators, ticker, chart_path
            )

            # Calculate advanced metrics
            returns = price_data_with_indicators["daily_return"].dropna()
            advanced_metrics = self.analyzer.calculate_advanced_metrics(
                returns, benchmark_returns
            )

            # Create comparative analysis against benchmark
            comparative_analysis = self.analyzer.create_comparative_analysis(
                returns, benchmark_returns
            )

            # Extract last row once to avoid repeated iloc[-1] lookups
            last_row = price_data_with_indicators.iloc[-1]

            # Helper function to safely extract float from last row
            def safe_float(value: Any) -> Optional[float]:
                return float(value) if not pd.isna(value) else None

            # Get technical indicators from last row
            latest_technical_indicators = TechnicalIndicators(
                rsi=safe_float(last_row["rsi"]),
                macd=safe_float(last_row["macd"]),
                macd_signal=safe_float(last_row["macd_signal"]),
                bollinger_upper=safe_float(last_row["bollinger_upper"]),
                bollinger_lower=safe_float(last_row["bollinger_lower"]),
                bollinger_position=safe_float(last_row["bollinger_position"]),
            )

            # Compute financial ratios and parse fundamentals
            ratios = self.analyzer.compute_ratios(ticker)
            fundamentals = self.analyzer.parse_fundamentals(ticker)

            # Extract key metrics with validation from last row
            latest_close = safe_float(last_row["close"]) or 0.0
            volatility = safe_float(last_row["volatility"]) or 0.0

            # Mean calculation doesn't need last row
            mean_return = price_data_with_indicators["daily_return"].mean()
            avg_return = float(mean_return) if not pd.isna(mean_return) else 0.0

            analysis = TickerAnalysis(
                ticker=ticker,
                csv_path=csv_path,
                chart_path=chart_path,
                latest_close=latest_close,
                avg_daily_return=avg_return,
                volatility=volatility,
                ratios=ratios,
                fundamentals=fundamentals,
                advanced_metrics=advanced_metrics,
                technical_indicators=latest_technical_indicators,
                comparative_analysis=comparative_analysis,
                sample_data=price_data_with_indicators.tail(3).to_dict(
                    orient="records"
                ),
            )

            # Check alerts
            analysis.alerts = self.alert_system.check_alerts(analysis)

            return analysis

        except Exception as e:
            # Use match/case for clean error categorization
            match e:
                case ValueError():
                    # Invalid ticker or validation error - permanent failure, don't retry
                    error_msg = f"Invalid ticker {ticker}: {e}"
                    logger.error(f"VALIDATION ERROR - {error_msg}")
                    print(f"  [!] {ticker}: Validation Error - {str(e)[:80]}")

                case OSError() | ConnectionError() | TimeoutError():
                    # Network error - temporary failure, could be retried
                    error_msg = f"Network error for {ticker}: {e}"
                    logger.warning(f"NETWORK ERROR (retryable) - {error_msg}")
                    print(f"  [!] {ticker}: Network Error (temporary) - {str(e)[:60]}")

                case KeyError() | pd.errors.ParserError():
                    # Data parsing error - permanent failure, data structure issue
                    error_msg = f"Data parsing error for {ticker}: {e}"
                    logger.error(f"PARSING ERROR - {error_msg}")
                    print(f"  [!] {ticker}: Data Format Error - {str(e)[:70]}")

                case TypeError():
                    # Type error - likely a code/logic issue, permanent failure
                    error_msg = f"Type error analyzing {ticker}: {e}"
                    logger.error(f"TYPE ERROR (code issue) - {error_msg}")
                    print(f"  [!] {ticker}: Type Error (internal) - {str(e)[:70]}")

                case _:
                    # Catch-all for unexpected errors - log with full context
                    error_msg = f"Unexpected error analyzing {ticker}: {type(e).__name__}: {e}"
                    logger.exception(f"UNEXPECTED ERROR - {error_msg}")  # .exception() logs stack trace
                    print(f"  [!] {ticker}: Unexpected Error - {type(e).__name__}")

            return self._create_error_analysis(ticker, error_msg)

    def run_from_natural_language(
        self, user_request: str, output_dir: str = "./reports"
    ) -> ReportMetadata:
        """
        Runs the financial report generation from a natural language request.

        Args:
            user_request (str): The natural language request from the user.
            output_dir (str): The directory to save the output files to.

        Returns:
            ReportMetadata: An object containing the report metadata.
        """
        try:
            # Parse natural language into structured request
            parsed = self.llm.parse_natural_language_request(user_request)

            # Create validated Pydantic model
            portfolio_request = PortfolioRequest(
                tickers=parsed.tickers,
                period=parsed.period,
                weights=None  # Natural language requests don't specify weights
            )

            return self.run(portfolio_request, output_dir)
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse request: {e}")
            raise

    def _fetch_benchmark(
        self, period: str, output_path: Path
    ) -> Tuple[Optional[TickerAnalysis], Optional[pd.Series]]:
        """Fetch benchmark data for comparative analysis.

        Args:
            period: Analysis period
            output_path: Output directory path

        Returns:
            Tuple of (benchmark_analysis, benchmark_returns)
        """
        print(f"\n📊 Fetching benchmark ({self.config.benchmark_ticker})...")
        benchmark_analysis = None
        benchmark_returns = None

        try:
            benchmark_analysis = self.analyze_ticker(
                self.config.benchmark_ticker, period, output_path
            )
            if not benchmark_analysis.error:
                benchmark_df = pd.read_csv(benchmark_analysis.csv_path)
                benchmark_returns = benchmark_df["daily_return"]
                print("✓ Benchmark loaded\n")
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as e:
            print(f"⚠️  Benchmark unavailable: {e}\n")

        return benchmark_analysis, benchmark_returns

    def _analyze_all_tickers(
        self,
        tickers: List[str],
        period: str,
        output_path: Path,
        benchmark_returns: Optional[pd.Series],
    ) -> Dict[str, TickerAnalysis]:
        """Analyze all tickers concurrently with progress tracking.

        Uses ThreadPoolExecutor to analyze multiple tickers in parallel,
        limited by config.max_workers.

        Args:
            tickers: List of ticker symbols
            period: Analysis period
            output_path: Output directory path
            benchmark_returns: Benchmark returns for comparison

        Returns:
            Dictionary mapping tickers to analysis results
        """
        print(
            f"🔍 Analyzing {len(tickers)} tickers (concurrent workers: {self.config.max_workers})..."
        )
        progress = ProgressTracker(len(tickers), "Analysis")

        analyses = {}

        # Use ThreadPoolExecutor for concurrent analysis
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all ticker analysis tasks
            future_to_ticker = {
                executor.submit(
                    self.analyze_ticker, ticker, period, output_path, benchmark_returns
                ): ticker
                for ticker in tickers
            }

            # Process completed futures as they finish
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    analysis = future.result()
                    analyses[ticker] = analysis
                    progress.update(ticker, not analysis.error)
                except Exception as e:
                    # Handle unexpected exceptions from worker threads
                    logger.error(f"Unexpected error analyzing {ticker}: {e}")
                    error_analysis = TickerAnalysis(
                        ticker=ticker,
                        csv_path=Path(),
                        chart_path=Path(),
                        latest_close=0.0,
                        avg_daily_return=0.0,
                        volatility=0.0,
                        ratios={},
                        fundamentals=None,
                        advanced_metrics=AdvancedMetrics(),
                        technical_indicators=TechnicalIndicators(),
                        sample_data=[],
                        error=f"Thread error: {str(e)}",
                    )
                    analyses[ticker] = error_analysis
                    progress.update(ticker, False)

        progress.complete()
        return analyses

    def _process_analysis_results(
        self, analyses: Dict[str, TickerAnalysis]
    ) -> Tuple[Dict[str, TickerAnalysis], Dict[str, TickerAnalysis]]:
        """Segregate successful and failed analyses.

        Args:
            analyses: All analysis results

        Returns:
            Tuple of (successful_analyses, failed_analyses)

        Raises:
            ValueError: If all analyses failed
        """
        successful = {
            ticker: analysis
            for ticker, analysis in analyses.items()
            if not analysis.error
        }
        failed = {
            ticker: analysis for ticker, analysis in analyses.items() if analysis.error
        }

        if not successful:
            raise ValueError("All analyses failed")

        # Print results summary
        print(f"\n✓ Analyzed: {len(successful)}/{len(analyses)}")
        if failed:
            print(f"⚠️  Failed ({len(failed)}):")
            for ticker, analysis in failed.items():
                # Show first 80 chars of error message
                error_preview = (
                    analysis.error[:80] + "..."
                    if len(analysis.error) > 80
                    else analysis.error
                )
                print(f"    • {ticker}: {error_preview}")

        return successful, failed

    def _calculate_portfolio_metrics_if_needed(
        self,
        successful: Dict[str, TickerAnalysis],
        portfolio_weights: Optional[Dict[str, float]],
    ) -> Optional[PortfolioMetrics]:
        """Calculate portfolio-level metrics if multiple tickers.

        Args:
            successful: Successfully analyzed tickers
            portfolio_weights: Optional portfolio weights

        Returns:
            Portfolio metrics or None
        """
        if len(successful) < 2:
            return None

        try:
            print("\n💼 Calculating portfolio metrics...")
            portfolio_metrics = self.portfolio_analyzer.calculate_portfolio_metrics(
                successful, portfolio_weights
            )
            print("✓ Portfolio analysis complete")
            return portfolio_metrics
        except (ValueError, KeyError, TypeError, OSError, pd.errors.ParserError) as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return None

    def _generate_comparison_charts(
        self, successful: Dict[str, TickerAnalysis], output_path: Path
    ) -> List[Path]:
        """Generate comparison and visualization charts.

        Args:
            successful: Successfully analyzed tickers
            output_path: Output directory path

        Returns:
            List of chart file paths
        """
        print("\n📊 Generating visualizations...")
        chart_files = [a.chart_path for a in successful.values() if a.chart_path]

        if len(successful) >= 2:
            try:
                comparison_path = output_path / "comparison_chart.png"
                self.chart_gen.create_comparison_chart(successful, comparison_path)
                chart_files.append(comparison_path)
                print("✓ Comparison chart created")
            except (ValueError, KeyError, OSError, pd.errors.ParserError) as e:
                logger.error(f"Comparison chart failed: {e}")

        return chart_files

    def _generate_and_review_report(
        self,
        analyses: Dict[str, TickerAnalysis],
        benchmark_analysis: Optional[TickerAnalysis],
        portfolio_metrics: Optional[PortfolioMetrics],
        period: str,
        output_path: Path,
    ) -> Tuple[str, List[str], int]:
        """Generate report with LLM and review quality.

        Args:
            analyses: All analysis results
            benchmark_analysis: Benchmark analysis
            portfolio_metrics: Portfolio metrics
            period: Analysis period
            output_path: Output directory path

        Returns:
            Tuple of (report_content, review_issues, quality_score)
        """
        print("\n📝 Generating report with AI insights...")
        report_content = self.llm.generate_detailed_report(
            analyses, benchmark_analysis, portfolio_metrics, period
        )

        print("🔍 Reviewing report quality...")
        review_issues, quality_score, full_review = self.llm.review_report(
            report_content, analyses
        )

        # Save full review to file
        from datetime import datetime, timezone
        import json

        review_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        review_path = output_path / f"quality_review_{review_timestamp}.json"

        with review_path.open("w", encoding="utf-8") as f:
            json.dump(full_review, f, indent=2)

        logger.info(f"Review saved: {review_path}")

        if review_issues:
            print(f"⚠️  Found {len(review_issues)} issues (Quality: {quality_score}/10)")
            suggestions_count = len(full_review.get("suggestions", []))
            if suggestions_count > 0:
                print(
                    f"💡 {suggestions_count} suggestions available in {review_path.name}"
                )
        else:
            print(f"✓ Report quality: {quality_score}/10")

        return report_content, review_issues, quality_score

    def _save_report(self, report_content: str, output_path: Path) -> Tuple[Path, str]:
        """Save report to file with timestamp.

        Args:
            report_content: Report markdown content
            output_path: Output directory path

        Returns:
            Tuple of (report_path, timestamp)
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_filename = f"financial_report_{timestamp}.md"
        report_path = output_path / report_filename

        with report_path.open("w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report saved: {report_path}")
        return report_path, timestamp

    def _collect_performance_metrics(
        self,
        start_time: float,
        tickers: List[str],
        successful: Dict[str, TickerAnalysis],
        failed: Dict[str, TickerAnalysis],
        chart_files: List[Path],
        quality_score: int,
        portfolio_metrics: Optional[PortfolioMetrics],
    ) -> Dict[str, Any]:
        """Collect execution performance metrics.

        Args:
            start_time: Execution start time
            tickers: All tickers analyzed
            successful: Successful analyses
            failed: Failed analyses
            chart_files: Generated chart files
            quality_score: Report quality score
            portfolio_metrics: Portfolio metrics

        Returns:
            Performance metrics dictionary
        """
        execution_time = time.time() - start_time
        return {
            "execution_time_seconds": round(execution_time, 2),
            "tickers_analyzed": len(tickers),
            "successful": len(successful),
            "failed": len(failed),
            "charts_generated": len(chart_files),
            "quality_score": quality_score,
            "portfolio_analyzed": portfolio_metrics is not None,
        }

    def run(
        self,
        request: PortfolioRequest,
        output_dir: str = "./reports",
    ) -> ReportMetadata:
        """
        Runs the financial report generation with validated request.

        Args:
            request (PortfolioRequest): Validated portfolio analysis request containing
                tickers, period, and optional weights.
            output_dir (str): The directory to save the output files to.

        Returns:
            ReportMetadata: An object containing the report metadata.

        Raises:
            ValueError: If request validation fails or all analyses fail.
        """
        start_time = time.time()

        # Extract validated data from request
        tickers = request.tickers
        period = request.period
        portfolio_weights = request.weights

        # Setup
        output_path = self._validate_output_path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.cache.clear_expired()

        logger.info("=" * 60)
        logger.info("Starting Advanced Financial Report Generation")
        logger.info("=" * 60)

        # Fetch benchmark
        benchmark_analysis, benchmark_returns = self._fetch_benchmark(
            period, output_path
        )

        # Analyze tickers
        analyses = self._analyze_all_tickers(
            tickers, period, output_path, benchmark_returns
        )
        successful, failed = self._process_analysis_results(analyses)

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics_if_needed(
            successful, portfolio_weights
        )

        # Generate charts
        chart_files = self._generate_comparison_charts(successful, output_path)

        # Generate and review report
        report_content, review_issues, quality_score = self._generate_and_review_report(
            analyses, benchmark_analysis, portfolio_metrics, period, output_path
        )

        # Save report
        report_path, timestamp = self._save_report(report_content, output_path)

        # Collect performance metrics
        performance_metrics = self._collect_performance_metrics(
            start_time,
            tickers,
            successful,
            failed,
            chart_files,
            quality_score,
            portfolio_metrics,
        )

        logger.info("=" * 60)

        return ReportMetadata(
            final_markdown_path=report_path,
            charts=chart_files,
            analyses=analyses,
            portfolio_metrics=portfolio_metrics,
            review_issues=review_issues,
            generated_at=timestamp,
            performance_metrics=performance_metrics,
        )
