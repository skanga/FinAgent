"""
financial_report_agent_fixed.py

Fixed version with explicit ticker handling and no guessing.
"""

import os
import json
import logging
import time
import hashlib
import pickle
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf

# Set matplotlib backend BEFORE importing pyplot to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ---------- Logging Configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
@dataclass
class Config:
    """Application configuration."""
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"
    default_period: str = "1y"
    max_retries: int = 3
    request_timeout: int = 30
    cache_ttl_hours: int = 24
    max_workers: int = 3
    risk_free_rate: float = 0.02
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        return cls(
            openai_api_key=api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            max_workers=int(os.getenv("MAX_WORKERS", "3")),
            cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24"))
        )

# ---------- Data Models ----------
@dataclass
class AdvancedMetrics:
    """Advanced financial risk metrics."""
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None
    r_squared: Optional[float] = None
    var_95: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None

@dataclass
class TickerAnalysis:
    """Analysis results for a single ticker."""
    ticker: str
    csv_path: str
    chart_path: str
    latest_close: float
    avg_daily_return: float
    volatility: float
    ratios: Dict[str, Optional[float]]
    advanced_metrics: AdvancedMetrics
    comparative_analysis: Optional[Any] = None
    sample_data: List[Dict[str, Any]] = None
    alerts: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.sample_data is None:
            self.sample_data = []
        if self.alerts is None:
            self.alerts = []

@dataclass
class ReportMetadata:
    """Metadata for generated report."""
    final_markdown_path: str
    charts: List[str]
    analyses: Dict[str, TickerAnalysis]
    review_issues: List[str]
    generated_at: str
    performance_metrics: Dict[str, Any]

# ---------- Enhanced Chart Generator (Thread-Safe) ----------
class ThreadSafeChartGenerator:
    """Thread-safe chart generation with proper resource management."""
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, ticker: str, output_path: str) -> None:
        """
        Create a price chart with moving averages (thread-safe).
        """
        # Create figure directly without using pyplot state
        fig, ax = plt.subplots(figsize=(12, 6))
        
        try:
            # Plot price and moving averages
            ax.plot(df['Date'], df['close'], label='Close Price', linewidth=2)
            if '30d_ma' in df.columns:
                ax.plot(df['Date'], df['30d_ma'], label='30-day MA', alpha=0.7)
            if '50d_ma' in df.columns:
                ax.plot(df['Date'], df['50d_ma'], label='50-day MA', alpha=0.7)
            
            ax.set_title(f'{ticker} Price History', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Created chart: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create chart for {ticker}: {e}")
            raise
        finally:
            # Critical: explicitly close and release memory
            plt.close(fig)
            plt.close('all')  # Ensure all figures are closed
    
    @staticmethod
    def create_comparison_chart(analyses: Dict[str, TickerAnalysis], output_path: str) -> None:
        """
        Create a comparison chart of multiple tickers (thread-safe).
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        try:
            plotted_count = 0
            for ticker, analysis in analyses.items():
                if analysis.error or not analysis.csv_path or not Path(analysis.csv_path).exists():
                    continue
                
                try:
                    df = pd.read_csv(analysis.csv_path)
                    if len(df) == 0:
                        continue
                        
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    
                    # Normalize prices to percentage change from start
                    start_price = df['close'].iloc[0]
                    if start_price > 0:  # Avoid division by zero
                        normalized_prices = (df['close'] / start_price - 1) * 100
                        ax.plot(df['Date'], normalized_prices, label=ticker, linewidth=2)
                        plotted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to plot {ticker} in comparison chart: {e}")
                    continue
            
            if plotted_count > 0:
                ax.set_title('Comparative Performance (Normalized)', fontsize=16, fontweight='bold')
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Percentage Change (%)', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Created comparison chart with {plotted_count} tickers: {output_path}")
            else:
                logger.warning("No valid data for comparison chart")
                
        except Exception as e:
            logger.error(f"Failed to create comparison chart: {e}")
            raise
        finally:
            plt.close(fig)
            plt.close('all')

# ---------- Enhanced Data Validation ----------
def validate_price_data(df: pd.DataFrame, ticker: str) -> None:
    """
    Enhanced data validation with better error messages.
    """
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    
    required_columns = ['Date', 'Close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for {ticker}: {missing_columns}")
    
    # Check for sufficient data points
    if len(df) < 5:
        raise ValueError(f"Insufficient data points for {ticker}: only {len(df)} rows")
    
    # Check for NaN values in critical columns
    if df['Close'].isna().all():
        raise ValueError(f"All Close prices are NaN for {ticker}")
    
    null_count = df['Close'].isna().sum()
    if null_count > len(df) * 0.1:  # More than 10% nulls
        raise ValueError(f"Too many null values in Close prices for {ticker}: {null_count}/{len(df)}")
    
    # Check for suspicious data patterns
    if (df['Close'] <= 0).any():
        raise ValueError(f"Invalid Close prices (non-positive values) for {ticker}")
    
    # Check price variability (avoid flat lines)
    price_std = df['Close'].std()
    if price_std == 0:
        raise ValueError(f"No price variability for {ticker}")

# ---------- Improved Data Fetcher ----------
class RobustDataFetcher:
    """Enhanced data fetcher with better error handling."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def fetch_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch historical price data with enhanced validation.
        """
        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty")
        
        ticker = ticker.strip().upper()
        logger.info(f"Fetching price history for {ticker} (period: {period})")
        
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, auto_adjust=False)
            
            if hist.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            hist = hist.reset_index()
            hist["ticker"] = ticker
            
            # Enhanced validation
            validate_price_data(hist, ticker)
            
            logger.info(f"Successfully fetched {len(hist)} rows for {ticker}")
            return hist
            
        except Exception as e:
            logger.error(f"Failed to fetch price history for {ticker}: {e}")
            raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    def validate_ticker_exists(self, ticker: str) -> bool:
        """
        Quick validation to check if a ticker exists.
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if ticker exists and has data, False otherwise
        """
        try:
            t = yf.Ticker(ticker)
            info = t.info
            # Check if we have basic info and it's not an empty dict or error
            if not info or 'symbol' not in info:
                return False
            return True
        except Exception:
            return False

# ---------- Enhanced Financial Analyzer ----------
class EnhancedFinancialAnalyzer:
    """Improved financial analyzer with better error handling."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def compute_metrics(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical metrics with robust error handling.
        """
        if df_prices.empty:
            raise ValueError("Cannot analyze empty DataFrame")
        
        required_cols = ['Date', 'Close']
        missing_cols = [col for col in required_cols if col not in df_prices.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        try:
            df = df_prices.copy()
            
            # Fix: Use utc=True to avoid datetime warning
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Basic price metrics
            df['close'] = df['Close'].astype(float)
            df['daily_return'] = df['close'].pct_change()
            
            # Moving averages with error handling
            min_periods_30 = min(5, len(df) // 6)  # Adaptive min periods
            min_periods_50 = min(10, len(df) // 4)
            
            df['30d_ma'] = df['close'].rolling(
                window=min(30, len(df)), 
                min_periods=min_periods_30
            ).mean()
            
            df['50d_ma'] = df['close'].rolling(
                window=min(50, len(df)), 
                min_periods=min_periods_50
            ).mean()
            
            # Volatility with adaptive window
            vol_window = min(20, len(df) // 3)
            df['volatility'] = df['daily_return'].rolling(
                window=vol_window, 
                min_periods=max(2, vol_window // 4)
            ).std()
            
            logger.info(f"Computed metrics for {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            raise ValueError(f"Analysis failed: {str(e)}")
    
    def calculate_advanced_metrics(self, returns: pd.Series) -> AdvancedMetrics:
        """
        Calculate risk metrics with robust error handling.
        """
        if len(returns) < 10:  # Require minimum data points
            return AdvancedMetrics()
        
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) < 10:
                return AdvancedMetrics()
            
            # Basic calculations
            annualized_return = returns_clean.mean() * 252
            annualized_vol = returns_clean.std() * np.sqrt(252)
            
            # Sharpe Ratio
            sharpe = (annualized_return - self.risk_free_rate) / annualized_vol if annualized_vol > 0 else None
            
            # Maximum Drawdown
            cumulative = (1 + returns_clean).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min() if not drawdown.empty else None
            
            # Value at Risk
            var_95 = np.percentile(returns_clean, 5) if len(returns_clean) > 0 else None
            
            return AdvancedMetrics(
                sharpe_ratio=float(sharpe) if sharpe is not None else None,
                max_drawdown=float(max_dd) if max_dd is not None else None,
                var_95=float(var_95) if var_95 is not None else None
            )
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return AdvancedMetrics()

# ---------- Main Orchestrator with Explicit Ticker Handling ----------
class FixedFinancialReportOrchestrator:
    """Fixed orchestrator with explicit ticker handling - no guessing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = RobustDataFetcher(timeout=config.request_timeout)
        self.analyzer = EnhancedFinancialAnalyzer(risk_free_rate=config.risk_free_rate)
        self.chart_gen = ThreadSafeChartGenerator()
    
    def validate_tickers(self, tickers: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate tickers and return valid and invalid lists.
        
        Args:
            tickers: List of ticker symbols to validate
            
        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            ticker_clean = ticker.strip().upper()
            
            # Basic format validation
            if not ticker_clean or len(ticker_clean) > 5 or not ticker_clean.isalpha():
                invalid_tickers.append(ticker)
                logger.warning(f"Invalid ticker format: {ticker}")
                continue
            
            # Check if ticker exists
            if self.fetcher.validate_ticker_exists(ticker_clean):
                valid_tickers.append(ticker_clean)
            else:
                invalid_tickers.append(ticker)
                logger.warning(f"Ticker not found or has no data: {ticker}")
        
        return valid_tickers, invalid_tickers
    
    def analyze_ticker_safe(self, ticker: str, period: str, output_dir: Path) -> TickerAnalysis:
        """
        Safe ticker analysis with comprehensive error handling.
        """
        logger.info(f"Analyzing {ticker}")
        
        try:
            # Fetch price data
            df_prices = self.fetcher.fetch_price_history(ticker, period)
            
            # Compute metrics
            df_analyzed = self.analyzer.compute_metrics(df_prices)
            
            # Save CSV
            csv_path = output_dir / f"{ticker}_prices.csv"
            df_analyzed.to_csv(csv_path, index=False)
            
            # Create chart (will be thread-safe)
            chart_path = output_dir / f"{ticker}_close.png"
            self.chart_gen.create_price_chart(df_analyzed, ticker, str(chart_path))
            
            # Calculate advanced metrics
            returns = df_analyzed['daily_return'].dropna()
            advanced_metrics = self.analyzer.calculate_advanced_metrics(returns)
            
            # Extract key metrics
            latest_close = float(df_analyzed['close'].iloc[-1])
            avg_return = float(df_analyzed['daily_return'].mean())
            volatility = float(df_analyzed['volatility'].iloc[-1]) if not pd.isna(df_analyzed['volatility'].iloc[-1]) else 0.0
            
            # Sample data
            sample_data = df_analyzed.tail(3).to_dict(orient="records")
            
            return TickerAnalysis(
                ticker=ticker,
                csv_path=str(csv_path),
                chart_path=str(chart_path),
                latest_close=latest_close,
                avg_daily_return=avg_return,
                volatility=volatility,
                ratios={},  # Simplified for this fix
                advanced_metrics=advanced_metrics,
                sample_data=sample_data
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {ticker}: {e}")
            return TickerAnalysis(
                ticker=ticker,
                csv_path="",
                chart_path="",
                latest_close=0.0,
                avg_daily_return=0.0,
                volatility=0.0,
                ratios={},
                advanced_metrics=AdvancedMetrics(),
                sample_data=[],
                error=str(e)
            )
    
    def run_sequential_analysis(self, tickers: List[str], period: str, output_dir: Path) -> Dict[str, TickerAnalysis]:
        """
        Sequential analysis to avoid threading issues.
        """
        analyses = {}
        for ticker in tickers:
            analyses[ticker] = self.analyze_ticker_safe(ticker, period, output_dir)
        return analyses
    
    def run(self, tickers: List[str], period: str, output_dir: str = "./report_out_fixed") -> ReportMetadata:
        """
        Main execution with explicit ticker handling.
        
        Args:
            tickers: List of EXACT ticker symbols to analyze
            period: Time period for analysis
            output_dir: Output directory for results
            
        Returns:
            ReportMetadata object
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Financial Report Generation")
        logger.info("=" * 60)
        
        # Validate tickers
        logger.info(f"Validating {len(tickers)} tickers: {tickers}")
        valid_tickers, invalid_tickers = self.validate_tickers(tickers)
        
        # Print warnings for invalid tickers
        if invalid_tickers:
            print(f"\n⚠️  WARNING: The following tickers are invalid and will be ignored:")
            for invalid in invalid_tickers:
                print(f"   - {invalid}")
            print()
        
        # Check if we have any valid tickers
        if not valid_tickers:
            raise ValueError("No valid tickers provided. Please check your ticker symbols.")
        
        logger.info(f"Analyzing {len(valid_tickers)} valid tickers: {valid_tickers}")
        logger.info(f"Period: {period}")
        
        # Analyze tickers SEQUENTIALLY to avoid threading issues
        analyses = self.run_sequential_analysis(valid_tickers, period, output_path)
        
        # Generate charts
        chart_files = []
        for analysis in analyses.values():
            if not analysis.error and analysis.chart_path:
                chart_files.append(analysis.chart_path)
        
        # Create comparison chart if we have multiple successful analyses
        successful_analyses = {t: a for t, a in analyses.items() if not a.error}
        if len(successful_analyses) >= 2:
            try:
                comparison_path = output_path / "comparison_chart.png"
                self.chart_gen.create_comparison_chart(successful_analyses, str(comparison_path))
                if Path(comparison_path).exists():
                    chart_files.append(str(comparison_path))
                    logger.info("Comparison chart created successfully")
            except Exception as e:
                logger.warning(f"Comparison chart failed: {e}")
        
        # Prepare data for report
        analysis_status = {}
        for ticker, analysis in analyses.items():
            analysis_status[ticker] = {
                'success': not analysis.error,
                'error': analysis.error,
                'analysis': analysis
            }
        
        context = {
            "tickers": valid_tickers,
            "invalid_tickers": invalid_tickers,
            "period": period,
            "successful_count": len(successful_analyses),
            "failed_count": len(analyses) - len(successful_analyses)
        }
        
        findings = {
            ticker: {
                "latest_close": analysis.latest_close,
                "avg_daily_return": analysis.avg_daily_return,
                "volatility": analysis.volatility,
                "sharpe_ratio": analysis.advanced_metrics.sharpe_ratio,
                "max_drawdown": analysis.advanced_metrics.max_drawdown,
                "var_95": analysis.advanced_metrics.var_95
            }
            for ticker, analysis in successful_analyses.items()
        }
        
        # Create a simple report
        report_content = self._create_simple_report(context, findings, analysis_status, chart_files)
        
        # Save final report
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report_filename = f"financial_report_{timestamp}.md"
        report_path = output_path / report_filename
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        # Performance metrics
        execution_time = time.time() - start_time
        performance_metrics = {
            "execution_time_seconds": round(execution_time, 2),
            "tickers_requested": len(tickers),
            "valid_tickers": len(valid_tickers),
            "invalid_tickers": len(invalid_tickers),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(analyses) - len(successful_analyses),
            "charts_generated": len(chart_files)
        }
        
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info("=" * 60)
        logger.info("Financial Report Generation Complete")
        logger.info("=" * 60)
        
        return ReportMetadata(
            final_markdown_path=str(report_path),
            charts=chart_files,
            analyses=analyses,
            review_issues=[],  # Simplified version
            generated_at=timestamp,
            performance_metrics=performance_metrics
        )
    
    def _create_simple_report(self, context: Dict, findings: Dict, 
                            analysis_status: Dict, charts: List[str]) -> str:
        """Create a simple markdown report."""
        report = []
        
        # Header
        report.append("# Financial Analysis Report")
        report.append(f"*Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        report.append("")
        
        # Executive Summary
        successful = [t for t, s in analysis_status.items() if s.get('success')]
        failed_in_analysis = [t for t, s in analysis_status.items() if not s.get('success')]
        
        report.append("## Executive Summary")
        report.append(f"Analysis of {len(successful)} stocks over {context['period']} period.")
        
        if context['invalid_tickers']:
            report.append(f"**Note:** {len(context['invalid_tickers'])} tickers were invalid and ignored: {', '.join(context['invalid_tickers'])}")
        
        if failed_in_analysis:
            report.append(f"**Note:** {len(failed_in_analysis)} tickers failed during analysis: {', '.join(failed_in_analysis)}")
        
        report.append("")
        
        # Key Metrics Table
        report.append("## Key Performance Metrics")
        report.append("| Ticker | Price | Avg Return | Volatility | Sharpe | Max DD | VaR 95% |")
        report.append("|--------|-------|------------|------------|--------|--------|---------|")
        
        for ticker, status in analysis_status.items():
            if status.get('success'):
                analysis = status['analysis']
                metrics = analysis.advanced_metrics
                report.append(f"| {ticker} | ${analysis.latest_close:.2f} | "
                            f"{analysis.avg_daily_return*100:.3f}% | "
                            f"{analysis.volatility*100:.2f}% | "
                            f"{metrics.sharpe_ratio or 'N/A':.2f} | "
                            f"{(metrics.max_drawdown or 0)*100:.1f}% | "
                            f"{(metrics.var_95 or 0)*100:.2f}% |")
        
        report.append("")
        report.append("*Avg Return: Average daily return, Volatility: Annualized standard deviation, " 
                     "Sharpe: Risk-adjusted return, Max DD: Maximum drawdown, VaR 95%: Value at Risk (95% confidence)*")
        report.append("")
        
        # Individual Analysis
        report.append("## Individual Analysis")
        for ticker, status in analysis_status.items():
            if status.get('success'):
                analysis = status['analysis']
                metrics = analysis.advanced_metrics
                
                report.append(f"### {ticker}")
                report.append(f"- **Current Price:** ${analysis.latest_close:.2f}")
                report.append(f"- **Average Daily Return:** {analysis.avg_daily_return*100:.3f}%")
                report.append(f"- **Volatility:** {analysis.volatility*100:.2f}%")
                
                if metrics.sharpe_ratio:
                    sharpe_rating = "excellent" if metrics.sharpe_ratio > 1.0 else "good" if metrics.sharpe_ratio > 0.5 else "moderate"
                    report.append(f"- **Sharpe Ratio:** {metrics.sharpe_ratio:.2f} ({sharpe_rating} risk-adjusted returns)")
                
                if metrics.max_drawdown:
                    report.append(f"- **Maximum Drawdown:** {metrics.max_drawdown*100:.1f}%")
                
                if metrics.var_95:
                    report.append(f"- **VaR 95%:** {metrics.var_95*100:.2f}% (worst-case daily loss with 95% confidence)")
                
                report.append("")
        
        # Data Quality Notes
        if context['invalid_tickers'] or failed_in_analysis:
            report.append("## Data Quality Notes")
            
            if context['invalid_tickers']:
                report.append("### Invalid Tickers (Ignored)")
                for ticker in context['invalid_tickers']:
                    report.append(f"- {ticker}: Ticker symbol not found or invalid")
                report.append("")
            
            if failed_in_analysis:
                report.append("### Analysis Failures")
                for ticker in failed_in_analysis:
                    error = analysis_status[ticker].get('error', 'Unknown error')
                    report.append(f"- {ticker}: {error}")
                report.append("")
        
        # Charts
        if charts:
            report.append("## Generated Charts")
            for chart in charts:
                chart_name = Path(chart).name
                report.append(f"- `{chart_name}`")
            report.append("")
        
        return "\n".join(report)

# ---------- CLI Argument Support ----------
def setup_cli_arguments():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate financial reports with risk analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                 # Use default tickers (GOOG, MSFT, ORCL)
  %(prog)s --tickers AAPL,TSLA,NVDA        # Analyze specific tickers
  %(prog)s --tickers AAPL,MSFT --period 6mo --output-dir ./my_report
  %(prog)s --tickers INVALID,TEST,AAPL     # Invalid tickers will be ignored with warnings
        """
    )
    
    parser.add_argument(
        '--tickers',
        '-t',
        type=str,
        help='Comma-separated list of EXACT ticker symbols to analyze (e.g., "GOOG,MSFT,ORCL")'
    )
    
    parser.add_argument(
        '--period',
        '-p',
        type=str,
        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        default='1y',
        help='Time period for analysis (default: 1y)'
    )
    
    parser.add_argument(
        '--output-dir',
        '-o', 
        type=str,
        default="./financial_reports",
        help='Output directory for reports and charts (default: ./financial_reports)'
    )
    
    return parser

def parse_ticker_argument(tickers_arg: Optional[str]) -> List[str]:
    """
    Parse ticker argument from CLI.
    
    Args:
        tickers_arg: Comma-separated string of tickers or None
        
    Returns:
        List of ticker symbols
    """
    if not tickers_arg:
        # Default tickers if none provided
        return ['GOOG', 'MSFT', 'ORCL']
    
    # Split by comma and clean up
    tickers = [t.strip().upper() for t in tickers_arg.split(',') if t.strip()]
    
    if not tickers:
        raise ValueError("No valid tickers provided after parsing")
    
    return tickers

# ---------- Main Entry Point ----------
def main():
    """Main entry point with explicit ticker handling."""
    parser = setup_cli_arguments()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config.from_env()
        
        # Parse tickers
        tickers = parse_ticker_argument(args.tickers)
        
        print("=" * 60)
        print("FINANCIAL REPORT GENERATOR")
        print("=" * 60)
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Period: {args.period}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)
        print()
        
        # Create orchestrator
        orchestrator = FixedFinancialReportOrchestrator(config)
        
        # Run analysis
        result = orchestrator.run(tickers, args.period, output_dir=args.output_dir)
        
        # Print results
        print("\n" + "=" * 60)
        print("REPORT GENERATION SUMMARY")
        print("=" * 60)
        print(f"📊 Report: {result.final_markdown_path}")
        print(f"⏱️  Execution Time: {result.performance_metrics['execution_time_seconds']}s")
        print(f"🖼️  Charts: {len(result.charts)} generated")
        print(f"📈 Tickers requested: {result.performance_metrics['tickers_requested']}")
        print(f"✅ Valid tickers: {result.performance_metrics['valid_tickers']}")
        print(f"❌ Invalid tickers: {result.performance_metrics['invalid_tickers']}")
        print(f"🔬 Successful analyses: {result.performance_metrics['successful_analyses']}")
        print(f"⚠️  Failed analyses: {result.performance_metrics['failed_analyses']}")
        
        print("\nAnalysis Results:")
        for ticker, analysis in result.analyses.items():
            if analysis.error:
                print(f"  ❌ {ticker}: {analysis.error}")
            else:
                print(f"  ✅ {ticker}: ${analysis.latest_close:.2f} | "
                      f"Return: {analysis.avg_daily_return*100:.3f}% | "
                      f"Vol: {analysis.volatility*100:.2f}% | "
                      f"Sharpe: {analysis.advanced_metrics.sharpe_ratio or 'N/A':.2f}")
        
        print("=" * 60)
        print("🎉 Report generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())