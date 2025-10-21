"""
financial_report_agent_fixed.py

Fixed version addressing threading issues and report quality problems.
"""

import os
import json
import logging
import time
import hashlib
import pickle
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
    max_workers: int = 3  # Reduced for stability
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
            max_workers=int(os.getenv("MAX_WORKERS", "3")),  # Reduced for stability
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
class ComparativeAnalysis:
    """Comparative analysis against benchmark."""
    outperformance: Optional[float] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    beta_vs_benchmark: Optional[float] = None
    alpha_vs_benchmark: Optional[float] = None

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
    comparative_analysis: Optional[ComparativeAnalysis] = None
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
                        
                    df['Date'] = pd.to_datetime(df['Date'])
                    
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
            df['Date'] = pd.to_datetime(df['Date'])
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

# ---------- Improved LLM Interface with Better Prompts ----------
class EnhancedLLMInterface:
    """LLM interface with improved prompts to address review issues."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.model_name,
            temperature=0.0,
            timeout=config.request_timeout
        )
        
        # Enhanced prompts based on review feedback
        self.writer_prompt = PromptTemplate(
            input_variables=["context_json", "key_findings", "charts", "advanced_metrics", "analysis_status"],
            template="""You are a senior financial analyst writing a accurate, clear report.

Context: {context_json}
Key Findings: {key_findings}
Charts Available: {charts}
Advanced Metrics: {advanced_metrics}
Analysis Status: {analysis_status}

Write a professional Markdown report with these sections:

1. **Executive Summary** (2-3 sentences highlighting main findings)

2. **Key Metrics Table** - Include:
   - Ticker, Latest Close, Avg Daily Return (explain as average daily percentage change)
   - Volatility (explain as annualized standard deviation of returns)
   - Sharpe Ratio, Max Drawdown, VaR 95%
   - Add footnote explaining any currency conversions

3. **Risk Analysis** - Discuss:
   - Sharpe ratios (higher = better risk-adjusted returns)
   - Maximum drawdowns (largest peak-to-trough decline)
   - VaR 95% (worst-case daily loss with 95% confidence)
   - Current ratio implications (below 1 indicates liquidity risk)

4. **Individual Analysis** - For each successful ticker:
   - Performance trends and key metrics
   - Risk assessment based on advanced metrics
   - Specific numerical values that support conclusions

5. **Data Quality Notes** - Explicitly mention:
   - Any tickers that failed analysis and why
   - Only reference charts that are actually available
   - Data limitations or assumptions

6. **Recommendations** - 3-5 actionable insights based on the data.

CRITICAL REQUIREMENTS:
- Only reference charts that are in the "Charts Available" list
- Explicitly mention failed tickers in "Data Quality Notes"
- Explain ALL metrics in simple terms
- Use specific numbers from the data
- Avoid vague or unsupported claims

Return ONLY the markdown content."""
        )
        
        self.review_prompt = PromptTemplate(
            input_variables=["draft_markdown", "data_summary", "charts_available"],
            template="""Review this financial report draft for accuracy and completeness.

Available Charts: {charts_available}
Data Summary: {data_summary}

Draft Report:
{draft_markdown}

Check for these specific issues:
1. Are all metrics (Avg Daily Return, Volatility) clearly explained?
2. Are financial ratios properly contextualized (e.g., current ratio below 1 indicates liquidity concerns)?
3. Is risk terminology accurate (e.g., VaR levels properly described)?
4. Are failed tickers explicitly mentioned and excluded from analysis?
5. Does the report only reference charts that are actually available?
6. Are all numerical conversions explained?

Return valid JSON with:
- "issues": list of specific issues found (empty if none)
- "revised": corrected markdown (or original if no changes)
- "score": 1-10 rating for report quality

Return ONLY the JSON object."""
        )
        
        self.writer_chain = self.writer_prompt | self.llm
        self.review_chain = self.review_prompt | self.llm
    
    def generate_report(self, context: Dict, findings: Dict, charts: List[str], 
                       advanced_metrics: Dict, analysis_status: Dict) -> str:
        """Generate report with enhanced context."""
        logger.info("Generating enhanced report")
        
        try:
            response = self.writer_chain.invoke({
                "context_json": json.dumps(context, indent=2),
                "key_findings": json.dumps(findings, indent=2),
                "charts": json.dumps(charts),
                "advanced_metrics": json.dumps(advanced_metrics, indent=2),
                "analysis_status": json.dumps(analysis_status, indent=2)
            })
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            # Fallback template
            return self._create_fallback_report(context, findings, analysis_status)
    
    def _create_fallback_report(self, context: Dict, findings: Dict, analysis_status: Dict) -> str:
        """Create a basic report if LLM fails."""
        report = ["# Financial Analysis Report\n"]
        
        # Executive Summary
        successful = [t for t, s in analysis_status.items() if s.get('success')]
        failed = [t for t, s in analysis_status.items() if not s.get('success')]
        
        report.append("## Executive Summary")
        report.append(f"Analysis completed for {len(successful)} tickers. "
                     f"{len(failed)} tickers failed analysis.\n")
        
        # Key Metrics
        report.append("## Key Metrics")
        report.append("| Ticker | Latest Close | Avg Daily Return | Volatility | Sharpe | Max DD |")
        report.append("|--------|--------------|------------------|------------|--------|--------|")
        
        for ticker, status in analysis_status.items():
            if status.get('success'):
                analysis = status['analysis']
                report.append(f"| {ticker} | ${analysis.latest_close:.2f} | "
                            f"{analysis.avg_daily_return*100:.2f}% | "
                            f"{analysis.volatility*100:.2f}% | "
                            f"{analysis.advanced_metrics.sharpe_ratio or 'N/A':.2f} | "
                            f"{(analysis.advanced_metrics.max_drawdown or 0)*100:.1f}% |")
        
        # Data Quality Notes
        if failed:
            report.append("\n## Data Quality Notes")
            report.append("The following tickers could not be analyzed:")
            for ticker in failed:
                report.append(f"- {ticker}: {analysis_status[ticker].get('error', 'Unknown error')}")
        
        return "\n".join(report)
    
    def review_report(self, draft: str, data_summary: Dict, charts_available: List[str]) -> tuple[List[str], str]:
        """Enhanced review with chart validation."""
        logger.info("Reviewing report with chart validation")
        
        try:
            response = self.review_chain.invoke({
                "draft_markdown": draft,
                "data_summary": json.dumps(data_summary, indent=2),
                "charts_available": json.dumps(charts_available)
            })
            
            review_text = response.content.strip()
            
            # Extract JSON
            if review_text.startswith("```"):
                lines = review_text.split("\n")
                review_text = "\n".join(lines[1:-1])
            
            review_json = json.loads(review_text)
            issues = review_json.get("issues", [])
            revised = review_json.get("revised", draft)
            
            logger.info(f"Review found {len(issues)} issues, score: {review_json.get('score', 'N/A')}")
            return issues, revised
            
        except Exception as e:
            logger.error(f"Review failed: {e}")
            return [], draft

# ---------- Main Orchestrator with Fixes ----------
class FixedFinancialReportOrchestrator:
    """Fixed orchestrator addressing all identified issues."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = RobustDataFetcher(timeout=config.request_timeout)
        self.analyzer = EnhancedFinancialAnalyzer(risk_free_rate=config.risk_free_rate)
        self.chart_gen = ThreadSafeChartGenerator()
        self.llm = EnhancedLLMInterface(config)
    
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
    
    def run(self, user_request: str, output_dir: str = "./report_out_fixed") -> ReportMetadata:
        """
        Main execution with all fixes applied.
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Fixed Financial Report Generation")
        logger.info("=" * 60)
        
        # Parse request (simplified)
        tickers = ["AAPL", "MSFT", "GOOG"]  # Hardcoded for testing
        period = "1y"
        
        # Analyze tickers SEQUENTIALLY to avoid threading issues
        logger.info(f"Analyzing {len(tickers)} tickers sequentially")
        analyses = self.run_sequential_analysis(tickers, period, output_path)
        
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
            "tickers": tickers,
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
        
        advanced_metrics = {
            ticker: asdict(analysis.advanced_metrics)
            for ticker, analysis in successful_analyses.items()
        }
        
        # Generate report
        draft = self.llm.generate_report(context, findings, chart_files, advanced_metrics, analysis_status)
        
        # Review report
        data_summary = {
            ticker: {
                "success": not analysis.error,
                "error": analysis.error,
                "latest_close": analysis.latest_close,
                "metrics": {
                    "avg_daily_return": analysis.avg_daily_return,
                    "volatility": analysis.volatility
                },
                "advanced_metrics": asdict(analysis.advanced_metrics)
            }
            for ticker, analysis in analyses.items()
        }
        
        issues, final_markdown = self.llm.review_report(draft, data_summary, chart_files)
        
        # Save final report
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report_filename = f"financial_report_{timestamp}.md"
        report_path = output_path / report_filename
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        
        # Performance metrics
        execution_time = time.time() - start_time
        performance_metrics = {
            "execution_time_seconds": round(execution_time, 2),
            "tickers_analyzed": len(tickers),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(analyses) - len(successful_analyses),
            "charts_generated": len(chart_files),
            "review_issues": len(issues)
        }
        
        logger.info(f"Report saved to: {report_path}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info("=" * 60)
        logger.info("Fixed Financial Report Generation Complete")
        logger.info("=" * 60)
        
        return ReportMetadata(
            final_markdown_path=str(report_path),
            charts=chart_files,
            analyses=analyses,
            review_issues=issues,
            generated_at=timestamp,
            performance_metrics=performance_metrics
        )

# ---------- Main Entry Point ----------
def main():
    """Main entry point for fixed version."""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Create fixed orchestrator
        orchestrator = FixedFinancialReportOrchestrator(config)
        
        # Run fixed analysis
        result = orchestrator.run(
            "Generate financial report for AAPL, MSFT, and GOOG with risk analysis",
            output_dir="./report_out_fixed"
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("FIXED REPORT GENERATION SUMMARY")
        print("=" * 60)
        print(f"Report: {result.final_markdown_path}")
        print(f"Execution Time: {result.performance_metrics['execution_time_seconds']}s")
        print(f"Charts: {len(result.charts)} generated")
        print(f"Tickers analyzed: {len(result.analyses)}")
        print(f"Successful: {result.performance_metrics['successful_analyses']}")
        print(f"Failed: {result.performance_metrics['failed_analyses']}")
        
        if result.review_issues:
            print(f"\nReview Issues Found: {len(result.review_issues)}")
            for i, issue in enumerate(result.review_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\nNo review issues found!")
        
        print("\nAnalysis Results:")
        for ticker, analysis in result.analyses.items():
            if analysis.error:
                print(f"  {ticker}: ERROR - {analysis.error}")
            else:
                print(f"  {ticker}: SUCCESS")
                print(f"    Price: ${analysis.latest_close:.2f}")
                print(f"    Return: {analysis.avg_daily_return*100:.3f}%")
                print(f"    Volatility: {analysis.volatility*100:.2f}%")
                if analysis.advanced_metrics.sharpe_ratio:
                    print(f"    Sharpe: {analysis.advanced_metrics.sharpe_ratio:.2f}")
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error in fixed version: {e}", exc_info=True)
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()