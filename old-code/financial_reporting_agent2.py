"""
financial_reporting_agent.py
"""

import os
import re
import json
import time
import pickle
import logging
import hashlib
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

from scipy import stats
from pathlib import Path
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from langchain_core.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed


# Set matplotlib backend BEFORE importing pyplot to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


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
    benchmark_ticker: str = "SPY"
    
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
            cache_ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
            benchmark_ticker=os.getenv("BENCHMARK_TICKER", "SPY")
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
    treynor_ratio: Optional[float] = None
    information_ratio: Optional[float] = None

@dataclass
class ComparativeAnalysis:
    """Comparative analysis against benchmark."""
    outperformance: Optional[float] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    beta_vs_benchmark: Optional[float] = None
    alpha_vs_benchmark: Optional[float] = None
    relative_volatility: Optional[float] = None

@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_position: Optional[float] = None

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
    technical_indicators: TechnicalIndicators
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
        Create a price chart with moving averages and Bollinger Bands.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        try:
            # Main price chart
            ax1.plot(df['Date'], df['close'], label='Close Price', linewidth=2, color='blue')
            if '30d_ma' in df.columns:
                ax1.plot(df['Date'], df['30d_ma'], label='30-day MA', alpha=0.7, color='orange')
            if '50d_ma' in df.columns:
                ax1.plot(df['Date'], df['50d_ma'], label='50-day MA', alpha=0.7, color='red')
            
            # Bollinger Bands if available
            if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
                ax1.fill_between(df['Date'], df['bollinger_upper'], df['bollinger_lower'], 
                               alpha=0.2, label='Bollinger Bands', color='gray')
            
            ax1.set_title(f'{ticker} Technical Analysis', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI subplot
            if 'rsi' in df.columns:
                ax2.plot(df['Date'], df['rsi'], label='RSI', linewidth=2, color='purple')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                ax2.set_ylabel('RSI', fontsize=12)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Created advanced chart: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create chart for {ticker}: {e}")
            raise
        finally:
            plt.close(fig)
            plt.close('all')
    
    @staticmethod
    def create_comparison_chart(analyses: Dict[str, TickerAnalysis], output_path: str) -> None:
        """Create comparison chart of multiple tickers."""
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
                    if start_price > 0:
                        normalized_prices = (df['close'] / start_price - 1) * 100
                        ax.plot(df['Date'], normalized_prices, label=ticker, linewidth=2)
                        plotted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to plot {ticker} in comparison chart: {e}")
                    continue
            
            if plotted_count > 0:
                ax.set_title('Comparative Performance (Normalized to 0%)', fontsize=16, fontweight='bold')
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
    
    @staticmethod
    def create_risk_reward_chart(analyses: Dict[str, TickerAnalysis], output_path: str) -> None:
        """Create risk-reward scatter plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        try:
            for ticker, analysis in analyses.items():
                if analysis.error:
                    continue
                
                returns = analysis.avg_daily_return * 100  # Convert to percentage
                risk = analysis.volatility * 100  # Convert to percentage
                
                ax.scatter(risk, returns, s=100, label=ticker, alpha=0.7)
                ax.annotate(ticker, (risk, returns), xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Risk (Volatility %)', fontsize=12)
            ax.set_ylabel('Return (Avg Daily Return %)', fontsize=12)
            ax.set_title('Risk-Return Profile', fontsize=16, fontweight='bold')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Created risk-reward chart: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create risk-reward chart: {e}")
            raise
        finally:
            plt.close(fig)
            plt.close('all')

# ---------- Advanced Financial Analyzer ----------
class AdvancedFinancialAnalyzer:
    """Comprehensive financial analysis with advanced metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02, benchmark_ticker: str = "SPY"):
        self.risk_free_rate = risk_free_rate
        self.benchmark_ticker = benchmark_ticker
    
    def compute_metrics(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """Compute comprehensive technical metrics."""
        df = df_prices.copy()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Basic price metrics
        df['close'] = df['Close'].astype(float)
        df['daily_return'] = df['close'].pct_change()
        
        # Moving averages
        df['30d_ma'] = df['close'].rolling(window=min(30, len(df)), min_periods=5).mean()
        df['50d_ma'] = df['close'].rolling(window=min(50, len(df)), min_periods=10).mean()
        df['200d_ma'] = df['close'].rolling(window=min(200, len(df)), min_periods=20).mean()
        
        # Volatility
        df['volatility'] = df['daily_return'].rolling(window=min(20, len(df)//3), min_periods=5).std()
        
        # RSI
        df['rsi'] = self._compute_rsi(df['close'])
        
        # Bollinger Bands
        df['bollinger_upper'], df['bollinger_lower'] = self._compute_bollinger_bands(df['close'])
        df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # MACD
        df['macd'], df['macd_signal'] = self._compute_macd(df['close'])
        
        return df
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_advanced_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> AdvancedMetrics:
        """Calculate comprehensive risk metrics."""
        if len(returns) < 10:
            return AdvancedMetrics()
        
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
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else None
        
        # Calmar Ratio
        calmar = annualized_return / abs(max_dd) if max_dd and max_dd != 0 else None
        
        # Value at Risk
        var_95 = np.percentile(returns_clean, 5)
        
        # Beta and Alpha (if benchmark provided)
        beta, alpha, r_squared, treynor, information_ratio = self._calculate_benchmark_metrics(
            returns_clean, benchmark_returns
        )
        
        return AdvancedMetrics(
            sharpe_ratio=float(sharpe) if sharpe is not None else None,
            max_drawdown=float(max_dd) if max_dd is not None else None,
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            var_95=float(var_95),
            sortino_ratio=float(sortino) if sortino is not None else None,
            calmar_ratio=float(calmar) if calmar is not None else None,
            treynor_ratio=treynor,
            information_ratio=information_ratio
        )
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series]) -> tuple:
        """Calculate metrics relative to benchmark."""
        if benchmark_returns is None or len(benchmark_returns) < 10:
            return None, None, None, None, None
        
        benchmark_clean = benchmark_returns.dropna()
        common_index = returns.index.intersection(benchmark_clean.index)
        
        if len(common_index) < 10:
            return None, None, None, None, None
        
        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_clean.loc[common_index]
        
        # Calculate beta using covariance
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else None
        
        # Calculate alpha
        alpha = None
        if beta is not None:
            benchmark_return = aligned_benchmark.mean() * 252
            alpha = (aligned_returns.mean() * 252 - self.risk_free_rate) - beta * (benchmark_return - self.risk_free_rate)
        
        # R-squared
        r_squared = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1] ** 2 if len(aligned_returns) > 1 else None
        
        # Treynor Ratio
        treynor = (aligned_returns.mean() * 252 - self.risk_free_rate) / beta if beta and beta != 0 else None
        
        # Information Ratio
        active_returns = aligned_returns - aligned_benchmark
        information_ratio = active_returns.mean() * 252 / (active_returns.std() * np.sqrt(252)) if active_returns.std() > 0 else None
        
        return beta, alpha, r_squared, treynor, information_ratio
    
    def compute_ratios(self, financials: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Compute financial ratios from statements."""
        ratios = {
            'pe_ratio': None,
            'forward_pe': None,
            'peg_ratio': None,
            'price_to_sales': None,
            'price_to_book': None,
            'debt_to_equity': None,
            'current_ratio': None,
            'quick_ratio': None,
            'return_on_equity': None,
            'return_on_assets': None,
            'profit_margin': None,
            'operating_margin': None,
            'gross_margin': None,
            'ebitda_margin': None
        }
        
        try:
            # Try to get info from yfinance
            # This is a simplified version - in practice you'd parse the financial statements
            pass
        except Exception as e:
            logger.warning(f"Could not compute financial ratios: {e}")
        
        return ratios

# ---------- Alert System ----------
class AlertSystem:
    """Monitor for significant market movements and data issues."""
    
    def __init__(self, volatility_threshold: float = 0.05, drawdown_threshold: float = -0.10):
        self.volatility_threshold = volatility_threshold
        self.drawdown_threshold = drawdown_threshold
    
    def check_alerts(self, analysis: TickerAnalysis) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # Price movement alerts
        if analysis.volatility > self.volatility_threshold:
            alerts.append(f"High volatility: {analysis.volatility*100:.2f}%")
        
        # Drawdown alerts
        if (analysis.advanced_metrics.max_drawdown and 
            analysis.advanced_metrics.max_drawdown < self.drawdown_threshold):
            alerts.append(f"Significant drawdown: {analysis.advanced_metrics.max_drawdown*100:.2f}%")
        
        # Risk metric alerts
        if analysis.advanced_metrics.var_95 and analysis.advanced_metrics.var_95 < -0.03:
            alerts.append(f"High VaR (95%): {analysis.advanced_metrics.var_95*100:.2f}%")
        
        # RSI alerts
        if (analysis.technical_indicators.rsi and 
            analysis.technical_indicators.rsi > 70):
            alerts.append(f"Overbought (RSI: {analysis.technical_indicators.rsi:.1f})")
        elif (analysis.technical_indicators.rsi and 
              analysis.technical_indicators.rsi < 30):
            alerts.append(f"Oversold (RSI: {analysis.technical_indicators.rsi:.1f})")
        
        # Comparative analysis alerts
        if (analysis.comparative_analysis and 
            analysis.comparative_analysis.outperformance and 
            analysis.comparative_analysis.outperformance < -0.05):
            alerts.append(f"Underperforming benchmark: {analysis.comparative_analysis.outperformance*100:.2f}%")
        
        return alerts

# ---------- Robust Data Fetcher ----------
class RobustDataFetcher:
    """Enhanced data fetcher with better error handling."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def fetch_price_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical price data with enhanced validation."""
        ticker = ticker.strip().upper()
        logger.info(f"Fetching price history for {ticker} (period: {period})")
        
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, auto_adjust=False)
            
            if hist.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            hist = hist.reset_index()
            hist["ticker"] = ticker
            
            # Basic validation
            if len(hist) < 5:
                raise ValueError(f"Insufficient data for {ticker}: only {len(hist)} rows")
            if hist['Close'].isna().all():
                raise ValueError(f"All Close prices are NaN for {ticker}")
            
            logger.info(f"Successfully fetched {len(hist)} rows for {ticker}")
            return hist
            
        except Exception as e:
            logger.error(f"Failed to fetch price history for {ticker}: {e}")
            raise ValueError(f"Failed to fetch data for {ticker}: {str(e)}")
    
    def validate_ticker_exists(self, ticker: str) -> bool:
        """Quick validation to check if a ticker exists."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return bool(info and 'symbol' in info)
        except Exception:
            return False

# ---------- Advanced LLM Interface ----------
class AdvancedLLMInterface:
    """LLM interface for generating sophisticated reports."""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.model_name,
            temperature=0.0,
            timeout=config.request_timeout
        )
    
    def generate_advanced_report(self, analyses: Dict[str, TickerAnalysis], 
                               benchmark_analysis: Optional[TickerAnalysis],
                               period: str) -> str:
        """Generate a comprehensive financial report using LLM."""
        
        # Prepare data for LLM
        report_data = self._prepare_report_data(analyses, benchmark_analysis, period)
        
        # For now, create a detailed markdown report without LLM
        # In a full implementation, you would use the LLM here
        return self._create_detailed_report(report_data)
    
    def _prepare_report_data(self, analyses: Dict[str, TickerAnalysis],
                           benchmark_analysis: Optional[TickerAnalysis],
                           period: str) -> Dict[str, Any]:
        """Prepare data for report generation."""
        successful_analyses = {t: a for t, a in analyses.items() if not a.error}
        failed_analyses = {t: a for t, a in analyses.items() if a.error}
        
        return {
            'period': period,
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_analyses,
            'benchmark_analysis': benchmark_analysis,
            'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        }
    
    def _create_detailed_report(self, report_data: Dict[str, Any]) -> str:
        """Create a detailed markdown report with all advanced features."""
        report = []
        
        # Header
        report.append("# 📊 Advanced Financial Analysis Report")
        report.append(f"*Generated on {report_data['generated_at']}*")
        report.append("")
        
        # Executive Summary
        report.append("## 🎯 Executive Summary")
        report.append(self._generate_executive_summary(report_data))
        report.append("")
        
        # Key Performance Metrics
        report.append("## 📈 Key Performance Metrics")
        report.append(self._generate_metrics_table(report_data))
        report.append("")
        
        # Risk Analysis
        report.append("## ⚠️ Risk Analysis")
        report.append(self._generate_risk_analysis(report_data))
        report.append("")
        
        # Technical Analysis
        report.append("## 🔧 Technical Analysis")
        report.append(self._generate_technical_analysis(report_data))
        report.append("")
        
        # Comparative Analysis
        if report_data['benchmark_analysis']:
            report.append("## 📊 Comparative Analysis vs Benchmark")
            report.append(self._generate_comparative_analysis(report_data))
            report.append("")
        
        # Individual Stock Analysis
        report.append("## 📋 Individual Stock Analysis")
        report.append(self._generate_individual_analysis(report_data))
        report.append("")
        
        # Investment Recommendations
        report.append("## 💡 Investment Recommendations")
        report.append(self._generate_recommendations(report_data))
        report.append("")
        
        # Data Quality Notes
        if report_data['failed_analyses']:
            report.append("## 📝 Data Quality Notes")
            report.append(self._generate_data_quality_notes(report_data))
            report.append("")
        
        return "\n".join(report)
    
    def _generate_executive_summary(self, report_data: Dict) -> str:
        """Generate executive summary."""
        successful = report_data['successful_analyses']
        period = report_data['period']
        
        if not successful:
            return "No successful analyses to summarize."
        
        # Calculate overall metrics
        best_performer = max(successful.items(), 
                           key=lambda x: x[1].avg_daily_return)
        worst_performer = min(successful.items(), 
                            key=lambda x: x[1].avg_daily_return)
        highest_risk = max(successful.items(), 
                         key=lambda x: x[1].volatility)
        
        summary = [
            f"This report analyzes {len(successful)} stocks over a {period} period. ",
            f"**{best_performer[0]}** showed the strongest performance with an average daily return of {best_performer[1].avg_daily_return*100:.3f}%, ",
            f"while **{worst_performer[0]}** had the lowest returns. ",
            f"**{highest_risk[0]}** exhibited the highest volatility at {highest_risk[1].volatility*100:.2f}%."
        ]
        
        return " ".join(summary)
    
    def _generate_metrics_table(self, report_data: Dict) -> str:
        """Generate comprehensive metrics table."""
        table = [
            "| Ticker | Price | Return | Volatility | Sharpe | Sortino | Max DD | VaR 95% | Beta | Alpha | RSI |",
            "|--------|-------|--------|------------|--------|---------|--------|---------|------|-------|-----|"
        ]
        
        for ticker, analysis in report_data['successful_analyses'].items():
            metrics = analysis.advanced_metrics
            tech = analysis.technical_indicators
            
            row = (
                f"| {ticker} | "
                f"${analysis.latest_close:.2f} | "
                f"{analysis.avg_daily_return*100:.3f}% | "
                f"{analysis.volatility*100:.2f}% | "
                f"{metrics.sharpe_ratio or 'N/A':.2f} | "
                f"{metrics.sortino_ratio or 'N/A':.2f} | "
                f"{(metrics.max_drawdown or 0)*100:.1f}% | "
                f"{(metrics.var_95 or 0)*100:.2f}% | "
                f"{metrics.beta or 'N/A':.2f} | "
                f"{(metrics.alpha or 0)*100:.2f}% | "
                f"{tech.rsi or 'N/A':.1f} |"
            )
            table.append(row)
        
        table.extend([
            "",
            "**Metrics Explanation:**",
            "- **Return:** Average daily return (%)",
            "- **Volatility:** Annualized standard deviation of returns (%)", 
            "- **Sharpe:** Risk-adjusted returns (higher is better)",
            "- **Sortino:** Downside risk-adjusted returns (higher is better)",
            "- **Max DD:** Maximum drawdown - worst peak-to-trough decline (%)",
            "- **VaR 95%:** Value at Risk - worst daily loss with 95% confidence (%)",
            "- **Beta:** Sensitivity to market movements",
            "- **Alpha:** Excess return over benchmark (%)",
            "- **RSI:** Relative Strength Index (30=oversold, 70=overbought)"
        ])
        
        return "\n".join(table)
    
    def _generate_risk_analysis(self, report_data: Dict) -> str:
        """Generate risk analysis section."""
        analysis = []
        
        for ticker, data in report_data['successful_analyses'].items():
            metrics = data.advanced_metrics
            risk_assessment = []
            
            # Sharpe ratio assessment
            if metrics.sharpe_ratio:
                if metrics.sharpe_ratio > 1.0:
                    risk_assessment.append("excellent risk-adjusted returns")
                elif metrics.sharpe_ratio > 0.5:
                    risk_assessment.append("good risk-adjusted returns")
                else:
                    risk_assessment.append("moderate risk-adjusted returns")
            
            # Drawdown assessment
            if metrics.max_drawdown:
                if metrics.max_drawdown < -0.20:
                    risk_assessment.append("high historical drawdown risk")
                elif metrics.max_drawdown < -0.10:
                    risk_assessment.append("moderate drawdown risk")
                else:
                    risk_assessment.append("low drawdown risk")
            
            # VaR assessment
            if metrics.var_95:
                if metrics.var_95 < -0.04:
                    risk_assessment.append("high daily loss potential")
                elif metrics.var_95 < -0.02:
                    risk_assessment.append("moderate daily loss potential")
                else:
                    risk_assessment.append("low daily loss potential")
            
            if risk_assessment:
                analysis.append(f"- **{ticker}:** {', '.join(risk_assessment)}")
        
        return "\n".join(analysis) if analysis else "No risk data available."
    
    def _generate_technical_analysis(self, report_data: Dict) -> str:
        """Generate technical analysis section."""
        analysis = []
        
        for ticker, data in report_data['successful_analyses'].items():
            tech = data.technical_indicators
            tech_analysis = []
            
            if tech.rsi:
                if tech.rsi > 70:
                    tech_analysis.append("overbought (RSI > 70)")
                elif tech.rsi < 30:
                    tech_analysis.append("oversold (RSI < 30)")
                else:
                    tech_analysis.append("neutral RSI")
            
            if tech.bollinger_position:
                if tech.bollinger_position > 0.8:
                    tech_analysis.append("near upper Bollinger Band")
                elif tech.bollinger_position < 0.2:
                    tech_analysis.append("near lower Bollinger Band")
            
            if tech_analysis:
                analysis.append(f"- **{ticker}:** {', '.join(tech_analysis)}")
        
        return "\n".join(analysis) if analysis else "No technical indicators available."
    
    def _generate_comparative_analysis(self, report_data: Dict) -> str:
        """Generate comparative analysis section."""
        analysis = []
        benchmark = report_data['benchmark_analysis']
        
        if not benchmark:
            return "No benchmark data available for comparison."
        
        for ticker, data in report_data['successful_analyses'].items():
            comp = data.comparative_analysis
            if comp:
                comparison = []
                
                if comp.outperformance:
                    if comp.outperformance > 0:
                        comparison.append(f"outperformed benchmark by {comp.outperformance*100:.2f}%")
                    else:
                        comparison.append(f"underperformed benchmark by {abs(comp.outperformance)*100:.2f}%")
                
                if comp.beta_vs_benchmark:
                    if comp.beta_vs_benchmark > 1.2:
                        comparison.append("high beta (more volatile than market)")
                    elif comp.beta_vs_benchmark < 0.8:
                        comparison.append("low beta (less volatile than market)")
                    else:
                        comparison.append("market-like volatility")
                
                if comparison:
                    analysis.append(f"- **{ticker}:** {', '.join(comparison)}")
        
        return "\n".join(analysis) if analysis else "No comparative data available."
    
    def _generate_individual_analysis(self, report_data: Dict) -> str:
        """Generate individual stock analysis."""
        analysis = []
        
        for ticker, data in report_data['successful_analyses'].items():
            metrics = data.advanced_metrics
            tech = data.technical_indicators
            comp = data.comparative_analysis
            
            stock_analysis = [f"### {ticker}"]
            stock_analysis.append(f"- **Current Price:** ${data.latest_close:.2f}")
            stock_analysis.append(f"- **Performance:** {data.avg_daily_return*100:.3f}% avg daily return, {data.volatility*100:.2f}% volatility")
            
            if metrics.sharpe_ratio:
                stock_analysis.append(f"- **Risk-Adjusted Returns:** Sharpe {metrics.sharpe_ratio:.2f}, Sortino {metrics.sortino_ratio or 'N/A':.2f}")
            
            if metrics.max_drawdown:
                stock_analysis.append(f"- **Risk Metrics:** Max drawdown {metrics.max_drawdown*100:.1f}%, VaR 95% {metrics.var_95*100:.2f}%")
            
            if tech.rsi:
                stock_analysis.append(f"- **Technical:** RSI {tech.rsi:.1f}")
            
            if comp and comp.beta_vs_benchmark:
                stock_analysis.append(f"- **Market Correlation:** Beta {comp.beta_vs_benchmark:.2f}, Alpha {comp.alpha_vs_benchmark*100 if comp.alpha_vs_benchmark else 0:.2f}%")
            
            if data.alerts:
                stock_analysis.append(f"- **Alerts:** {', '.join(data.alerts)}")
            
            analysis.extend(stock_analysis)
            analysis.append("")
        
        return "\n".join(analysis)
    
    def _generate_recommendations(self, report_data: Dict) -> str:
        """Generate investment recommendations."""
        successful = report_data['successful_analyses']
        
        if not successful:
            return "No recommendations available."
        
        # Sort by Sharpe ratio (risk-adjusted returns)
        best_risk_adjusted = sorted(
            successful.items(),
            key=lambda x: x[1].advanced_metrics.sharpe_ratio or -999,
            reverse=True
        )[:3]
        
        # Sort by total return
        best_performers = sorted(
            successful.items(),
            key=lambda x: x[1].avg_daily_return,
            reverse=True
        )[:3]
        
        # Sort by lowest risk
        lowest_risk = sorted(
            successful.items(),
            key=lambda x: x[1].volatility
        )[:3]
        
        recommendations = [
            "### 🏆 Best Risk-Adjusted Returns",
            *[f"- **{ticker}** (Sharpe: {data.advanced_metrics.sharpe_ratio or 'N/A':.2f})" 
              for ticker, data in best_risk_adjusted],
            "",
            "### 🚀 Top Performers",
            *[f"- **{ticker}** ({data.avg_daily_return*100:.3f}% avg daily return)" 
              for ticker, data in best_performers],
            "",
            "### 🛡️ Lowest Risk",
            *[f"- **{ticker}** ({data.volatility*100:.2f}% volatility)" 
              for ticker, data in lowest_risk]
        ]
        
        return "\n".join(recommendations)
    
    def _generate_data_quality_notes(self, report_data: Dict) -> str:
        """Generate data quality notes."""
        notes = []
        
        for ticker, analysis in report_data['failed_analyses'].items():
            notes.append(f"- **{ticker}:** {analysis.error}")
        
        return "\n".join(notes)

# ---------- Main Orchestrator ----------
class AdvancedFinancialReportOrchestrator:
    """Advanced orchestrator with all features restored."""
    
    def __init__(self, config: Config):
        self.config = config
        self.fetcher = RobustDataFetcher(timeout=config.request_timeout)
        self.analyzer = AdvancedFinancialAnalyzer(
            risk_free_rate=config.risk_free_rate,
            benchmark_ticker=config.benchmark_ticker
        )
        self.chart_gen = ThreadSafeChartGenerator()
        self.alert_system = AlertSystem()
        self.llm_interface = AdvancedLLMInterface(config)
    
    def validate_tickers(self, tickers: List[str]) -> tuple[List[str], List[str]]:
        """Validate tickers and return valid and invalid lists."""
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
    
    def analyze_ticker(self, ticker: str, period: str, output_dir: Path, 
                      benchmark_returns: Optional[pd.Series] = None) -> TickerAnalysis:
        """Comprehensive ticker analysis with all advanced features."""
        logger.info(f"Analyzing {ticker}")
        
        try:
            # Fetch price data
            df_prices = self.fetcher.fetch_price_history(ticker, period)
            
            # Compute comprehensive metrics
            df_analyzed = self.analyzer.compute_metrics(df_prices)
            
            # Save CSV
            csv_path = output_dir / f"{ticker}_prices.csv"
            df_analyzed.to_csv(csv_path, index=False)
            
            # Create advanced chart
            chart_path = output_dir / f"{ticker}_technical.png"
            self.chart_gen.create_price_chart(df_analyzed, ticker, str(chart_path))
            
            # Calculate advanced metrics
            returns = df_analyzed['daily_return'].dropna()
            advanced_metrics = self.analyzer.calculate_advanced_metrics(returns, benchmark_returns)
            
            # Get latest technical indicators
            latest_tech = TechnicalIndicators(
                rsi=float(df_analyzed['rsi'].iloc[-1]) if 'rsi' in df_analyzed.columns else None,
                macd=float(df_analyzed['macd'].iloc[-1]) if 'macd' in df_analyzed.columns else None,
                macd_signal=float(df_analyzed['macd_signal'].iloc[-1]) if 'macd_signal' in df_analyzed.columns else None,
                bollinger_upper=float(df_analyzed['bollinger_upper'].iloc[-1]) if 'bollinger_upper' in df_analyzed.columns else None,
                bollinger_lower=float(df_analyzed['bollinger_lower'].iloc[-1]) if 'bollinger_lower' in df_analyzed.columns else None,
                bollinger_position=float(df_analyzed['bollinger_position'].iloc[-1]) if 'bollinger_position' in df_analyzed.columns else None
            )
            
            # Extract key metrics
            latest_close = float(df_analyzed['close'].iloc[-1])
            avg_return = float(df_analyzed['daily_return'].mean())
            volatility = float(df_analyzed['volatility'].iloc[-1]) if not pd.isna(df_analyzed['volatility'].iloc[-1]) else 0.0
            
            # Compute financial ratios
            ratios = self.analyzer.compute_ratios({})  # Empty for now
            
            # Sample data
            sample_data = df_analyzed.tail(5).to_dict(orient="records")
            
            analysis = TickerAnalysis(
                ticker=ticker,
                csv_path=str(csv_path),
                chart_path=str(chart_path),
                latest_close=latest_close,
                avg_daily_return=avg_return,
                volatility=volatility,
                ratios=ratios,
                advanced_metrics=advanced_metrics,
                technical_indicators=latest_tech,
                sample_data=sample_data
            )
            
            # Check for alerts
            analysis.alerts = self.alert_system.check_alerts(analysis)
            
            return analysis
            
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
                technical_indicators=TechnicalIndicators(),
                sample_data=[],
                error=str(e)
            )
    
    def run_sequential_analysis(self, tickers: List[str], period: str, output_dir: Path,
                              benchmark_returns: Optional[pd.Series] = None) -> Dict[str, TickerAnalysis]:
        """Sequential analysis to avoid threading issues."""
        analyses = {}
        for ticker in tickers:
            analyses[ticker] = self.analyze_ticker(ticker, period, output_dir, benchmark_returns)
        return analyses
    
    def run(self, tickers: List[str], period: str, output_dir: str = "./advanced_reports") -> ReportMetadata:
        """Main execution with all advanced features."""
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("Starting Advanced Financial Report Generation")
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
        
        # Fetch benchmark data for comparative analysis
        benchmark_analysis = None
        benchmark_returns = None
        
        try:
            benchmark_analysis = self.analyze_ticker(self.config.benchmark_ticker, period, output_path)
            if not benchmark_analysis.error:
                benchmark_df = pd.read_csv(benchmark_analysis.csv_path)
                benchmark_returns = benchmark_df['daily_return']
                logger.info(f"Benchmark analysis completed for {self.config.benchmark_ticker}")
        except Exception as e:
            logger.warning(f"Could not analyze benchmark {self.config.benchmark_ticker}: {e}")
        
        # Analyze tickers SEQUENTIALLY
        analyses = self.run_sequential_analysis(valid_tickers, period, output_path, benchmark_returns)
        
        # Perform comparative analysis for each ticker
        if benchmark_returns is not None:
            for ticker, analysis in analyses.items():
                if not analysis.error:
                    try:
                        ticker_df = pd.read_csv(analysis.csv_path)
                        ticker_returns = ticker_df['daily_return']
                        
                        # Align the series
                        common_index = ticker_returns.dropna().index.intersection(benchmark_returns.dropna().index)
                        if len(common_index) > 10:
                            aligned_ticker = ticker_returns.loc[common_index]
                            aligned_benchmark = benchmark_returns.loc[common_index]
                            
                            # Calculate comparative metrics
                            ticker_cumulative = (1 + aligned_ticker).prod() - 1
                            benchmark_cumulative = (1 + aligned_benchmark).prod() - 1
                            outperformance = ticker_cumulative - benchmark_cumulative
                            correlation = aligned_ticker.corr(aligned_benchmark)
                            
                            analysis.comparative_analysis = ComparativeAnalysis(
                                outperformance=float(outperformance),
                                correlation=float(correlation) if not pd.isna(correlation) else None,
                                beta_vs_benchmark=analysis.advanced_metrics.beta,
                                alpha_vs_benchmark=analysis.advanced_metrics.alpha
                            )
                    except Exception as e:
                        logger.warning(f"Comparative analysis failed for {ticker}: {e}")
        
        # Generate charts
        chart_files = []
        for analysis in analyses.values():
            if not analysis.error and analysis.chart_path:
                chart_files.append(analysis.chart_path)
        
        # Create comparison chart
        successful_analyses = {t: a for t, a in analyses.items() if not a.error}
        if len(successful_analyses) >= 2:
            try:
                comparison_path = output_path / "comparison_chart.png"
                self.chart_gen.create_comparison_chart(successful_analyses, str(comparison_path))
                chart_files.append(str(comparison_path))
                
                risk_reward_path = output_path / "risk_reward_chart.png"
                self.chart_gen.create_risk_reward_chart(successful_analyses, str(risk_reward_path))
                chart_files.append(str(risk_reward_path))
                
            except Exception as e:
                logger.warning(f"Chart generation failed: {e}")
        
        # Generate advanced report
        report_content = self.llm_interface.generate_advanced_report(
            analyses, benchmark_analysis, period
        )
        
        # Save final report
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        report_filename = f"advanced_financial_report_{timestamp}.md"
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
            "charts_generated": len(chart_files),
            "benchmark_analyzed": benchmark_analysis is not None and not benchmark_analysis.error
        }
        
        logger.info(f"Advanced report saved to: {report_path}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info("=" * 60)
        logger.info("Advanced Financial Report Generation Complete")
        logger.info("=" * 60)
        
        return ReportMetadata(
            final_markdown_path=str(report_path),
            charts=chart_files,
            analyses=analyses,
            review_issues=[],  # We handle quality in the report itself
            generated_at=timestamp,
            performance_metrics=performance_metrics
        )

# ---------- CLI Setup ----------
def setup_cli_arguments():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate advanced financial reports with comprehensive analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                 # Use default tickers (GOOG, MSFT, ORCL)
  %(prog)s --tickers AAPL,TSLA,NVDA        # Analyze specific tickers
  %(prog)s --tickers AAPL,MSFT --period 6mo --output-dir ./my_advanced_report
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
        default="./advanced_financial_reports",
        help='Output directory for reports and charts (default: ./advanced_financial_reports)'
    )
    
    return parser

def parse_ticker_argument(tickers_arg: Optional[str]) -> List[str]:
    """Parse ticker argument from CLI."""
    if not tickers_arg:
        return ['GOOG', 'MSFT', 'ORCL']
    
    tickers = [t.strip().upper() for t in tickers_arg.split(',') if t.strip()]
    
    if not tickers:
        raise ValueError("No valid tickers provided after parsing")
    
    return tickers

# ---------- Main Entry Point ----------
def main():
    """Main entry point for advanced version."""
    parser = setup_cli_arguments()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config.from_env()
        
        # Parse tickers
        tickers = parse_ticker_argument(args.tickers)
        
        print("=" * 60)
        print("ADVANCED FINANCIAL REPORT GENERATOR")
        print("=" * 60)
        print(f"📊 Tickers: {', '.join(tickers)}")
        print(f"⏰ Period: {args.period}")
        print(f"📁 Output: {args.output_dir}")
        print(f"⚖️  Benchmark: {config.benchmark_ticker}")
        print("=" * 60)
        print()
        
        # Create advanced orchestrator
        orchestrator = AdvancedFinancialReportOrchestrator(config)
        
        # Run advanced analysis
        result = orchestrator.run(tickers, args.period, output_dir=args.output_dir)
        
        # Print results
        print("\n" + "=" * 60)
        print("ADVANCED REPORT GENERATION SUMMARY")
        print("=" * 60)
        print(f"📊 Report: {result.final_markdown_path}")
        print(f"⏱️  Execution Time: {result.performance_metrics['execution_time_seconds']}s")
        print(f"🖼️  Charts: {len(result.charts)} generated")
        print(f"📈 Tickers requested: {result.performance_metrics['tickers_requested']}")
        print(f"✅ Valid tickers: {result.performance_metrics['valid_tickers']}")
        print(f"❌ Invalid tickers: {result.performance_metrics['invalid_tickers']}")
        print(f"🔬 Successful analyses: {result.performance_metrics['successful_analyses']}")
        print(f"⚠️  Failed analyses: {result.performance_metrics['failed_analyses']}")
        print(f"⚖️  Benchmark analyzed: {result.performance_metrics['benchmark_analyzed']}")
        
        print("\nAdvanced Features Included:")
        print("  ✅ Technical Analysis (RSI, MACD, Bollinger Bands)")
        print("  ✅ Advanced Risk Metrics (Sharpe, Sortino, Calmar, VaR)")
        print("  ✅ Comparative Analysis vs Benchmark")
        print("  ✅ Risk-Reward Analysis")
        print("  ✅ Alert System for significant movements")
        print("  ✅ Comprehensive Investment Recommendations")
        
        print("\nAnalysis Results:")
        for ticker, analysis in result.analyses.items():
            if analysis.error:
                print(f"  ❌ {ticker}: {analysis.error}")
            else:
                alerts = " ⚠️" if analysis.alerts else ""
                print(f"  ✅ {ticker}: ${analysis.latest_close:.2f} | "
                      f"Return: {analysis.avg_daily_return*100:.3f}% | "
                      f"Vol: {analysis.volatility*100:.2f}% | "
                      f"Sharpe: {analysis.advanced_metrics.sharpe_ratio or 'N/A':.2f}{alerts}")
        
        print("=" * 60)
        print("🎉 Advanced report generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
