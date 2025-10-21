"""
Chart generation with thread-safety.
"""
import gc
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict
from models import TickerAnalysis
from constants import (
    RSI_OVERBOUGHT_THRESHOLD,
    RSI_OVERSOLD_THRESHOLD,
    PRICE_CHART_SIZE,
    COMPARISON_CHART_SIZE,
    RISK_REWARD_CHART_SIZE,
    CHART_DPI,
    CHART_TITLE_FONTSIZE,
    CHART_AXIS_FONTSIZE,
    PRICE_CHART_HEIGHT_RATIOS
)

logger = logging.getLogger(__name__)


class ThreadSafeChartGenerator:
    """Thread-safe chart generation."""
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, ticker: str, output_path: str) -> None:
        """Create comprehensive price chart with technical indicators."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PRICE_CHART_SIZE,
                                        gridspec_kw={'height_ratios': PRICE_CHART_HEIGHT_RATIOS})
        
        try:
            # Main price chart
            ax1.plot(df['Date'], df['close'], label='Close Price', linewidth=2, color='blue')
            if '30d_ma' in df.columns:
                ax1.plot(df['Date'], df['30d_ma'], label='30-day MA', alpha=0.7, color='orange')
            if '50d_ma' in df.columns:
                ax1.plot(df['Date'], df['50d_ma'], label='50-day MA', alpha=0.7, color='red')

            # Bollinger Bands
            if 'bollinger_upper' in df.columns and 'bollinger_lower' in df.columns:
                ax1.fill_between(df['Date'], df['bollinger_upper'], df['bollinger_lower'],
                               alpha=0.2, label='Bollinger Bands', color='gray')

            ax1.set_title(f'{ticker} Technical Analysis', fontsize=CHART_TITLE_FONTSIZE, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=CHART_AXIS_FONTSIZE)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # RSI subplot
            if 'rsi' in df.columns:
                ax2.plot(df['Date'], df['rsi'], label='RSI', linewidth=2, color='purple')
                ax2.axhline(y=RSI_OVERBOUGHT_THRESHOLD, color='r', linestyle='--', alpha=0.7,
                           label=f'Overbought ({RSI_OVERBOUGHT_THRESHOLD})')
                ax2.axhline(y=RSI_OVERSOLD_THRESHOLD, color='g', linestyle='--', alpha=0.7,
                           label=f'Oversold ({RSI_OVERSOLD_THRESHOLD})')
                ax2.set_ylabel('RSI', fontsize=CHART_AXIS_FONTSIZE)
                ax2.set_xlabel('Date', fontsize=CHART_AXIS_FONTSIZE)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)

            plt.tight_layout()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=CHART_DPI, bbox_inches='tight')
            logger.info(f"Created chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close('all')
            # Clear figure and axes references
            del fig, ax1, ax2
            # Force garbage collection for large objects
            gc.collect()
    
    @staticmethod
    def create_comparison_chart(analyses: Dict[str, TickerAnalysis], output_path: str) -> None:
        """Create normalized comparison chart."""
        fig, ax = plt.subplots(figsize=COMPARISON_CHART_SIZE)

        try:
            plotted_count = 0
            for ticker, analysis in analyses.items():
                if analysis.error or not Path(analysis.csv_path).exists():
                    continue

                df = None  # Initialize for cleanup
                try:
                    df = pd.read_csv(analysis.csv_path)
                    if len(df) == 0:
                        continue

                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    start_price = df['close'].iloc[0]
                    if start_price > 0:
                        normalized = (df['close'] / start_price - 1) * 100
                        ax.plot(df['Date'], normalized, label=ticker, linewidth=2)
                        plotted_count += 1

                except (KeyError, ValueError, TypeError, OSError, pd.errors.ParserError) as e:
                    logger.warning(f"Failed to plot {ticker}: {e}")
                    continue
                finally:
                    # Clean up DataFrame after each ticker to free memory
                    if df is not None:
                        del df

            if plotted_count > 0:
                ax.set_title('Comparative Performance (Normalized)', fontsize=CHART_TITLE_FONTSIZE, fontweight='bold')
                ax.set_xlabel('Date', fontsize=CHART_AXIS_FONTSIZE)
                ax.set_ylabel('Return (%)', fontsize=CHART_AXIS_FONTSIZE)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=CHART_DPI, bbox_inches='tight')
                logger.info(f"Created comparison chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close('all')
            # Clear figure and axes references
            del fig, ax
            # Force garbage collection for large objects
            gc.collect()
    
    @staticmethod
    def create_risk_reward_chart(analyses: Dict[str, TickerAnalysis], output_path: str) -> None:
        """Create risk-reward scatter plot."""
        fig, ax = plt.subplots(figsize=RISK_REWARD_CHART_SIZE)

        try:
            for ticker, analysis in analyses.items():
                if analysis.error:
                    continue

                returns = analysis.avg_daily_return * 100
                risk = analysis.volatility * 100

                ax.scatter(risk, returns, s=100, label=ticker, alpha=0.7)
                ax.annotate(ticker, (risk, returns), xytext=(5, 5), textcoords='offset points')

            ax.set_xlabel('Risk (Volatility %)', fontsize=CHART_AXIS_FONTSIZE)
            ax.set_ylabel('Return (Avg Daily %)', fontsize=CHART_AXIS_FONTSIZE)
            ax.set_title('Risk-Return Profile', fontsize=CHART_TITLE_FONTSIZE, fontweight='bold')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.legend()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=CHART_DPI, bbox_inches='tight')
            logger.info(f"Created risk-reward chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close('all')
            # Clear figure and axes references
            del fig, ax
            # Force garbage collection for large objects
            gc.collect()
