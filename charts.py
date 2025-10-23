"""
Chart generation with thread-safety.
"""

import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict
from models import TickerAnalysis
from constants import AnalysisThresholds, ChartConfiguration

logger = logging.getLogger(__name__)


class ThreadSafeChartGenerator:
    """Thread-safe chart generation."""

    @staticmethod
    def create_price_chart(
        price_data_with_indicators: pd.DataFrame, ticker: str, output_path: Path
    ) -> None:
        """
        Creates a comprehensive price chart with technical indicators.

        Args:
            price_data_with_indicators (pd.DataFrame): The DataFrame containing the price history with computed indicators.
            ticker (str): The ticker symbol.
            output_path (str): The path to save the chart to.
        """
        fig, (price_axis, rsi_axis) = plt.subplots(
            2,
            1,
            figsize=ChartConfiguration.PRICE_CHART_SIZE,
            gridspec_kw={"height_ratios": ChartConfiguration.PRICE_CHART_HEIGHT_RATIOS},
        )

        try:
            # Main price chart
            price_axis.plot(
                price_data_with_indicators["Date"],
                price_data_with_indicators["close"],
                label="Close Price",
                linewidth=2,
                color="blue",
            )
            if "30d_ma" in price_data_with_indicators.columns:
                price_axis.plot(
                    price_data_with_indicators["Date"],
                    price_data_with_indicators["30d_ma"],
                    label="30-day MA",
                    alpha=0.7,
                    color="orange",
                )
            if "50d_ma" in price_data_with_indicators.columns:
                price_axis.plot(
                    price_data_with_indicators["Date"],
                    price_data_with_indicators["50d_ma"],
                    label="50-day MA",
                    alpha=0.7,
                    color="red",
                )

            # Bollinger Bands
            if (
                "bollinger_upper" in price_data_with_indicators.columns
                and "bollinger_lower" in price_data_with_indicators.columns
            ):
                price_axis.fill_between(
                    price_data_with_indicators["Date"],
                    price_data_with_indicators["bollinger_upper"],
                    price_data_with_indicators["bollinger_lower"],
                    alpha=0.2,
                    label="Bollinger Bands",
                    color="gray",
                )

            price_axis.set_title(
                f"{ticker} Technical Analysis",
                fontsize=ChartConfiguration.TITLE_FONTSIZE,
                fontweight="bold",
            )
            price_axis.set_ylabel("Price ($)", fontsize=ChartConfiguration.AXIS_FONTSIZE)
            price_axis.legend()
            price_axis.grid(True, alpha=0.3)

            # RSI subplot
            if "rsi" in price_data_with_indicators.columns:
                rsi_axis.plot(
                    price_data_with_indicators["Date"],
                    price_data_with_indicators["rsi"],
                    label="RSI",
                    linewidth=2,
                    color="purple",
                )
                rsi_axis.axhline(
                    y=AnalysisThresholds.OVERBOUGHT_RSI,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Overbought ({AnalysisThresholds.OVERBOUGHT_RSI})",
                )
                rsi_axis.axhline(
                    y=AnalysisThresholds.OVERSOLD_RSI,
                    color="g",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Oversold ({AnalysisThresholds.OVERSOLD_RSI})",
                )
                rsi_axis.set_ylabel("RSI", fontsize=ChartConfiguration.AXIS_FONTSIZE)
                rsi_axis.set_xlabel("Date", fontsize=ChartConfiguration.AXIS_FONTSIZE)
                rsi_axis.legend()
                rsi_axis.grid(True, alpha=0.3)
                rsi_axis.set_ylim(0, 100)

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close("all")
            # Clear figure and axes references
            del fig, price_axis, rsi_axis

    @staticmethod
    def create_comparison_chart(
        analyses: Dict[str, TickerAnalysis], output_path: Path
    ) -> None:
        """
        Creates a normalized comparison chart of multiple tickers.

        Args:
            analyses (Dict[str, TickerAnalysis]): A dictionary of ticker analyses.
            output_path (str): The path to save the chart to.
        """
        fig, comparison_axis = plt.subplots(figsize=ChartConfiguration.COMPARISON_CHART_SIZE)

        try:
            plotted_count = 0
            for ticker, analysis in analyses.items():
                if analysis.error or not Path(analysis.csv_path).exists():
                    continue

                ticker_price_data = None  # Initialize for cleanup
                try:
                    ticker_price_data = pd.read_csv(analysis.csv_path)
                    if len(ticker_price_data) == 0:
                        continue

                    ticker_price_data["Date"] = pd.to_datetime(
                        ticker_price_data["Date"], utc=True
                    )
                    start_price = ticker_price_data["close"].iloc[0]
                    if start_price > 0:
                        percentage_return_from_start = (
                            ticker_price_data["close"] / start_price - 1
                        ) * 100
                        comparison_axis.plot(
                            ticker_price_data["Date"],
                            percentage_return_from_start,
                            label=ticker,
                            linewidth=2,
                        )
                        plotted_count += 1

                except (
                    KeyError,
                    ValueError,
                    TypeError,
                    OSError,
                    pd.errors.ParserError,
                ) as e:
                    logger.debug(f"Failed to plot {ticker}: {e}")
                    continue
                finally:
                    # Clean up DataFrame after each ticker to free memory
                    if ticker_price_data is not None:
                        del ticker_price_data

            if plotted_count > 0:
                comparison_axis.set_title(
                    "Comparative Performance (Normalized)",
                    fontsize=ChartConfiguration.TITLE_FONTSIZE,
                    fontweight="bold",
                )
                comparison_axis.set_xlabel("Date", fontsize=ChartConfiguration.AXIS_FONTSIZE)
                comparison_axis.set_ylabel("Return (%)", fontsize=ChartConfiguration.AXIS_FONTSIZE)
                comparison_axis.legend()
                comparison_axis.grid(True, alpha=0.3)
                comparison_axis.axhline(y=0, color="k", linestyle="-", alpha=0.5)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
                logger.debug(f"Created comparison chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close("all")
            # Clear figure and axes references
            del fig, comparison_axis

    @staticmethod
    def create_risk_reward_chart(
        analyses: Dict[str, TickerAnalysis], output_path: Path
    ) -> None:
        """
        Creates a risk-reward scatter plot of multiple tickers.

        Args:
            analyses (Dict[str, TickerAnalysis]): A dictionary of ticker analyses.
            output_path (str): The path to save the chart to.
        """
        fig, scatter_axis = plt.subplots(figsize=ChartConfiguration.RISK_REWARD_CHART_SIZE)

        try:
            for ticker, analysis in analyses.items():
                if analysis.error:
                    continue

                average_daily_return_pct = analysis.avg_daily_return * 100
                daily_volatility_pct = analysis.volatility * 100

                scatter_axis.scatter(
                    daily_volatility_pct,
                    average_daily_return_pct,
                    s=100,
                    label=ticker,
                    alpha=0.7,
                )
                scatter_axis.annotate(
                    ticker,
                    (daily_volatility_pct, average_daily_return_pct),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            scatter_axis.set_xlabel("Risk (Volatility %)", fontsize=ChartConfiguration.AXIS_FONTSIZE)
            scatter_axis.set_ylabel(
                "Return (Avg Daily %)", fontsize=ChartConfiguration.AXIS_FONTSIZE
            )
            scatter_axis.set_title(
                "Risk-Return Profile", fontsize=ChartConfiguration.TITLE_FONTSIZE, fontweight="bold"
            )
            scatter_axis.axhline(y=0, color="k", linestyle="-", alpha=0.3)
            scatter_axis.axvline(x=0, color="k", linestyle="-", alpha=0.3)
            scatter_axis.grid(True, alpha=0.3)
            scatter_axis.legend()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created risk-reward chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close("all")
            # Clear figure and axes references
            del fig, scatter_axis
