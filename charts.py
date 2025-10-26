"""
Chart generation with thread-safety.
"""

import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from models import TickerAnalysis
from constants import AnalysisThresholds, ChartConfiguration, OptionsAnalysisParameters

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
        fig, (price_axis, rsi_axis, stoch_axis) = plt.subplots(
            3,
            1,
            figsize=(14, 12),
            gridspec_kw={"height_ratios": [3, 1, 1]},
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
            if "200d_ma" in price_data_with_indicators.columns:
                price_axis.plot(
                    price_data_with_indicators["Date"],
                    price_data_with_indicators["200d_ma"],
                    label="200-day MA",
                    alpha=0.7,
                    color="green",
                    linewidth=2,
                )
            # VWAP removed: Not appropriate for daily data (intraday indicator only)

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
                rsi_axis.legend(loc="upper left", fontsize=8)
                rsi_axis.grid(True, alpha=0.3)
                rsi_axis.set_ylim(0, 100)

            # Stochastic Oscillator subplot
            if "stochastic_k" in price_data_with_indicators.columns:
                stoch_axis.plot(
                    price_data_with_indicators["Date"],
                    price_data_with_indicators["stochastic_k"],
                    label="%K (fast)",
                    linewidth=2,
                    color="blue",
                )
                if "stochastic_d" in price_data_with_indicators.columns:
                    stoch_axis.plot(
                        price_data_with_indicators["Date"],
                        price_data_with_indicators["stochastic_d"],
                        label="%D (slow)",
                        linewidth=2,
                        color="red",
                    )
                stoch_axis.axhline(
                    y=AnalysisThresholds.OVERBOUGHT_STOCHASTIC,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Overbought ({AnalysisThresholds.OVERBOUGHT_STOCHASTIC})",
                )
                stoch_axis.axhline(
                    y=AnalysisThresholds.OVERSOLD_STOCHASTIC,
                    color="g",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Oversold ({AnalysisThresholds.OVERSOLD_STOCHASTIC})",
                )
                stoch_axis.set_ylabel("Stochastic", fontsize=ChartConfiguration.AXIS_FONTSIZE)
                stoch_axis.set_xlabel("Date", fontsize=ChartConfiguration.AXIS_FONTSIZE)
                stoch_axis.legend(loc="upper left", fontsize=8)
                stoch_axis.grid(True, alpha=0.3)
                stoch_axis.set_ylim(0, 100)

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created chart: {output_path}")

        finally:
            # Explicit cleanup to prevent memory leaks
            plt.close(fig)
            plt.close("all")
            # Clear figure and axes references
            del fig, price_axis, rsi_axis, stoch_axis

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
                # Skip if analysis has errors
                if analysis.error:
                    continue

                # Validate CSV path
                if not analysis.csv_path:
                    logger.debug(f"Skipping {ticker}: csv_path is None or empty")
                    continue

                csv_path = Path(analysis.csv_path)

                # Check if path exists
                if not csv_path.exists():
                    logger.debug(f"Skipping {ticker}: CSV file does not exist at {csv_path}")
                    continue

                # Check if path is a file (not a directory)
                if not csv_path.is_file():
                    logger.debug(f"Skipping {ticker}: Path is not a file: {csv_path}")
                    continue

                # Check if file has non-zero size
                if csv_path.stat().st_size == 0:
                    logger.debug(f"Skipping {ticker}: CSV file is empty: {csv_path}")
                    continue

                ticker_price_data = None  # Initialize for cleanup
                try:
                    ticker_price_data = pd.read_csv(csv_path)
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

    # ========================================================================
    # OPTIONS VISUALIZATION METHODS
    # ========================================================================

    @staticmethod
    def create_options_chain_heatmap(
        chains: List, output_path: Path, metric: str = "volume"
    ) -> None:
        """
        Creates a heatmap of options chain data (volume, OI, or IV).

        Args:
            chains: List of OptionsChain objects
            output_path: Path to save the chart
            metric: "volume", "open_interest", or "implied_volatility"
        """

        if not chains:
            logger.warning("No chains provided for heatmap")
            return

        fig, (call_ax, put_ax) = plt.subplots(
            1, 2, figsize=OptionsAnalysisParameters.HEATMAP_RESOLUTION
        )

        try:
            # Collect all strikes across all expirations
            all_strikes = set()
            for chain in chains:
                all_strikes.update([c.strike for c in chain.calls])
                all_strikes.update([p.strike for p in chain.puts])

            strikes = sorted(list(all_strikes))
            expirations = [chain.expiration for chain in chains]

            if not strikes or not expirations:
                logger.warning("No strikes or expirations found")
                return

            # Create data matrices for calls and puts
            call_data = np.zeros((len(strikes), len(expirations)))
            put_data = np.zeros((len(strikes), len(expirations)))

            for exp_idx, chain in enumerate(chains):
                for strike_idx, strike in enumerate(strikes):
                    # Find call at this strike
                    matching_calls = [c for c in chain.calls if c.strike == strike]
                    if matching_calls:
                        call = matching_calls[0]
                        if metric == "volume":
                            call_data[strike_idx, exp_idx] = call.volume or 0
                        elif metric == "open_interest":
                            call_data[strike_idx, exp_idx] = call.open_interest or 0
                        elif metric == "implied_volatility":
                            call_data[strike_idx, exp_idx] = (call.implied_volatility or 0) * 100

                    # Find put at this strike
                    matching_puts = [p for p in chain.puts if p.strike == strike]
                    if matching_puts:
                        put = matching_puts[0]
                        if metric == "volume":
                            put_data[strike_idx, exp_idx] = put.volume or 0
                        elif metric == "open_interest":
                            put_data[strike_idx, exp_idx] = put.open_interest or 0
                        elif metric == "implied_volatility":
                            put_data[strike_idx, exp_idx] = (put.implied_volatility or 0) * 100

            # Plot calls heatmap
            im1 = call_ax.imshow(
                call_data, aspect="auto", cmap="YlOrRd", interpolation="nearest"
            )
            call_ax.set_title(f"Calls - {metric.replace('_', ' ').title()}", fontweight="bold")
            call_ax.set_ylabel("Strike Price")
            call_ax.set_xlabel("Expiration")

            # Set ticks
            call_ax.set_yticks(range(0, len(strikes), max(1, len(strikes) // 10)))
            call_ax.set_yticklabels([f"${strikes[i]:.0f}" for i in range(0, len(strikes), max(1, len(strikes) // 10))])
            call_ax.set_xticks(range(len(expirations)))
            call_ax.set_xticklabels([exp.strftime("%m/%d") for exp in expirations], rotation=45)

            plt.colorbar(im1, ax=call_ax)

            # Plot puts heatmap
            im2 = put_ax.imshow(
                put_data, aspect="auto", cmap="YlGnBu", interpolation="nearest"
            )
            put_ax.set_title(f"Puts - {metric.replace('_', ' ').title()}", fontweight="bold")
            put_ax.set_ylabel("Strike Price")
            put_ax.set_xlabel("Expiration")

            # Set ticks
            put_ax.set_yticks(range(0, len(strikes), max(1, len(strikes) // 10)))
            put_ax.set_yticklabels([f"${strikes[i]:.0f}" for i in range(0, len(strikes), max(1, len(strikes) // 10))])
            put_ax.set_xticks(range(len(expirations)))
            put_ax.set_xticklabels([exp.strftime("%m/%d") for exp in expirations], rotation=45)

            plt.colorbar(im2, ax=put_ax)

            # Add spot price line if available
            if chains:
                spot = chains[0].underlying_price
                spot_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
                call_ax.axhline(y=spot_idx, color="black", linestyle="--", linewidth=2, alpha=0.7)
                put_ax.axhline(y=spot_idx, color="black", linestyle="--", linewidth=2, alpha=0.7)

            ticker = chains[0].ticker if chains else "Unknown"
            fig.suptitle(
                f"{ticker} Options Chain - {metric.replace('_', ' ').title()}",
                fontsize=16,
                fontweight="bold",
            )

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path, dpi=OptionsAnalysisParameters.HEATMAP_DPI, bbox_inches="tight"
            )
            logger.debug(f"Created options chain heatmap: {output_path}")

        finally:
            plt.close(fig)
            plt.close("all")

    @staticmethod
    def create_greeks_visualization(
        chain, spot_price: float, output_path: Path
    ) -> None:
        """
        Creates a multi-panel visualization of Greeks vs. underlying price.

        Args:
            chain: OptionsChain with Greeks calculated
            spot_price: Current underlying price
            output_path: Path to save the chart
        """

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"{chain.ticker} - Greeks Sensitivity Analysis", fontsize=16, fontweight="bold")

        try:
            # Get ATM contracts for analysis
            atm_calls = [c for c in chain.calls if c.greeks and abs(c.strike - spot_price) / spot_price < 0.05]
            atm_puts = [p for p in chain.puts if p.greeks and abs(p.strike - spot_price) / spot_price < 0.05]

            if not atm_calls and not atm_puts:
                logger.warning("No contracts with Greeks found for visualization")
                return

            # Use nearest ATM call and put
            call = sorted(atm_calls, key=lambda x: abs(x.strike - spot_price))[0] if atm_calls else None
            put = sorted(atm_puts, key=lambda x: abs(x.strike - spot_price))[0] if atm_puts else None

            # Note: Price range for sensitivity analysis would be:
            # price_range = np.linspace(spot_price * 0.8, spot_price * 1.2, 50)
            # We'll just plot the Greeks at current spot (simplified version)
            # For a full sensitivity analysis, we'd need to recalculate Greeks at each price

            # Panel 1: Delta
            ax_delta = axes[0, 0]
            if call and call.greeks:
                ax_delta.bar(["Call Delta"], [call.greeks.delta], color="green", alpha=0.7)
            if put and put.greeks:
                ax_delta.bar(["Put Delta"], [put.greeks.delta], color="red", alpha=0.7)
            ax_delta.set_ylabel("Delta")
            ax_delta.set_title("Delta (Rate of Change)")
            ax_delta.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax_delta.grid(True, alpha=0.3)

            # Panel 2: Gamma
            ax_gamma = axes[0, 1]
            if call and call.greeks:
                ax_gamma.bar(["Call Gamma"], [call.greeks.gamma], color="blue", alpha=0.7)
            if put and put.greeks:
                ax_gamma.bar(["Put Gamma"], [put.greeks.gamma], color="purple", alpha=0.7)
            ax_gamma.set_ylabel("Gamma")
            ax_gamma.set_title("Gamma (Acceleration)")
            ax_gamma.grid(True, alpha=0.3)

            # Panel 3: Theta
            ax_theta = axes[1, 0]
            if call and call.greeks:
                ax_theta.bar(["Call Theta"], [call.greeks.theta], color="orange", alpha=0.7)
            if put and put.greeks:
                ax_theta.bar(["Put Theta"], [put.greeks.theta], color="brown", alpha=0.7)
            ax_theta.set_ylabel("Theta ($/day)")
            ax_theta.set_title("Theta (Time Decay)")
            ax_theta.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax_theta.grid(True, alpha=0.3)

            # Panel 4: Vega
            ax_vega = axes[1, 1]
            if call and call.greeks:
                ax_vega.bar(["Call Vega"], [call.greeks.vega], color="teal", alpha=0.7)
            if put and put.greeks:
                ax_vega.bar(["Put Vega"], [put.greeks.vega], color="navy", alpha=0.7)
            ax_vega.set_ylabel("Vega ($/1% IV)")
            ax_vega.set_title("Vega (Volatility Sensitivity)")
            ax_vega.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created Greeks visualization: {output_path}")

        finally:
            plt.close(fig)
            plt.close("all")

    @staticmethod
    def create_pnl_diagram(
        strategy, output_path: Path
    ) -> None:
        """
        Creates a profit/loss diagram for an options strategy.

        Args:
            strategy: OptionsStrategy with P&L scenarios calculated
            output_path: Path to save the chart
        """

        if not strategy.pnl_scenarios:
            logger.warning("No P&L scenarios found for strategy")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        try:
            # Extract data from scenarios
            prices = [s.underlying_price for s in strategy.pnl_scenarios]
            pnls = [s.total_pnl for s in strategy.pnl_scenarios]

            # Plot P&L curve
            ax.plot(prices, pnls, linewidth=2.5, color="blue", label="P&L at Expiration")
            ax.fill_between(
                prices, pnls, 0, where=[p >= 0 for p in pnls], alpha=0.3, color="green", label="Profit"
            )
            ax.fill_between(
                prices, pnls, 0, where=[p < 0 for p in pnls], alpha=0.3, color="red", label="Loss"
            )

            # Add breakeven lines
            if strategy.breakeven_points:
                for be in strategy.breakeven_points:
                    ax.axvline(x=be, color="orange", linestyle="--", linewidth=2, alpha=0.7)
                    ax.text(be, max(pnls) * 0.9, f"BE: \\${be:.2f}", rotation=90, va="top")

            # Add max profit/loss annotations
            if strategy.max_profit is not None:
                ax.axhline(
                    y=strategy.max_profit, color="green", linestyle=":", linewidth=1.5, alpha=0.7
                )
                ax.text(
                    min(prices), strategy.max_profit, f"Max Profit: \\${strategy.max_profit:.2f}",
                    va="bottom", ha="left", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7)
                )

            if strategy.max_loss is not None:
                ax.axhline(
                    y=-strategy.max_loss, color="red", linestyle=":", linewidth=1.5, alpha=0.7
                )
                ax.text(
                    min(prices), -strategy.max_loss, f"Max Loss: \\${strategy.max_loss:.2f}",
                    va="top", ha="left", bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7)
                )

            # Zero line
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

            # Current spot price (if available)
            if strategy.legs:
                spot = strategy.legs[0].contract.strike  # Approximation
                ax.axvline(x=spot, color="purple", linestyle="-", linewidth=2, alpha=0.5, label="Current Price")

            ax.set_xlabel("Underlying Price at Expiration", fontsize=12)
            ax.set_ylabel("Profit/Loss ($)", fontsize=12)
            ax.set_title(
                f"P&L Diagram - {strategy.description}\n"
                f"Net Premium: \\${strategy.net_premium:.2f} | "
                f"Capital: \\${strategy.capital_required:.2f}",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created P&L diagram: {output_path}")

        finally:
            plt.close(fig)
            plt.close("all")

    @staticmethod
    def create_iv_surface(
        chains: List, ticker: str, output_path: Path
    ) -> None:
        """
        Creates implied volatility surface/skew visualization.

        Args:
            chains: List of OptionsChain objects
            ticker: Ticker symbol
            output_path: Path to save the chart
        """

        if not chains:
            logger.warning("No chains provided for IV surface")
            return

        fig = plt.figure(figsize=OptionsAnalysisParameters.IV_SURFACE_RESOLUTION)
        ax = fig.add_subplot(111, projection="3d")

        try:
            # Collect data points
            strikes = []
            days_to_exp = []
            ivs = []

            for chain in chains:
                dte = chain.days_to_expiration

                # Collect from calls
                for call in chain.calls:
                    if call.implied_volatility and call.implied_volatility > 0:
                        strikes.append(call.strike)
                        days_to_exp.append(dte)
                        ivs.append(call.implied_volatility * 100)  # Convert to percentage

                # Collect from puts
                for put in chain.puts:
                    if put.implied_volatility and put.implied_volatility > 0:
                        strikes.append(put.strike)
                        days_to_exp.append(dte)
                        ivs.append(put.implied_volatility * 100)

            if not strikes:
                logger.warning("No IV data found for surface plot")
                # Create 2D skew plot instead
                fig, ax2d = plt.subplots(figsize=(10, 6))
                ax2d.text(
                    0.5, 0.5, "Insufficient IV data for surface plot",
                    ha="center", va="center", fontsize=14
                )
                ax2d.set_title(f"{ticker} - IV Surface (Insufficient Data)")
                plt.tight_layout()
                fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
                return

            # Create scatter plot
            scatter = ax.scatter(
                strikes, days_to_exp, ivs, c=ivs, cmap="viridis", s=50, alpha=0.6
            )

            ax.set_xlabel("Strike Price", fontsize=10)
            ax.set_ylabel("Days to Expiration", fontsize=10)
            ax.set_zlabel("Implied Volatility (%)", fontsize=10)
            ax.set_title(f"{ticker} - Implied Volatility Surface", fontsize=14, fontweight="bold")

            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label("IV %", rotation=270, labelpad=15)

            # Adjust viewing angle
            ax.view_init(elev=20, azim=45)

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created IV surface: {output_path}")

        finally:
            plt.close(fig)
            plt.close("all")

    @staticmethod
    def create_iv_skew_2d(
        chain, ticker: str, output_path: Path
    ) -> None:
        """
        Creates a 2D IV skew plot for a single expiration.

        Args:
            chain: OptionsChain object
            ticker: Ticker symbol
            output_path: Path to save the chart
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        try:
            # Collect call IVs
            call_strikes = []
            call_ivs = []
            for call in chain.calls:
                if call.implied_volatility and call.implied_volatility > 0:
                    call_strikes.append(call.strike)
                    call_ivs.append(call.implied_volatility * 100)

            # Collect put IVs
            put_strikes = []
            put_ivs = []
            for put in chain.puts:
                if put.implied_volatility and put.implied_volatility > 0:
                    put_strikes.append(put.strike)
                    put_ivs.append(put.implied_volatility * 100)

            # Plot
            if call_strikes:
                ax.plot(call_strikes, call_ivs, "o-", label="Calls", color="green", linewidth=2, markersize=6)
            if put_strikes:
                ax.plot(put_strikes, put_ivs, "o-", label="Puts", color="red", linewidth=2, markersize=6)

            # Mark ATM
            spot = chain.underlying_price
            ax.axvline(x=spot, color="purple", linestyle="--", linewidth=2, alpha=0.7, label=f"Spot: ${spot:.2f}")

            ax.set_xlabel("Strike Price", fontsize=12)
            ax.set_ylabel("Implied Volatility (%)", fontsize=12)
            ax.set_title(
                f"{ticker} - IV Skew\nExpiration: {chain.expiration.strftime('%Y-%m-%d')} ({chain.days_to_expiration} DTE)",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=ChartConfiguration.CHART_DPI, bbox_inches="tight")
            logger.debug(f"Created IV skew plot: {output_path}")

        finally:
            plt.close(fig)
            plt.close("all")
