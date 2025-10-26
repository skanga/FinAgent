"""
Financial analysis components.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional, Dict, Tuple, Any
from models import (
    AdvancedMetrics,
    ComparativeAnalysis,
    FundamentalData,
    PortfolioMetrics,
    TickerAnalysis,
)
from constants import (
    TechnicalIndicatorParameters,
    TimeConstants,
    LimitsAndConstraints,
    DataValidation,
    Defaults,
)

logger = logging.getLogger(__name__)


class AdvancedFinancialAnalyzer:
    """Comprehensive financial analysis with fundamentals parsing."""

    def __init__(
        self, risk_free_rate: float = 0.02, benchmark_ticker: str = "SPY"
    ) -> None:
        """
        Initializes the AdvancedFinancialAnalyzer with a risk-free rate and a benchmark ticker.

        Args:
            risk_free_rate (float): The risk-free rate to use in calculations.
            benchmark_ticker (str): The ticker of the benchmark to use for comparison.
        """
        self.risk_free_rate = risk_free_rate
        self.benchmark_ticker = benchmark_ticker

    def compute_metrics(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical metrics for a given DataFrame of prices.

        Args:
            df_prices (pd.DataFrame): A DataFrame containing the price history of a ticker.

        Returns:
            pd.DataFrame: The DataFrame with the computed metrics.
        """
        # Initial transformations
        enriched_price_data = (
            df_prices.copy()
            .assign(Date=lambda x: pd.to_datetime(x["Date"], utc=True))
            .sort_values("Date")
            .reset_index(drop=True)
        )

        # Pre-calculate window sizes to avoid multiple len() calls
        n_rows = len(enriched_price_data)

        # Handle edge case of empty DataFrame
        if n_rows == 0:
            # Add required columns with empty values using assign
            return enriched_price_data.assign(
                close=pd.Series(dtype=float),
                daily_return=pd.Series(dtype=float),
                **{
                    col: pd.Series(dtype=float)
                    for col in [
                        "30d_ma",
                        "50d_ma",
                        "200d_ma",
                        "volatility",
                        "rsi",
                        "bollinger_upper",
                        "bollinger_lower",
                        "bollinger_position",
                        "macd",
                        "macd_signal",
                        "atr",
                        "stochastic_k",
                        "stochastic_d",
                        "obv",
                        "vwap",
                    ]
                }
            )

        # Calculate window parameters once
        short_term_ma_window = min(TechnicalIndicatorParameters.MA_SHORT_PERIOD, n_rows)
        long_term_ma_window = min(TechnicalIndicatorParameters.MA_LONG_PERIOD, n_rows)
        very_long_ma_window = min(TechnicalIndicatorParameters.MA_VERY_LONG_PERIOD, n_rows)
        vol_window = min(TechnicalIndicatorParameters.VOLATILITY_WINDOW, max(TimeConstants.MIN_ROWS_FOR_FULL_ANALYSIS, n_rows // TimeConstants.DATA_POINTS_DIVISOR_FOR_VOL_WINDOW))

        # Ensure min_periods doesn't exceed window size
        ma_short_min_periods = min(TimeConstants.MIN_PERIODS_SHORT_MA, short_term_ma_window)
        ma_long_min_periods = min(TimeConstants.MIN_PERIODS_LONG_MA, long_term_ma_window)
        ma_very_long_min_periods = min(TimeConstants.MIN_PERIODS_LONG_MA, very_long_ma_window)
        vol_min_periods = min(TimeConstants.MIN_PERIODS_VOLATILITY, vol_window)

        # Pre-calculate price series (used multiple times)
        close_prices = enriched_price_data["Close"].astype(float)

        # Check if OHLCV data is available for advanced indicators
        has_high_low = "High" in enriched_price_data.columns and "Low" in enriched_price_data.columns
        has_volume = "Volume" in enriched_price_data.columns

        if has_high_low:
            high_prices = enriched_price_data["High"].astype(float)
            low_prices = enriched_price_data["Low"].astype(float)
        else:
            # Create synthetic High/Low from Close if not available
            high_prices = close_prices
            low_prices = close_prices

        if has_volume:
            volume = enriched_price_data["Volume"].astype(float)
        else:
            # Create synthetic volume if not available
            volume = pd.Series([1000000] * len(close_prices), index=close_prices.index, dtype=float)

        # Calculate technical indicators that don't depend on other computed columns
        rsi_values = self._compute_rsi(close_prices)
        bollinger_upper, bollinger_lower = self._compute_bollinger_bands(close_prices)
        macd_values, macd_signal_values = self._compute_macd(close_prices)

        # Only calculate OHLCV-dependent indicators if data is available
        atr_values = self._compute_atr(high_prices, low_prices, close_prices) if has_high_low else pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        stoch_k, stoch_d = (self._compute_stochastic(high_prices, low_prices, close_prices) if has_high_low
                            else (pd.Series([np.nan] * len(close_prices), index=close_prices.index),
                                  pd.Series([np.nan] * len(close_prices), index=close_prices.index)))
        obv_values = self._compute_obv(close_prices, volume) if has_volume else pd.Series([np.nan] * len(close_prices), index=close_prices.index)
        # VWAP removed: Not appropriate for daily data (intraday indicator only)

        # Single pass: add all computed columns using assign()
        enriched_price_data = enriched_price_data.assign(
            close=close_prices,
            daily_return=lambda x: x["close"].pct_change(fill_method=None),
            **{
                "30d_ma": close_prices.rolling(
                    window=short_term_ma_window, min_periods=ma_short_min_periods
                ).mean(),
                "50d_ma": close_prices.rolling(
                    window=long_term_ma_window, min_periods=ma_long_min_periods
                ).mean(),
                "200d_ma": close_prices.rolling(
                    window=very_long_ma_window, min_periods=ma_very_long_min_periods
                ).mean(),
                "rsi": rsi_values,
                "bollinger_upper": bollinger_upper,
                "bollinger_lower": bollinger_lower,
                "macd": macd_values,
                "macd_signal": macd_signal_values,
                "atr": atr_values,
                "stochastic_k": stoch_k,
                "stochastic_d": stoch_d,
                "obv": obv_values,
                # "vwap": removed - not appropriate for daily data
            }
        )

        # Add volatility (depends on daily_return) and bollinger_position (depends on bollinger bands)
        enriched_price_data = enriched_price_data.assign(
            volatility=lambda x: x["daily_return"].rolling(
                window=vol_window, min_periods=vol_min_periods
            ).std(),
            bollinger_position=lambda x: np.where(
                (x["bollinger_upper"] - x["bollinger_lower"]) > 0,
                (x["close"] - x["bollinger_lower"]) / (x["bollinger_upper"] - x["bollinger_lower"]),
                np.nan,
            )
        )

        return enriched_price_data

    def _compute_rsi(self, prices: pd.Series, period: int = TechnicalIndicatorParameters.RSI_PERIOD) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Handle division by zero edge cases
        # When loss = 0 (all gains), RS = infinity → RSI = 100
        # When gain = 0 (all losses), RS = 0 → RSI = 0
        # When both = 0 (no movement), RS = 1 → RSI = 50 (neutral)
        import numpy as np
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.where(loss > 0, gain / loss, np.where(gain > 0, np.inf, 1.0))
            rsi = 100 - (100 / (1 + rs))

        # Clip to valid RSI range [0, 100]
        return pd.Series(np.clip(rsi, 0, 100), index=prices.index)

    def _compute_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = TechnicalIndicatorParameters.BOLLINGER_PERIOD,
        std_dev: int = TechnicalIndicatorParameters.BOLLINGER_STD_DEV,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std * std_dev), sma - (std * std_dev)

    def _compute_macd(
        self,
        prices: pd.Series,
        fast: int = TechnicalIndicatorParameters.MACD_FAST_PERIOD,
        slow: int = TechnicalIndicatorParameters.MACD_SLOW_PERIOD,
        signal: int = TechnicalIndicatorParameters.MACD_SIGNAL_PERIOD,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd, macd.ewm(span=signal).mean()

    def _compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = TechnicalIndicatorParameters.ATR_PERIOD,
    ) -> pd.Series:
        """
        Compute Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range of an asset
        for that period. Higher ATR values indicate higher volatility.
        """
        # True Range is the greatest of:
        # 1. Current High - Current Low
        # 2. |Current High - Previous Close|
        # 3. |Current Low - Previous Close|
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _compute_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = TechnicalIndicatorParameters.STOCHASTIC_K_PERIOD,
        d_period: int = TechnicalIndicatorParameters.STOCHASTIC_D_PERIOD,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute Stochastic Oscillator (%K and %D).

        Stochastic Oscillator compares a closing price to its price range over time.
        %K is the fast line, %D is the slow line (signal).
        Values range from 0-100: >80 = overbought, <20 = oversold.
        """
        # %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        # Handle division by zero when highest_high == lowest_low (flat prices)
        # In this case, price is neither high nor low, so use neutral value of 50
        import numpy as np
        denominator = highest_high - lowest_low
        stoch_k = np.where(
            denominator > 0,
            100 * (close - lowest_low) / denominator,
            50.0  # Neutral value when range is zero (flat prices)
        )
        stoch_k = pd.Series(stoch_k, index=close.index)

        # %D is the moving average of %K
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def _compute_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Compute On-Balance Volume (OBV) using vectorized operations.

        OBV uses volume flow to predict changes in stock price.
        Rising OBV suggests accumulation, falling OBV suggests distribution.

        Vectorized implementation is ~100x faster than the previous loop-based version.
        """
        import numpy as np

        # Calculate price changes (NaN for first element)
        price_diff = close.diff()

        # Create signed volume based on price movement
        # +volume if price increased, -volume if price decreased, 0 if unchanged
        signed_volume = np.where(price_diff > 0, volume,
                                 np.where(price_diff < 0, -volume, 0))

        # Convert to Series for easier manipulation
        signed_volume_series = pd.Series(signed_volume, index=close.index)

        # Set first value to 0 so cumsum starts correctly
        signed_volume_series.iloc[0] = 0

        # Cumulative sum of signed volume changes, plus initial volume
        obv = signed_volume_series.cumsum() + volume.iloc[0]

        return obv

    # _compute_vwap method removed: VWAP is an intraday indicator designed for
    # minute/tick data within a single trading day. It's not appropriate for
    # daily OHLCV data. The previous implementation created a misleading
    # "cumulative VWAP" across the entire time period.

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> Optional[float]:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

        CVaR is the expected loss given that the loss exceeds VaR.
        It's a better tail risk measure than VaR as it considers the magnitude of extreme losses.

        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 0.95 for 95%)

        Returns:
            CVaR as a decimal (negative value indicates loss)
        """
        if len(returns) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None

        returns_clean = returns.dropna()
        if len(returns_clean) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None

        # Calculate VaR threshold
        var_threshold = np.percentile(returns_clean, (1 - confidence_level) * 100)

        # CVaR is the mean of returns below the VaR threshold
        tail_losses = returns_clean[returns_clean <= var_threshold]

        if len(tail_losses) > 0:
            return float(tail_losses.mean())

        return None

    def _calculate_drawdown_duration(self, returns: pd.Series) -> Optional[int]:
        """
        Calculate maximum drawdown duration (time to recover).

        Returns the number of trading days from peak to recovery for the maximum drawdown.

        Args:
            returns: Series of returns

        Returns:
            Number of days, or None if no recovery or insufficient data
        """
        if len(returns) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None

        returns_clean = returns.dropna()
        if len(returns_clean) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None

        # Calculate cumulative returns
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Find the maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()

        if pd.isna(max_dd_value) or max_dd_value == 0:
            return None

        # Find when the drawdown started (most recent peak before max DD)
        drawdown_start = None
        for i in range(len(drawdown)):
            if drawdown.index[i] >= max_dd_idx:
                break
            if drawdown.iloc[i] == 0:  # At a peak
                drawdown_start = drawdown.index[i]

        if drawdown_start is None:
            drawdown_start = drawdown.index[0]

        # Find recovery point (next time drawdown returns to 0 after max DD)
        recovery_idx = None
        for i in range(len(drawdown)):
            if drawdown.index[i] > max_dd_idx and drawdown.iloc[i] >= 0:
                recovery_idx = drawdown.index[i]
                break

        if recovery_idx is None:
            # Still in drawdown - no recovery yet
            return None

        # Calculate duration in trading days
        duration = len(returns_clean.loc[drawdown_start:recovery_idx]) - 1
        return int(duration) if duration > 0 else None

    def _calculate_capture_ratios(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Up Capture and Down Capture ratios vs benchmark.

        Up Capture: How much of the benchmark's gains the portfolio captures
        Down Capture: How much of the benchmark's losses the portfolio captures

        Args:
            returns: Portfolio/ticker returns
            benchmark_returns: Benchmark returns

        Returns:
            Tuple of (up_capture, down_capture) as ratios (1.0 = 100%)
        """
        if benchmark_returns is None or len(benchmark_returns) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None, None

        # Align returns
        benchmark_clean = benchmark_returns.dropna()
        common_index = returns.index.intersection(benchmark_clean.index)

        if len(common_index) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None, None

        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_clean.loc[common_index]

        # Clean any remaining NaN
        if aligned_returns.isna().any() or aligned_benchmark.isna().any():
            aligned_returns = aligned_returns.dropna()
            aligned_benchmark = aligned_benchmark.dropna()
            common_idx = aligned_returns.index.intersection(aligned_benchmark.index)
            if len(common_idx) < TimeConstants.MIN_DATA_POINTS_BASIC:
                return None, None
            aligned_returns = aligned_returns.loc[common_idx]
            aligned_benchmark = aligned_benchmark.loc[common_idx]

        # Separate up and down periods
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0

        # Calculate capture ratios
        up_capture = None
        if up_periods.sum() > 0:
            portfolio_up_return = aligned_returns[up_periods].mean()
            benchmark_up_return = aligned_benchmark[up_periods].mean()
            if benchmark_up_return != 0:
                up_capture = float(portfolio_up_return / benchmark_up_return)

        down_capture = None
        if down_periods.sum() > 0:
            portfolio_down_return = aligned_returns[down_periods].mean()
            benchmark_down_return = aligned_benchmark[down_periods].mean()
            if benchmark_down_return != 0:
                down_capture = float(portfolio_down_return / benchmark_down_return)

        return up_capture, down_capture

    def _calculate_rolling_metrics(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series], window: int = 60
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Calculate rolling Sharpe and Beta statistics.

        Args:
            returns: Portfolio/ticker returns
            benchmark_returns: Benchmark returns (needed for rolling beta)
            window: Rolling window size in days (default 60)

        Returns:
            Tuple of (rolling_sharpe_mean, rolling_sharpe_std, rolling_beta_mean, rolling_beta_std)
        """
        if len(returns) < window:
            return None, None, None, None

        returns_clean = returns.dropna()
        if len(returns_clean) < window:
            return None, None, None, None

        # Calculate rolling Sharpe with NaN/inf filtering
        # Use geometric compounding for rolling returns
        rolling_mean = returns_clean.rolling(window=window).mean()
        rolling_return = (1 + rolling_mean) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
        rolling_std = returns_clean.rolling(window=window).std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)

        # Avoid division by zero and filter out invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_std

        # Filter out NaN, inf, and -inf values
        rolling_sharpe_clean = rolling_sharpe.replace([np.inf, -np.inf], np.nan).dropna()

        sharpe_mean = None
        sharpe_std = None
        if len(rolling_sharpe_clean) > 0:
            # Use nanmean and nanstd to handle any remaining NaN values
            mean_val = np.nanmean(rolling_sharpe_clean)
            std_val = np.nanstd(rolling_sharpe_clean)

            # Only return if values are finite
            if np.isfinite(mean_val):
                sharpe_mean = float(mean_val)
            if np.isfinite(std_val):
                sharpe_std = float(std_val)

        # Calculate rolling Beta if benchmark available
        beta_mean, beta_std = None, None
        if benchmark_returns is not None and len(benchmark_returns) >= window:
            benchmark_clean = benchmark_returns.dropna()
            common_index = returns_clean.index.intersection(benchmark_clean.index)

            if len(common_index) >= window:
                aligned_returns = returns_clean.loc[common_index]
                aligned_benchmark = benchmark_clean.loc[common_index]

                # Calculate rolling covariance and variance with error handling
                with np.errstate(divide='ignore', invalid='ignore'):
                    rolling_cov = aligned_returns.rolling(window=window).cov(aligned_benchmark)
                    rolling_var = aligned_benchmark.rolling(window=window).var()

                    # Avoid division by zero
                    rolling_beta = rolling_cov / rolling_var

                # Filter out NaN, inf, and -inf values
                rolling_beta_clean = rolling_beta.replace([np.inf, -np.inf], np.nan).dropna()

                if len(rolling_beta_clean) > 0:
                    mean_val = np.nanmean(rolling_beta_clean)
                    std_val = np.nanstd(rolling_beta_clean)

                    # Only return if values are finite
                    if np.isfinite(mean_val):
                        beta_mean = float(mean_val)
                    if np.isfinite(std_val):
                        beta_std = float(std_val)

        return sharpe_mean, sharpe_std, beta_mean, beta_std

    def calculate_advanced_metrics(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None
    ) -> AdvancedMetrics:
        """
        Calculates advanced financial metrics for a given series of returns.

        Args:
            returns (pd.Series): A series of returns.
            benchmark_returns (Optional[pd.Series]): A series of benchmark returns.

        Returns:
            AdvancedMetrics: An object containing the advanced metrics.
        """
        if len(returns) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return AdvancedMetrics()

        returns_clean = returns.dropna()
        if len(returns_clean) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return AdvancedMetrics()

        # Calculate annualized metrics from daily returns
        # Annualized return uses geometric compounding: (1 + mean_daily_return)^252 - 1
        # Daily std dev × √252 = annualized volatility
        mean_daily_return = returns_clean.mean()
        annualized_return = (1 + mean_daily_return) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
        annualized_vol = returns_clean.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)

        # Sharpe Ratio = (Annualized Return - Risk Free Rate) / Annualized Volatility
        # This is ALREADY the annualized Sharpe ratio (no further scaling needed)
        sharpe = (
            (annualized_return - self.risk_free_rate) / annualized_vol
            if annualized_vol > 0
            else None
        )

        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() if not drawdown.empty else None

        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = (
            downside_returns.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)
            if len(downside_returns) > 0
            else LimitsAndConstraints.MIN_DOWNSIDE_DEVIATION
        )
        sortino = (
            (annualized_return - self.risk_free_rate) / downside_deviation
            if downside_deviation > 0
            else None
        )

        calmar = annualized_return / abs(max_dd) if max_dd and max_dd != 0 else None
        var_95 = np.percentile(returns_clean, 5)

        beta, alpha, r_squared, treynor, information_ratio = (
            self._calculate_benchmark_metrics(returns_clean, benchmark_returns)
        )

        # Calculate new portfolio indicators
        cvar_95 = self._calculate_cvar(returns_clean, confidence_level=0.95)
        max_dd_duration = self._calculate_drawdown_duration(returns_clean)
        up_capture, down_capture = self._calculate_capture_ratios(returns_clean, benchmark_returns)
        rolling_sharpe_mean, rolling_sharpe_std, rolling_beta_mean, rolling_beta_std = (
            self._calculate_rolling_metrics(returns_clean, benchmark_returns, window=60)
        )

        return AdvancedMetrics(
            sharpe_ratio=(
                float(sharpe) if sharpe is not None and not pd.isna(sharpe) else None
            ),
            max_drawdown=(
                float(max_dd) if max_dd is not None and not pd.isna(max_dd) else None
            ),
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            var_95=float(var_95) if not pd.isna(var_95) else None,
            sortino_ratio=(
                float(sortino) if sortino is not None and not pd.isna(sortino) else None
            ),
            calmar_ratio=(
                float(calmar) if calmar is not None and not pd.isna(calmar) else None
            ),
            treynor_ratio=treynor,
            information_ratio=information_ratio,
            cvar_95=cvar_95,
            max_drawdown_duration=max_dd_duration,
            up_capture=up_capture,
            down_capture=down_capture,
            rolling_sharpe_mean=rolling_sharpe_mean,
            rolling_sharpe_std=rolling_sharpe_std,
            rolling_beta_mean=rolling_beta_mean,
            rolling_beta_std=rolling_beta_std,
        )

    def _calculate_benchmark_metrics(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate metrics relative to benchmark."""
        if benchmark_returns is None or len(benchmark_returns) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None, None, None, None, None

        benchmark_clean = benchmark_returns.dropna()
        common_index = returns.index.intersection(benchmark_clean.index)

        if len(common_index) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None, None, None, None, None

        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_clean.loc[common_index]

        # Validate we have sufficient data after alignment
        if (
            len(aligned_returns) < TimeConstants.MIN_DATA_POINTS_BASIC
            or len(aligned_benchmark) < TimeConstants.MIN_DATA_POINTS_BASIC
        ):
            return None, None, None, None, None

        # Ensure no NaN/inf values that could break calculations
        if aligned_returns.isna().any() or aligned_benchmark.isna().any():
            aligned_returns = aligned_returns.dropna()
            aligned_benchmark = aligned_benchmark.dropna()
            # Re-align after dropping NaN
            common_idx = aligned_returns.index.intersection(aligned_benchmark.index)
            if len(common_idx) < TimeConstants.MIN_DATA_POINTS_BASIC:
                return None, None, None, None, None
            aligned_returns = aligned_returns.loc[common_idx]
            aligned_benchmark = aligned_benchmark.loc[common_idx]

        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        # Use threshold for near-zero variance to avoid division issues
        beta = (
            covariance / benchmark_variance
            if benchmark_variance > LimitsAndConstraints.NEAR_ZERO_VARIANCE_THRESHOLD
            and not np.isnan(benchmark_variance)
            else None
        )

        alpha = None
        if beta is not None:
            # Use geometric compounding for annualized returns
            benchmark_return = (1 + aligned_benchmark.mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
            ticker_return = (1 + aligned_returns.mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
            alpha = (ticker_return - self.risk_free_rate) - beta * (benchmark_return - self.risk_free_rate)

        # Calculate r_squared, handling zero standard deviation
        ticker_std = aligned_returns.std()
        benchmark_std = aligned_benchmark.std()
        if ticker_std > 0 and benchmark_std > 0 and len(aligned_returns) > 1:
            with np.errstate(invalid='ignore', divide='ignore'):
                corr_matrix = np.corrcoef(aligned_returns, aligned_benchmark)
                corr = corr_matrix[0, 1]
                r_squared = float(corr ** 2) if not np.isnan(corr) else None
        else:
            r_squared = None
        # Treynor ratio uses geometric compounding for returns
        treynor = None
        if beta and beta != 0:
            ticker_return_treynor = (1 + aligned_returns.mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
            treynor = (ticker_return_treynor - self.risk_free_rate) / beta

        active_returns = aligned_returns - aligned_benchmark
        # Information ratio: annualized active return / annualized tracking error
        active_mean = active_returns.mean()
        active_return_annualized = (1 + active_mean) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
        information_ratio = (
            active_return_annualized
            / (active_returns.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR))
            if active_returns.std() > 0
            else None
        )

        return beta, alpha, r_squared, treynor, information_ratio

    def create_comparative_analysis(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series]
    ) -> Optional[ComparativeAnalysis]:
        """
        Create comparative analysis against benchmark.

        Args:
            returns: Series of daily returns for the ticker
            benchmark_returns: Series of daily returns for the benchmark

        Returns:
            ComparativeAnalysis object or None if benchmark unavailable
        """
        if benchmark_returns is None or len(benchmark_returns) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None

        benchmark_clean = benchmark_returns.dropna()
        common_index = returns.index.intersection(benchmark_clean.index)

        if len(common_index) < TimeConstants.MIN_DATA_POINTS_BASIC:
            return None

        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_clean.loc[common_index]

        # Validate sufficient data after alignment
        if (
            len(aligned_returns) < TimeConstants.MIN_DATA_POINTS_BASIC
            or len(aligned_benchmark) < TimeConstants.MIN_DATA_POINTS_BASIC
        ):
            return None

        # Clean NaN/inf values
        if aligned_returns.isna().any() or aligned_benchmark.isna().any():
            aligned_returns = aligned_returns.dropna()
            aligned_benchmark = aligned_benchmark.dropna()
            common_idx = aligned_returns.index.intersection(aligned_benchmark.index)
            if len(common_idx) < TimeConstants.MIN_DATA_POINTS_BASIC:
                return None
            aligned_returns = aligned_returns.loc[common_idx]
            aligned_benchmark = aligned_benchmark.loc[common_idx]

        # Calculate comparative metrics
        # 1. Beta (systematic risk)
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = (
            float(covariance / benchmark_variance)
            if benchmark_variance > LimitsAndConstraints.NEAR_ZERO_VARIANCE_THRESHOLD
            and not np.isnan(benchmark_variance)
            else None
        )

        # 2. Alpha (excess return vs expected) - use geometric compounding
        alpha = None
        if beta is not None:
            benchmark_return = (1 + aligned_benchmark.mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
            ticker_return = (1 + aligned_returns.mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
            alpha = float(
                (ticker_return - self.risk_free_rate)
                - beta * (benchmark_return - self.risk_free_rate)
            )

        # 3. Correlation (linear relationship strength)
        # Handle zero standard deviation case to avoid warnings
        ticker_std = aligned_returns.std()
        benchmark_std = aligned_benchmark.std()
        if ticker_std > 0 and benchmark_std > 0 and len(aligned_returns) > 1:
            # Suppress warnings for correlation calculation
            with np.errstate(invalid='ignore', divide='ignore'):
                corr_matrix = np.corrcoef(aligned_returns, aligned_benchmark)
                correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else None
        else:
            # If either series has zero variance, correlation is undefined
            correlation = None

        # 4. Tracking error (volatility of excess returns)
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = (
            float(active_returns.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR))
            if len(active_returns) > 1 and active_returns.std() > 0
            else None
        )

        # 5. Information ratio (risk-adjusted active return) - use geometric compounding
        active_return_annualized = (1 + active_returns.mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
        information_ratio = (
            float(
                active_return_annualized
                / (active_returns.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR))
            )
            if active_returns.std() > 0
            else None
        )

        # 6. Outperformance (cumulative excess return)
        ticker_cumulative = (1 + aligned_returns).prod() - 1
        benchmark_cumulative = (1 + aligned_benchmark).prod() - 1
        outperformance = float(ticker_cumulative - benchmark_cumulative)

        # 7. Relative volatility (volatility ratio)
        ticker_vol = aligned_returns.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)
        benchmark_vol = aligned_benchmark.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)
        relative_volatility = (
            float(ticker_vol / benchmark_vol)
            if benchmark_vol > LimitsAndConstraints.NEAR_ZERO_VARIANCE_THRESHOLD
            else None
        )

        return ComparativeAnalysis(
            outperformance=outperformance,
            correlation=correlation,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            beta_vs_benchmark=beta,
            alpha_vs_benchmark=alpha,
            relative_volatility=relative_volatility,
        )

    def compute_ratios(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Computes financial ratios for a given ticker using yfinance.

        Args:
            ticker (str): The ticker to compute the ratios for.

        Returns:
            Dict[str, Optional[float]]: A dictionary of financial ratios.
        """
        ratios = {
            "pe_ratio": None,
            "forward_pe": None,
            "peg_ratio": None,
            "price_to_sales": None,
            "price_to_book": None,
            "debt_to_equity": None,
            "current_ratio": None,
            "quick_ratio": None,
            "return_on_equity": None,
            "return_on_assets": None,
            "profit_margin": None,
            "operating_margin": None,
            "gross_margin": None,
            "dividend_yield": None,
            "beta": None,
            "ev_ebitda": None,  # Enterprise Value / EBITDA
            "fcf_yield": None,  # Free Cash Flow Yield
            "interest_coverage": None,  # Interest Coverage Ratio
        }

        try:
            ticker_obj = yf.Ticker(ticker)
            ticker_fundamental_info = ticker_obj.info

            if not ticker_fundamental_info:
                return ratios

            ratio_mapping = {
                "pe_ratio": "trailingPE",
                "forward_pe": "forwardPE",
                "peg_ratio": "pegRatio",
                "price_to_sales": "priceToSalesTrailing12Months",
                "price_to_book": "priceToBook",
                "debt_to_equity": "debtToEquity",
                "current_ratio": "currentRatio",
                "quick_ratio": "quickRatio",
                "return_on_equity": "returnOnEquity",
                "return_on_assets": "returnOnAssets",
                "profit_margin": "profitMargins",
                "operating_margin": "operatingMargins",
                "gross_margin": "grossMargins",
                "dividend_yield": "dividendYield",
                "beta": "beta",
            }

            for ratio_key, info_key in ratio_mapping.items():
                if (
                    info_key in ticker_fundamental_info
                    and ticker_fundamental_info[info_key] is not None
                ):
                    ratios[ratio_key] = float(ticker_fundamental_info[info_key])

            # Calculate EV/EBITDA from yfinance info
            enterprise_value = ticker_fundamental_info.get("enterpriseValue")
            ebitda = ticker_fundamental_info.get("ebitda")
            if enterprise_value and ebitda and ebitda > 0:
                ratios["ev_ebitda"] = float(enterprise_value / ebitda)

            # Calculate Free Cash Flow Yield
            free_cash_flow = ticker_fundamental_info.get("freeCashflow")
            market_cap = ticker_fundamental_info.get("marketCap")
            if free_cash_flow and market_cap and market_cap > 0:
                ratios["fcf_yield"] = float(free_cash_flow / market_cap)

            # Calculate Interest Coverage (EBIT / Interest Expense)
            # Note: yfinance provides this directly, but we can also calculate from financial statements
            interest_coverage = ticker_fundamental_info.get("interestCoverage")
            if interest_coverage:
                ratios["interest_coverage"] = float(interest_coverage)
            else:
                # Try to calculate from EBIT and interest expense
                ebit = ticker_fundamental_info.get("ebit")
                interest_expense = ticker_fundamental_info.get("interestExpense")
                if ebit and interest_expense and interest_expense != 0:
                    ratios["interest_coverage"] = float(ebit / abs(interest_expense))

            logger.debug(
                f"Computed {sum(1 for v in ratios.values() if v is not None)} ratios for {ticker}"
            )

        except (KeyError, ValueError, TypeError, AttributeError, Exception) as e:
            logger.debug(f"Could not compute ratios for {ticker}: {e}")

        return ratios

    def parse_fundamentals(self, ticker: str) -> FundamentalData:
        """
        Parses fundamental financial data for a given ticker.

        Args:
            ticker (str): The ticker to parse the fundamentals for.

        Returns:
            FundamentalData: An object containing the fundamental data.
        """
        fundamentals = FundamentalData()

        try:
            t = yf.Ticker(ticker)
            income_stmt = t.quarterly_income_stmt
            balance_sheet = t.quarterly_balance_sheet
            cash_flow = t.quarterly_cashflow

            # Parse income statement
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]
                fundamentals.revenue = self._safe_get_value(
                    latest_income, "Total Revenue"
                )
                fundamentals.net_income = self._safe_get_value(
                    latest_income, "Net Income"
                )
                fundamentals.gross_profit = self._safe_get_value(
                    latest_income, "Gross Profit"
                )
                fundamentals.operating_income = self._safe_get_value(
                    latest_income, "Operating Income"
                )
                fundamentals.ebitda = self._safe_get_value(latest_income, "EBITDA")
                fundamentals.interest_expense = self._safe_get_value(
                    latest_income, "Interest Expense"
                )

                # Calculate growth rates (YoY) using helper method
                fundamentals.revenue_growth = self._calculate_yoy_growth(
                    income_stmt, "Total Revenue"
                )
                fundamentals.earnings_growth = self._calculate_yoy_growth(
                    income_stmt, "Net Income"
                )

                # Calculate CAGR (3-year and 5-year)
                fundamentals.revenue_cagr_3y = self._calculate_cagr(
                    income_stmt, "Total Revenue", 3
                )
                fundamentals.revenue_cagr_5y = self._calculate_cagr(
                    income_stmt, "Total Revenue", 5
                )
                fundamentals.earnings_cagr_3y = self._calculate_cagr(
                    income_stmt, "Net Income", 3
                )
                fundamentals.earnings_cagr_5y = self._calculate_cagr(
                    income_stmt, "Net Income", 5
                )

            # Parse balance sheet
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                fundamentals.total_assets = self._safe_get_value(
                    latest_balance, "Total Assets"
                )
                fundamentals.total_liabilities = self._safe_get_value(
                    latest_balance, "Total Liabilities Net Minority Interest"
                )
                fundamentals.shareholders_equity = self._safe_get_value(
                    latest_balance, "Stockholders Equity"
                )

            # Parse cash flow
            if cash_flow is not None and not cash_flow.empty:
                latest_cf = cash_flow.iloc[:, 0]
                fundamentals.operating_cash_flow = self._safe_get_value(
                    latest_cf, "Operating Cash Flow"
                )
                fundamentals.free_cash_flow = self._safe_get_value(
                    latest_cf, "Free Cash Flow"
                )

            non_none_count = sum(
                1 for field in asdict(fundamentals).values() if field is not None
            )
            logger.debug(f"Parsed {non_none_count} fundamental metrics for {ticker}")

        except (
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            IndexError,
            Exception,
        ) as e:
            logger.debug(f"Could not parse fundamentals for {ticker}: {e}")

        return fundamentals

    def _safe_get_value(self, series: pd.Series, key: str) -> Optional[float]:
        """
        Safely extracts a float value from a pandas Series by key.

        Args:
            series (pd.Series): The pandas Series to extract the value from.
            key (str): The key of the value to extract.

        Returns:
            Optional[float]: The extracted value as a float, or None if the key is not found or the value is not a valid number.
        """
        try:
            if key in series.index:
                value = series[key]
                if pd.notna(value):
                    return float(value)
        except (KeyError, ValueError, TypeError):
            pass
        return None

    def _calculate_yoy_growth(
        self,
        dataframe: pd.DataFrame,
        metric_key: str,
        current_col: int = DataValidation.CURRENT_PERIOD_INDEX,
        yoy_col: int = DataValidation.YOY_QUARTERS_LOOKBACK,
    ) -> Optional[float]:
        """
        Calculate year-over-year growth rate for a given metric.

        Args:
            dataframe: Financial statement DataFrame (income_stmt, balance_sheet, etc.)
            metric_key: Key to extract from the DataFrame (e.g., 'Total Revenue', 'Net Income')
            current_col: Column index for current period (default: 0 = most recent)
            yoy_col: Column index for year-ago period (default: 4 = 4 quarters ago)

        Returns:
            Growth rate as a decimal (e.g., 0.15 for 15% growth), or None if cannot calculate
        """
        if dataframe is None or dataframe.empty:
            return None

        if len(dataframe.columns) < DataValidation.MIN_QUARTERS_FOR_GROWTH:
            return None

        current_value = self._safe_get_value(dataframe.iloc[:, current_col], metric_key)
        yoy_value = self._safe_get_value(dataframe.iloc[:, yoy_col], metric_key)

        if current_value and yoy_value and yoy_value != 0:
            return (current_value - yoy_value) / yoy_value

        return None

    def _calculate_cagr(
        self,
        dataframe: pd.DataFrame,
        metric_key: str,
        years: int,
    ) -> Optional[float]:
        """
        Calculate Compound Annual Growth Rate (CAGR) for a given metric.

        CAGR = (Ending Value / Beginning Value) ^ (1 / Years) - 1

        Args:
            dataframe: Financial statement DataFrame (quarterly data)
            metric_key: Key to extract from the DataFrame (e.g., 'Total Revenue', 'Net Income')
            years: Number of years for CAGR calculation (3 or 5)

        Returns:
            CAGR as a decimal (e.g., 0.12 for 12% annual growth), or None if cannot calculate
        """
        if dataframe is None or dataframe.empty:
            return None

        # Need years * 4 quarters of data + 1 for the current period
        required_quarters = years * 4 + 1
        if len(dataframe.columns) < required_quarters:
            return None

        # Current value (most recent quarter)
        current_value = self._safe_get_value(dataframe.iloc[:, 0], metric_key)
        # Value from 'years' ago (years * 4 quarters back)
        start_value = self._safe_get_value(dataframe.iloc[:, years * 4], metric_key)

        if current_value and start_value and start_value > 0 and current_value > 0:
            # CAGR formula: (ending/beginning)^(1/years) - 1
            cagr = (current_value / start_value) ** (1 / years) - 1
            return float(cagr)

        return None


class PortfolioAnalyzer:
    """Portfolio-level metrics computation with LRU cache."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        cache_max_size: int = Defaults.PORTFOLIO_CACHE_MAX_SIZE,
    ) -> None:
        """
        Initializes the PortfolioAnalyzer with a risk-free rate and bounded cache.

        Args:
            risk_free_rate (float): The risk-free rate to use in calculations.
            cache_max_size (int): Maximum number of portfolio combinations to cache.
                When exceeded, least recently used entries are evicted.
        """
        self.risk_free_rate = risk_free_rate
        self._cache_max_size = cache_max_size
        # Use OrderedDict for LRU cache (preserves insertion order)
        self._returns_cache: OrderedDict[frozenset, pd.DataFrame] = OrderedDict()

    def calculate_portfolio_metrics(
        self,
        analyses: Dict[str, TickerAnalysis],
        weights: Optional[Dict[str, float]] = None,
    ) -> PortfolioMetrics:
        """
        Calculates portfolio-level metrics based on a dictionary of ticker analyses and weights.

        Args:
            analyses (Dict[str, TickerAnalysis]): A dictionary of ticker analyses.
            weights (Optional[Dict[str, float]]): A dictionary of weights for each ticker.

        Returns:
            PortfolioMetrics: An object containing the portfolio metrics.
        """
        # Validate and prepare data
        successful = self._validate_analyses(analyses)
        weights = self._validate_and_normalize_weights(successful, weights)

        # Load returns data
        returns_df = self._load_returns_data(successful)

        # Fall back to simple metrics if insufficient data
        if returns_df is None or len(returns_df) < TimeConstants.MIN_DATA_POINTS_PORTFOLIO:
            return self._simple_portfolio_metrics(successful, weights)

        # Calculate portfolio returns time series
        portfolio_returns_series = self._calculate_portfolio_returns(returns_df, weights)

        # Calculate various metric groups
        basic_metrics = self._calculate_basic_portfolio_metrics(portfolio_returns_series)
        risk_metrics = self._calculate_portfolio_risk_metrics(portfolio_returns_series)
        diversification_metrics = self._calculate_diversification_metrics(
            returns_df, weights, basic_metrics["portfolio_volatility"]
        )
        contribution_metrics = self._calculate_contribution_metrics(
            returns_df, weights, successful
        )

        # Log results
        logger.info(
            f"Portfolio: Return={basic_metrics['portfolio_return']*100:.2f}%, "
            f"Vol={basic_metrics['portfolio_volatility']*100:.2f}%, "
            f"Sharpe={basic_metrics['portfolio_sharpe']:.2f}"
        )

        # Combine all metrics
        return PortfolioMetrics(
            **basic_metrics,
            **risk_metrics,
            **diversification_metrics,
            **contribution_metrics,
        )

    def _validate_analyses(
        self, analyses: Dict[str, TickerAnalysis]
    ) -> Dict[str, TickerAnalysis]:
        """Validate analyses and return only successful ones.

        Args:
            analyses: All ticker analyses

        Returns:
            Dictionary of successful analyses only

        Raises:
            ValueError: If no successful analyses
        """
        successful = {t: a for t, a in analyses.items() if not a.error}
        if not successful:
            raise ValueError("No successful analyses for portfolio calculation")
        return successful

    def _validate_and_normalize_weights(
        self,
        successful: Dict[str, TickerAnalysis],
        weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Validate and normalize portfolio weights.

        Args:
            successful: Successful analyses
            weights: Optional weights dictionary

        Returns:
            Validated and normalized weights

        Raises:
            ValueError: If weights are invalid
        """
        # Default to equal weights
        if weights is None:
            n = len(successful)
            weights = {ticker: 1.0 / n for ticker in successful.keys()}

        # Validate non-negative
        if any(w < 0 for w in weights.values()):
            raise ValueError("Portfolio weights cannot be negative.")

        # Validate sum to 1
        if abs(sum(weights.values()) - 1.0) > LimitsAndConstraints.PORTFOLIO_WEIGHT_TOLERANCE:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")

        return weights

    def _load_returns_data(
        self, successful: Dict[str, TickerAnalysis]
    ) -> Optional[pd.DataFrame]:
        """Load and align returns data from ticker analyses with LRU caching.

        Args:
            successful: Successful analyses

        Returns:
            DataFrame with aligned returns, or None if insufficient data
        """
        # Create cache key from ticker set
        cache_key = frozenset(successful.keys())

        # Check cache first - move to end (mark as recently used)
        if cache_key in self._returns_cache:
            logger.debug(f"Using cached returns data for {len(successful)} tickers")
            # Move to end to mark as recently used (LRU)
            self._returns_cache.move_to_end(cache_key)
            return self._returns_cache[cache_key]

        # Load returns data from CSV files
        returns_data = {}
        for ticker, analysis in successful.items():
            try:
                df = pd.read_csv(analysis.csv_path)
                returns_data[ticker] = df["daily_return"].dropna()
            except (OSError, pd.errors.ParserError, KeyError, ValueError) as e:
                logger.debug(f"Could not load returns for {ticker}: {e}")

        if len(returns_data) < 2:
            return None

        # Align returns to common dates
        returns_df = pd.DataFrame(returns_data).dropna()

        # Evict oldest entry if cache is full (LRU eviction)
        if len(self._returns_cache) >= self._cache_max_size:
            evicted_key = next(iter(self._returns_cache))  # Get first (oldest) key
            self._returns_cache.pop(evicted_key)
            logger.debug(
                f"Evicted oldest cache entry (cache size: {len(self._returns_cache)})"
            )

        # Cache the result (added at the end, marking as most recently used)
        self._returns_cache[cache_key] = returns_df
        logger.debug(
            f"Cached returns data for {len(successful)} tickers "
            f"(cache size: {len(self._returns_cache)}/{self._cache_max_size})"
        )

        return returns_df

    def _calculate_portfolio_returns(
        self, returns_df: pd.DataFrame, weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns time series.

        Args:
            returns_df: DataFrame with individual ticker returns
            weights: Portfolio weights

        Returns:
            Series of portfolio returns
        """
        weights_array = np.array([weights[t] for t in returns_df.columns])
        portfolio_returns = returns_df.values @ weights_array
        return pd.Series(portfolio_returns)

    def _calculate_basic_portfolio_metrics(
        self, portfolio_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate basic portfolio performance metrics.

        Args:
            portfolio_returns: Portfolio returns time series

        Returns:
            Dictionary with return, volatility, and Sharpe ratio
        """
        # Use geometric compounding for annualized return
        mean_daily_return = portfolio_returns.mean()
        portfolio_return = (1 + mean_daily_return) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
        portfolio_volatility = portfolio_returns.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)
        portfolio_sharpe = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0
            else None
        )

        return {
            "portfolio_return": portfolio_return,
            "portfolio_volatility": portfolio_volatility,
            "portfolio_sharpe": portfolio_sharpe,
        }

    def _calculate_portfolio_risk_metrics(
        self, portfolio_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate portfolio risk metrics.

        Args:
            portfolio_returns: Portfolio returns time series

        Returns:
            Dictionary with VaR, CVaR, max drawdown, and drawdown duration
        """
        # Value at Risk
        portfolio_var_95 = np.percentile(portfolio_returns, 5)

        # Conditional VaR (Expected Shortfall)
        var_threshold = np.percentile(portfolio_returns, 5)
        tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
        portfolio_cvar_95 = float(tail_losses.mean()) if len(tail_losses) > 0 else None

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        portfolio_max_drawdown = drawdown.min()

        # Max drawdown duration
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()

        portfolio_max_dd_duration = None
        if not pd.isna(max_dd_value) and max_dd_value != 0:
            # Find drawdown start
            drawdown_start = None
            for i in range(len(drawdown)):
                if drawdown.index[i] >= max_dd_idx:
                    break
                if drawdown.iloc[i] == 0:
                    drawdown_start = drawdown.index[i]

            if drawdown_start is None:
                drawdown_start = drawdown.index[0]

            # Find recovery
            recovery_idx = None
            for i in range(len(drawdown)):
                if drawdown.index[i] > max_dd_idx and drawdown.iloc[i] >= 0:
                    recovery_idx = drawdown.index[i]
                    break

            if recovery_idx is not None:
                duration = len(portfolio_returns.loc[drawdown_start:recovery_idx]) - 1
                portfolio_max_dd_duration = int(duration) if duration > 0 else None

        return {
            "portfolio_var_95": portfolio_var_95,
            "portfolio_cvar_95": portfolio_cvar_95,
            "portfolio_max_drawdown": portfolio_max_drawdown,
            "portfolio_max_dd_duration": portfolio_max_dd_duration,
        }

    def _calculate_diversification_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_volatility: float,
    ) -> Dict[str, Any]:
        """Calculate diversification and correlation metrics.

        Args:
            returns_df: DataFrame with individual ticker returns
            weights: Portfolio weights
            portfolio_volatility: Portfolio volatility

        Returns:
            Dictionary with correlation matrix and diversification ratio
        """
        # Correlation matrix
        correlation_matrix = returns_df.corr().to_dict()

        # Diversification ratio
        individual_vols = returns_df.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)
        weighted_vol_sum = sum(
            weights[t] * individual_vols[t] for t in returns_df.columns
        )
        diversification_ratio = (
            weighted_vol_sum / portfolio_volatility
            if portfolio_volatility > 0
            else None
        )

        return {
            "correlation_matrix": correlation_matrix,
            "diversification_ratio": diversification_ratio,
        }

    def _calculate_contribution_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: Dict[str, float],
        successful: Dict[str, TickerAnalysis],
    ) -> Dict[str, Any]:
        """Calculate return contribution and concentration metrics.

        Args:
            returns_df: DataFrame with individual ticker returns
            weights: Portfolio weights
            successful: Successful analyses

        Returns:
            Dictionary with weights, contributors, concentration, and total value
        """
        # Return contribution - use geometric compounding
        individual_returns = {
            t: (1 + returns_df[t].mean()) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1
            for t in returns_df.columns
        }

        # Calculate absolute contributions (weight × return)
        contributions = [
            (t, weights[t] * individual_returns[t]) for t in returns_df.columns
        ]

        # Calculate total portfolio return (sum of all contributions)
        total_contribution = sum(contrib[1] for contrib in contributions)

        # Convert to relative contribution percentages (contribution / total)
        # This ensures contributions sum to 1.0 (100%)
        if abs(total_contribution) > 1e-10:  # Avoid division by near-zero
            top_contributors = [
                (ticker, (contribution / total_contribution))
                for ticker, contribution in contributions
            ]
        else:
            # If total contribution is near zero, use equal attribution
            top_contributors = [(ticker, 1.0 / len(contributions)) for ticker, _ in contributions]

        # Sort by absolute contribution magnitude
        top_contributors = sorted(top_contributors, key=lambda x: abs(x[1]), reverse=True)

        # Concentration risk (Herfindahl index)
        concentration_risk = sum(w**2 for w in weights.values())

        # Total value (assuming 100 shares base)
        total_value = sum(
            weights[ticker] * analysis.latest_close * 100
            for ticker, analysis in successful.items()
            if ticker in weights
        )

        return {
            "weights": weights,
            "top_contributors": top_contributors,
            "concentration_risk": concentration_risk,
            "total_value": total_value,
        }

    def _simple_portfolio_metrics(
        self, analyses: Dict[str, TickerAnalysis], weights: Dict[str, float]
    ) -> PortfolioMetrics:
        """Simplified metrics when full analysis isn't possible."""
        total_value = sum(
            weights[ticker] * analysis.latest_close * 100
            for ticker, analysis in analyses.items()
            if ticker in weights
        )

        avg_return = sum(
            weights[t] * a.avg_daily_return for t, a in analyses.items() if t in weights
        )

        avg_vol = sum(
            weights[t] * a.volatility for t, a in analyses.items() if t in weights
        )

        # Use geometric compounding for annualized return
        portfolio_return = (1 + avg_return) ** TimeConstants.TRADING_DAYS_PER_YEAR - 1

        return PortfolioMetrics(
            total_value=total_value,
            portfolio_return=portfolio_return,
            portfolio_volatility=avg_vol * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR),
            weights=weights,
        )

    def clear_cache(self) -> None:
        """Clear the returns data cache to free memory."""
        self._returns_cache.clear()
        logger.debug("Cleared returns data cache")


# ============================================================================
# PORTFOLIO OPTIONS ANALYZER
# ============================================================================


class PortfolioOptionsAnalyzer:
    """Portfolio-level options analysis and hedging recommendations."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initialize the PortfolioOptionsAnalyzer.

        Args:
            risk_free_rate: Annual risk-free rate as decimal
        """
        self.risk_free_rate = risk_free_rate

    def calculate_portfolio_options_metrics(
        self,
        analyses: Dict[str, Any],  # Dict[str, TickerAnalysis] with options_analysis
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Calculate aggregate options metrics across portfolio.

        Args:
            analyses: Dictionary of ticker analyses (with options_analysis populated)
            weights: Portfolio weights

        Returns:
            PortfolioOptionsMetrics with aggregated Greeks and recommendations
        """
        from models_options import PortfolioOptionsMetrics

        # Filter to analyses with options data
        options_analyses = {
            ticker: analysis
            for ticker, analysis in analyses.items()
            if not analysis.error and analysis.options_analysis is not None
        }

        if not options_analyses:
            logger.warning("No options analyses available for portfolio metrics")
            return None

        # Default to equal weights
        if weights is None:
            n = len(options_analyses)
            weights = {ticker: 1.0 / n for ticker in options_analyses.keys()}

        # Aggregate Greeks across all positions
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0

        delta_by_ticker = {}
        vega_by_ticker = {}
        theta_by_ticker = {}

        for ticker, analysis in options_analyses.items():
            weight = weights.get(ticker, 0.0)
            options_data = analysis.options_analysis

            if not options_data or not options_data.chains:
                continue

            # Aggregate Greeks from all chains for this ticker
            ticker_delta = 0.0
            ticker_gamma = 0.0
            ticker_theta = 0.0
            ticker_vega = 0.0
            ticker_rho = 0.0

            for chain in options_data.chains:
                for contract in chain.all_contracts:
                    if contract.greeks:
                        # Assume long position (quantity = 1)
                        ticker_delta += contract.greeks.delta_dollars or 0.0
                        ticker_gamma += contract.greeks.gamma or 0.0
                        ticker_theta += contract.greeks.theta or 0.0
                        ticker_vega += contract.greeks.vega or 0.0
                        ticker_rho += contract.greeks.rho or 0.0

            # Weight by portfolio allocation
            delta_by_ticker[ticker] = ticker_delta * weight
            vega_by_ticker[ticker] = ticker_vega * weight
            theta_by_ticker[ticker] = ticker_theta * weight

            total_delta += ticker_delta * weight
            total_gamma += ticker_gamma * weight
            total_theta += ticker_theta * weight
            total_vega += ticker_vega * weight
            total_rho += ticker_rho * weight

        # Calculate delta hedge required (shares needed to neutralize)
        delta_hedge_required = -total_delta if abs(total_delta) > 10 else None

        # Generate hedging suggestions
        vega_hedge_suggestion = self._generate_vega_hedge_suggestion(total_vega)
        theta_management_suggestion = self._generate_theta_suggestion(total_theta)

        # Find largest Greek exposure
        largest_greek_exposure = self._find_largest_greek_exposure(
            delta_by_ticker, vega_by_ticker, theta_by_ticker
        )

        # Calculate concentration risk (Herfindahl index for Greeks)
        greek_concentration_risk = self._calculate_greek_concentration(
            delta_by_ticker, vega_by_ticker, theta_by_ticker
        )

        # Generate overall risk assessment
        risk_assessment = self._generate_risk_assessment(
            total_delta, total_gamma, total_theta, total_vega
        )

        # Generate hedging recommendations
        hedging_recommendations = self._generate_hedging_recommendations(
            total_delta, total_vega, total_theta, delta_by_ticker
        )

        return PortfolioOptionsMetrics(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            total_rho=total_rho,
            delta_by_ticker=delta_by_ticker,
            vega_by_ticker=vega_by_ticker,
            theta_by_ticker=theta_by_ticker,
            net_portfolio_delta=total_delta,
            portfolio_gamma_risk=abs(total_gamma),
            portfolio_theta_decay=total_theta,
            portfolio_vega_exposure=total_vega,
            delta_hedge_required=delta_hedge_required,
            vega_hedge_suggestion=vega_hedge_suggestion,
            theta_management_suggestion=theta_management_suggestion,
            largest_greek_exposure=largest_greek_exposure,
            greek_concentration_risk=greek_concentration_risk,
            hedging_recommendations=hedging_recommendations,
            risk_assessment=risk_assessment,
        )

    def _generate_vega_hedge_suggestion(self, total_vega: float) -> str:
        """Generate vega hedging suggestion based on exposure."""
        if abs(total_vega) < 50:
            return "Portfolio is approximately vega-neutral. No hedging required."
        elif total_vega > 0:
            return (
                f"Portfolio is long vega (+{total_vega:.2f}). "
                "Consider selling volatility via strangles or iron condors to reduce exposure."
            )
        else:
            return (
                f"Portfolio is short vega ({total_vega:.2f}). "
                "Consider buying volatility via long straddles or calendar spreads to reduce risk."
            )

    def _generate_theta_suggestion(self, total_theta: float) -> str:
        """Generate theta management suggestion."""
        if abs(total_theta) < 10:
            return "Portfolio theta is minimal. Time decay impact is low."
        elif total_theta > 0:
            return (
                f"Portfolio benefits from time decay (+${total_theta:.2f}/day). "
                "Monitor positions approaching expiration."
            )
        else:
            return (
                f"Portfolio loses ${abs(total_theta):.2f}/day to theta decay. "
                "Consider closing expiring positions or rolling to later expirations."
            )

    def _find_largest_greek_exposure(
        self,
        delta_by_ticker: Dict[str, float],
        vega_by_ticker: Dict[str, float],
        theta_by_ticker: Dict[str, float],
    ) -> Optional[Tuple[str, str, float]]:
        """Find the single largest Greek exposure across portfolio."""
        largest = None
        max_abs_value = 0

        for ticker, delta in delta_by_ticker.items():
            if abs(delta) > max_abs_value:
                max_abs_value = abs(delta)
                largest = (ticker, "Delta", delta)

        for ticker, vega in vega_by_ticker.items():
            if abs(vega) > max_abs_value:
                max_abs_value = abs(vega)
                largest = (ticker, "Vega", vega)

        for ticker, theta in theta_by_ticker.items():
            if abs(theta) > max_abs_value:
                max_abs_value = abs(theta)
                largest = (ticker, "Theta", theta)

        return largest

    def _calculate_greek_concentration(
        self,
        delta_by_ticker: Dict[str, float],
        vega_by_ticker: Dict[str, float],
        theta_by_ticker: Dict[str, float],
    ) -> float:
        """Calculate Herfindahl index for Greek concentration."""
        # Combine all Greeks
        all_exposures = []
        all_exposures.extend(delta_by_ticker.values())
        all_exposures.extend(vega_by_ticker.values())
        all_exposures.extend(theta_by_ticker.values())

        if not all_exposures:
            return 0.0

        total_exposure = sum(abs(x) for x in all_exposures)
        if total_exposure == 0:
            return 0.0

        # Calculate Herfindahl index
        herfindahl = sum((abs(x) / total_exposure) ** 2 for x in all_exposures)
        return herfindahl

    def _generate_risk_assessment(
        self, delta: float, gamma: float, theta: float, vega: float
    ) -> str:
        """Generate overall risk assessment narrative."""
        risks = []

        if abs(delta) > 100:
            risks.append(
                f"High directional risk (Delta: {delta:.2f}). Portfolio is highly sensitive to underlying price movements."
            )

        if abs(gamma) > 0.5:
            risks.append(
                f"High gamma exposure ({gamma:.4f}). Delta will change rapidly with price movements."
            )

        if theta < -50:
            risks.append(
                f"Significant time decay (${abs(theta):.2f}/day). Portfolio loses value each day if prices don't move."
            )

        if abs(vega) > 200:
            risks.append(
                f"High volatility sensitivity (Vega: {vega:.2f}). Portfolio is very exposed to IV changes."
            )

        if not risks:
            return "Portfolio Greeks are well-balanced with moderate risk across all dimensions."

        return "Risk Assessment:\n" + "\n".join(f"• {risk}" for risk in risks)

    def _generate_hedging_recommendations(
        self,
        total_delta: float,
        total_vega: float,
        total_theta: float,
        delta_by_ticker: Dict[str, float],
    ) -> str:
        """Generate specific hedging recommendations."""
        recommendations = []

        # Delta hedging
        if abs(total_delta) > 100:
            shares_to_hedge = int(-total_delta)
            action = "buy" if shares_to_hedge > 0 else "sell"
            recommendations.append(
                f"Delta Hedge: {action.title()} {abs(shares_to_hedge)} shares of underlying to neutralize delta exposure."
            )

        # Vega hedging
        if abs(total_vega) > 200:
            if total_vega > 0:
                recommendations.append(
                    "Vega Hedge: Sell ATM straddles or iron condors to reduce long volatility exposure."
                )
            else:
                recommendations.append(
                    "Vega Hedge: Buy calendar spreads or long straddles to reduce short volatility risk."
                )

        # Theta management
        if total_theta < -100:
            recommendations.append(
                "Theta Management: Consider closing near-expiration positions or rolling to later dates to reduce time decay."
            )

        # Position concentration
        if delta_by_ticker:
            max_delta_ticker = max(delta_by_ticker.items(), key=lambda x: abs(x[1]))
            if abs(max_delta_ticker[1]) > abs(total_delta) * 0.5:
                recommendations.append(
                    f"Concentration Risk: {max_delta_ticker[0]} represents {abs(max_delta_ticker[1])/abs(total_delta)*100:.1f}% of delta exposure. Consider diversifying."
                )

        if not recommendations:
            return "Portfolio is well-hedged. No immediate hedging actions required."

        return "Hedging Recommendations:\n" + "\n".join(
            f"{i+1}. {rec}" for i, rec in enumerate(recommendations)
        )
