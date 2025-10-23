"""
Financial analysis components.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
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
                        "volatility",
                        "rsi",
                        "bollinger_upper",
                        "bollinger_lower",
                        "bollinger_position",
                        "macd",
                        "macd_signal",
                    ]
                }
            )

        # Calculate window parameters once
        short_term_ma_window = min(TechnicalIndicatorParameters.MA_SHORT_PERIOD, n_rows)
        long_term_ma_window = min(TechnicalIndicatorParameters.MA_LONG_PERIOD, n_rows)
        vol_window = min(TechnicalIndicatorParameters.VOLATILITY_WINDOW, max(TimeConstants.MIN_ROWS_FOR_FULL_ANALYSIS, n_rows // TimeConstants.DATA_POINTS_DIVISOR_FOR_VOL_WINDOW))

        # Ensure min_periods doesn't exceed window size
        ma_short_min_periods = min(TimeConstants.MIN_PERIODS_SHORT_MA, short_term_ma_window)
        ma_long_min_periods = min(TimeConstants.MIN_PERIODS_LONG_MA, long_term_ma_window)
        vol_min_periods = min(TimeConstants.MIN_PERIODS_VOLATILITY, vol_window)

        # Pre-calculate close prices (used multiple times)
        close_prices = enriched_price_data["Close"].astype(float)

        # Calculate technical indicators that don't depend on other computed columns
        rsi_values = self._compute_rsi(close_prices)
        bollinger_upper, bollinger_lower = self._compute_bollinger_bands(close_prices)
        macd_values, macd_signal_values = self._compute_macd(close_prices)

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
                "rsi": rsi_values,
                "bollinger_upper": bollinger_upper,
                "bollinger_lower": bollinger_lower,
                "macd": macd_values,
                "macd_signal": macd_signal_values,
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
        rs = gain / loss
        return 100 - (100 / (1 + rs))

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

        annualized_return = returns_clean.mean() * TimeConstants.TRADING_DAYS_PER_YEAR
        annualized_vol = returns_clean.std() * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR)

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
            benchmark_return = aligned_benchmark.mean() * TimeConstants.TRADING_DAYS_PER_YEAR
            alpha = (
                aligned_returns.mean() * TimeConstants.TRADING_DAYS_PER_YEAR - self.risk_free_rate
            ) - beta * (benchmark_return - self.risk_free_rate)

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
        treynor = (
            (aligned_returns.mean() * TimeConstants.TRADING_DAYS_PER_YEAR - self.risk_free_rate)
            / beta
            if beta and beta != 0
            else None
        )

        active_returns = aligned_returns - aligned_benchmark
        information_ratio = (
            active_returns.mean()
            * TimeConstants.TRADING_DAYS_PER_YEAR
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

        # 2. Alpha (excess return vs expected)
        alpha = None
        if beta is not None:
            benchmark_return = aligned_benchmark.mean() * TimeConstants.TRADING_DAYS_PER_YEAR
            ticker_return = aligned_returns.mean() * TimeConstants.TRADING_DAYS_PER_YEAR
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

        # 5. Information ratio (risk-adjusted active return)
        information_ratio = (
            float(
                active_returns.mean()
                * TimeConstants.TRADING_DAYS_PER_YEAR
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

                # Calculate growth rates (YoY) using helper method
                fundamentals.revenue_growth = self._calculate_yoy_growth(
                    income_stmt, "Total Revenue"
                )
                fundamentals.earnings_growth = self._calculate_yoy_growth(
                    income_stmt, "Net Income"
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


class PortfolioAnalyzer:
    """Portfolio-level metrics computation."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        """
        Initializes the PortfolioAnalyzer with a risk-free rate.

        Args:
            risk_free_rate (float): The risk-free rate to use in calculations.
        """
        self.risk_free_rate = risk_free_rate
        self._returns_cache: Dict[str, pd.DataFrame] = {}  # Cache for loaded returns data

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
        """Load and align returns data from ticker analyses.

        Args:
            successful: Successful analyses

        Returns:
            DataFrame with aligned returns, or None if insufficient data
        """
        # Create cache key from ticker set
        cache_key = frozenset(successful.keys())

        # Check cache first
        if cache_key in self._returns_cache:
            logger.debug(f"Using cached returns data for {len(successful)} tickers")
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

        # Cache the result
        self._returns_cache[cache_key] = returns_df
        logger.debug(f"Cached returns data for {len(successful)} tickers")

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
        portfolio_return = portfolio_returns.mean() * TimeConstants.TRADING_DAYS_PER_YEAR
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
    ) -> Dict[str, float]:
        """Calculate portfolio risk metrics.

        Args:
            portfolio_returns: Portfolio returns time series

        Returns:
            Dictionary with VaR and max drawdown
        """
        # Value at Risk
        portfolio_var_95 = np.percentile(portfolio_returns, 5)

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        portfolio_max_drawdown = drawdown.min()

        return {
            "portfolio_var_95": portfolio_var_95,
            "portfolio_max_drawdown": portfolio_max_drawdown,
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
        # Return contribution
        individual_returns = {
            t: returns_df[t].mean() * TimeConstants.TRADING_DAYS_PER_YEAR for t in returns_df.columns
        }
        contributions = [
            (t, weights[t] * individual_returns[t]) for t in returns_df.columns
        ]
        top_contributors = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

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

        return PortfolioMetrics(
            total_value=total_value,
            portfolio_return=avg_return * TimeConstants.TRADING_DAYS_PER_YEAR,
            portfolio_volatility=avg_vol * np.sqrt(TimeConstants.TRADING_DAYS_PER_YEAR),
            weights=weights,
        )

    def clear_cache(self) -> None:
        """Clear the returns data cache to free memory."""
        self._returns_cache.clear()
        logger.debug("Cleared returns data cache")
