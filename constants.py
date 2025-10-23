"""
Configuration constants for financial analysis.

This module centralizes all magic numbers and thresholds used throughout the application,
making them easy to find, understand, and modify.

All constants are organized into classes for better organization and to eliminate
duplication. Use `ClassName.CONSTANT_NAME` to access values.
"""

# ============================================================================
# THRESHOLD CLASSES (Organized Configuration)
# ============================================================================


class AnalysisThresholds:
    """Analysis alert and signal thresholds."""

    # Volatility thresholds
    HIGH_VOLATILITY = 0.05  # 5% daily volatility triggers alert

    # Drawdown thresholds
    LARGE_DRAWDOWN = -0.10  # -10% drawdown triggers alert

    # RSI thresholds
    OVERBOUGHT_RSI = 70  # Above this indicates overbought conditions
    OVERSOLD_RSI = 30  # Below this indicates oversold conditions

    # VaR threshold
    VALUE_AT_RISK_95_THRESHOLD = -0.03  # VaR 95% threshold for alerts

    # Performance thresholds
    UNDERPERFORMANCE = -0.05  # -5% vs benchmark triggers alert


class TechnicalIndicatorParameters:
    """Technical indicator calculation parameters and window sizes."""

    # RSI (Relative Strength Index)
    RSI_PERIOD = 14  # Standard RSI calculation period

    # Moving Averages
    MA_SHORT_PERIOD = 30  # 30-day moving average (short-term trend)
    MA_LONG_PERIOD = 50  # 50-day moving average (long-term trend)
    VOLATILITY_WINDOW = 20  # Rolling window for volatility calculation

    # Bollinger Bands
    BOLLINGER_PERIOD = 20  # Period for Bollinger Bands calculation
    BOLLINGER_STD_DEV = 2  # Number of standard deviations for bands

    # MACD (Moving Average Convergence Divergence)
    MACD_FAST_PERIOD = 12  # Fast EMA period
    MACD_SLOW_PERIOD = 26  # Slow EMA period
    MACD_SIGNAL_PERIOD = 9  # Signal line EMA period


class TimeConstants:
    """Time and annualization constants."""

    # Trading days
    TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year

    # Minimum data requirements
    MIN_DATA_POINTS_BASIC = 10  # Minimum data points for basic calculations
    MIN_DATA_POINTS_PORTFOLIO = 20  # Minimum data points for portfolio analysis

    # Minimum periods for rolling calculations
    MIN_PERIODS_SHORT_MA = 5  # Minimum periods for short-term moving average
    MIN_PERIODS_LONG_MA = 10  # Minimum periods for long-term moving average
    MIN_PERIODS_VOLATILITY = 5  # Minimum periods for volatility calculation

    # Data availability thresholds
    MIN_ROWS_FOR_FULL_ANALYSIS = 1  # Minimum number of rows to attempt analysis
    DATA_POINTS_DIVISOR_FOR_VOL_WINDOW = 3  # Divisor for calculating volatility window
    # Explanation: vol_window = min(VOLATILITY_WINDOW, max(1, n_rows // 3))
    # This ensures volatility calculations adapt to available data:
    # - With 60 days of data: vol_window = min(20, 20) = 20 days
    # - With 30 days of data: vol_window = min(20, 10) = 10 days
    # - With 9 days of data: vol_window = min(20, 3) = 3 days


class LimitsAndConstraints:
    """System limits and constraints."""

    # Ticker limits
    MAX_TICKERS_ALLOWED = 20  # Maximum number of tickers to analyze at once

    # Near-zero thresholds
    NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10  # Threshold for near-zero variance detection
    MIN_DOWNSIDE_DEVIATION = 0.001  # Minimum downside deviation to avoid division by zero

    # Portfolio weights
    PORTFOLIO_WEIGHT_TOLERANCE = 0.01  # Acceptable deviation from sum=1.0 for weights

    # Valid periods for yfinance
    VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]


class ChartConfiguration:
    """Chart and visualization configuration."""

    # Figure sizes (width, height in inches)
    PRICE_CHART_SIZE = (14, 10)  # Main price chart with subplots
    COMPARISON_CHART_SIZE = (14, 8)  # Comparison/portfolio charts
    RISK_REWARD_CHART_SIZE = (12, 8)  # Risk-reward scatter plot

    # Chart DPI
    CHART_DPI = 150  # Resolution for saved charts

    # Font sizes
    TITLE_FONTSIZE = 16
    AXIS_FONTSIZE = 12

    # Subplot height ratios
    PRICE_CHART_HEIGHT_RATIOS = [3, 1]  # Main chart : RSI chart


class DataValidation:
    """Data validation constants."""

    # YoY growth calculation
    YOY_QUARTERS_LOOKBACK = 4  # Number of quarters for year-over-year comparison
    CURRENT_PERIOD_INDEX = 0  # Index for current period in financial statements

    # Minimum required quarters for growth calculation
    MIN_QUARTERS_FOR_GROWTH = 5  # Need at least 5 quarters (current + 4 back)


class Defaults:
    """Default configuration values."""

    # Cache defaults (can be overridden by config)
    CACHE_TTL_HOURS = 24  # Default cache time-to-live

    # Risk-free rate (can be overridden by config)
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

    # Benchmark ticker (can be overridden by config)
    BENCHMARK_TICKER = "SPY"  # S&P 500 ETF

    # Parallel processing (can be overridden by config)
    MAX_WORKERS = 3  # Default number of parallel worker threads

    # Request timeout (can be overridden by config)
    REQUEST_TIMEOUT = 30  # seconds

    # Report quality scoring
    MIN_QUALITY_SCORE = 0  # Minimum quality score
    MAX_QUALITY_SCORE = 10  # Maximum quality score


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_constants() -> None:
    """
    Validates that all constants are within reasonable ranges.

    Raises:
        ValueError: If any constant is invalid.
    """
    # RSI thresholds
    if not (0 < AnalysisThresholds.OVERBOUGHT_RSI <= 100):
        raise ValueError(
            f"OVERBOUGHT_RSI must be between 0 and 100, got {AnalysisThresholds.OVERBOUGHT_RSI}"
        )

    if not (0 <= AnalysisThresholds.OVERSOLD_RSI < 100):
        raise ValueError(
            f"OVERSOLD_RSI must be between 0 and 100, got {AnalysisThresholds.OVERSOLD_RSI}"
        )

    if AnalysisThresholds.OVERSOLD_RSI >= AnalysisThresholds.OVERBOUGHT_RSI:
        raise ValueError(
            "OVERSOLD_RSI must be less than OVERBOUGHT_RSI"
        )

    # Period validations
    if TechnicalIndicatorParameters.RSI_PERIOD <= 0:
        raise ValueError(f"RSI_PERIOD must be positive, got {TechnicalIndicatorParameters.RSI_PERIOD}")

    if TechnicalIndicatorParameters.MA_SHORT_PERIOD >= TechnicalIndicatorParameters.MA_LONG_PERIOD:
        raise ValueError("MA_SHORT_PERIOD must be less than MA_LONG_PERIOD")

    if TechnicalIndicatorParameters.MACD_FAST_PERIOD >= TechnicalIndicatorParameters.MACD_SLOW_PERIOD:
        raise ValueError("MACD_FAST_PERIOD must be less than MACD_SLOW_PERIOD")

    # Limits
    if LimitsAndConstraints.MAX_TICKERS_ALLOWED <= 0:
        raise ValueError(
            f"MAX_TICKERS_ALLOWED must be positive, got {LimitsAndConstraints.MAX_TICKERS_ALLOWED}"
        )

    if TimeConstants.TRADING_DAYS_PER_YEAR <= 0:
        raise ValueError(
            f"TRADING_DAYS_PER_YEAR must be positive, got {TimeConstants.TRADING_DAYS_PER_YEAR}"
        )


# Validate on import
validate_constants()


# ============================================================================
# DOCUMENTATION HELPERS
# ============================================================================


def get_all_constants() -> dict:
    """
    Gets a dictionary of all constants organized by class.

    Returns:
        dict: A nested dictionary of class names -> constant names -> values.
    """
    classes = [
        AnalysisThresholds,
        TechnicalIndicatorParameters,
        TimeConstants,
        LimitsAndConstraints,
        ChartConfiguration,
        DataValidation,
        Defaults,
    ]

    result = {}
    for cls in classes:
        result[cls.__name__] = {
            name: value
            for name, value in vars(cls).items()
            if name.isupper() and not name.startswith("_")
        }

    return result


def print_constants() -> None:
    """
    Prints all constants in a readable format organized by class.
    """
    constants = get_all_constants()

    print("=" * 80)
    print("FINANCIAL ANALYSIS CONSTANTS")
    print("=" * 80)

    for class_name, class_constants in constants.items():
        print(f"\n{class_name}:")
        print("-" * 80)
        for name, value in sorted(class_constants.items()):
            print(f"  {name:50} = {value}")

    print("=" * 80)


if __name__ == "__main__":
    # When run directly, print all constants
    print_constants()
