"""
Configuration constants for financial analysis.

This module centralizes all magic numbers and thresholds used throughout the application,
making them easy to find, understand, and modify.
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
    HIGH_VAR = -0.03  # VaR 95% threshold for alerts

    # Performance thresholds
    UNDERPERFORMANCE = -0.05  # -5% vs benchmark triggers alert


class TechnicalIndicators:
    """Technical indicator calculation parameters."""

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


# ============================================================================
# BACKWARD COMPATIBILITY (Legacy Constants)
# ============================================================================
# These maintain backward compatibility with existing code
# New code should use the class-based constants above

# RSI (Relative Strength Index)
RSI_PERIOD = TechnicalIndicators.RSI_PERIOD
RSI_OVERBOUGHT_THRESHOLD = AnalysisThresholds.OVERBOUGHT_RSI
RSI_OVERSOLD_THRESHOLD = AnalysisThresholds.OVERSOLD_RSI

# Moving Averages
MA_SHORT_PERIOD = TechnicalIndicators.MA_SHORT_PERIOD
MA_LONG_PERIOD = TechnicalIndicators.MA_LONG_PERIOD
VOLATILITY_WINDOW = TechnicalIndicators.VOLATILITY_WINDOW

# Bollinger Bands
BOLLINGER_PERIOD = TechnicalIndicators.BOLLINGER_PERIOD
BOLLINGER_STD_DEV = TechnicalIndicators.BOLLINGER_STD_DEV

# MACD (Moving Average Convergence Divergence)
MACD_FAST_PERIOD = TechnicalIndicators.MACD_FAST_PERIOD
MACD_SLOW_PERIOD = TechnicalIndicators.MACD_SLOW_PERIOD
MACD_SIGNAL_PERIOD = TechnicalIndicators.MACD_SIGNAL_PERIOD

# ============================================================================
# TIME AND ANNUALIZATION
# ============================================================================

# Trading days
TRADING_DAYS_PER_YEAR = 252  # Standard number of trading days in a year

# Minimum data requirements
MIN_DATA_POINTS_BASIC = 10  # Minimum data points for basic calculations
MIN_DATA_POINTS_PORTFOLIO = 20  # Minimum data points for portfolio analysis

# ============================================================================
# ALERT THRESHOLDS (Backward Compatibility)
# ============================================================================

# Volatility alerts
VOLATILITY_ALERT_THRESHOLD = AnalysisThresholds.HIGH_VOLATILITY
HIGH_VAR_THRESHOLD = AnalysisThresholds.HIGH_VAR

# Drawdown alerts
DRAWDOWN_ALERT_THRESHOLD = AnalysisThresholds.LARGE_DRAWDOWN

# Performance alerts
UNDERPERFORMANCE_THRESHOLD = AnalysisThresholds.UNDERPERFORMANCE

# ============================================================================
# LIMITS AND CONSTRAINTS
# ============================================================================

# Ticker limits
MAX_TICKERS_ALLOWED = 20  # Maximum number of tickers to analyze at once

# Near-zero thresholds
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10  # Threshold for near-zero variance detection
MIN_DOWNSIDE_DEVIATION = 0.001  # Minimum downside deviation to avoid division by zero

# Portfolio weights
PORTFOLIO_WEIGHT_TOLERANCE = 0.01  # Acceptable deviation from sum=1.0 for weights

# Valid periods for yfinance
VALID_PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

# ============================================================================
# CHART CONFIGURATION
# ============================================================================

# Figure sizes (width, height in inches)
PRICE_CHART_SIZE = (14, 10)  # Main price chart with subplots
COMPARISON_CHART_SIZE = (14, 8)  # Comparison/portfolio charts
RISK_REWARD_CHART_SIZE = (12, 8)  # Risk-reward scatter plot

# Chart DPI
CHART_DPI = 150  # Resolution for saved charts

# Font sizes
CHART_TITLE_FONTSIZE = 16
CHART_AXIS_FONTSIZE = 12

# Subplot height ratios
PRICE_CHART_HEIGHT_RATIOS = [3, 1]  # Main chart : RSI chart

# ============================================================================
# DATA VALIDATION
# ============================================================================

# YoY growth calculation
YOY_QUARTERS_LOOKBACK = 4  # Number of quarters for year-over-year comparison
CURRENT_PERIOD_INDEX = 0  # Index for current period in financial statements

# Minimum required quarters for growth calculation
MIN_QUARTERS_FOR_GROWTH = 5  # Need at least 5 quarters (current + 4 back)

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

# Cache defaults (can be overridden by config)
DEFAULT_CACHE_TTL_HOURS = 24  # Default cache time-to-live

# ============================================================================
# RISK PARAMETERS
# ============================================================================

# Risk-free rate (can be overridden by config)
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# Benchmark ticker (can be overridden by config)
DEFAULT_BENCHMARK_TICKER = "SPY"  # S&P 500 ETF

# ============================================================================
# COMPUTATION DEFAULTS
# ============================================================================

# Parallel processing (can be overridden by config)
DEFAULT_MAX_WORKERS = 3  # Default number of parallel worker threads

# Request timeout (can be overridden by config)
DEFAULT_REQUEST_TIMEOUT = 30  # seconds

# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================

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
    if not (0 < RSI_OVERBOUGHT_THRESHOLD <= 100):
        raise ValueError(f"RSI_OVERBOUGHT_THRESHOLD must be between 0 and 100, got {RSI_OVERBOUGHT_THRESHOLD}")

    if not (0 <= RSI_OVERSOLD_THRESHOLD < 100):
        raise ValueError(f"RSI_OVERSOLD_THRESHOLD must be between 0 and 100, got {RSI_OVERSOLD_THRESHOLD}")

    if RSI_OVERSOLD_THRESHOLD >= RSI_OVERBOUGHT_THRESHOLD:
        raise ValueError("RSI_OVERSOLD_THRESHOLD must be less than RSI_OVERBOUGHT_THRESHOLD")

    # Period validations
    if RSI_PERIOD <= 0:
        raise ValueError(f"RSI_PERIOD must be positive, got {RSI_PERIOD}")

    if MA_SHORT_PERIOD >= MA_LONG_PERIOD:
        raise ValueError("MA_SHORT_PERIOD must be less than MA_LONG_PERIOD")

    if MACD_FAST_PERIOD >= MACD_SLOW_PERIOD:
        raise ValueError("MACD_FAST_PERIOD must be less than MACD_SLOW_PERIOD")

    # Limits
    if MAX_TICKERS_ALLOWED <= 0:
        raise ValueError(f"MAX_TICKERS_ALLOWED must be positive, got {MAX_TICKERS_ALLOWED}")

    if TRADING_DAYS_PER_YEAR <= 0:
        raise ValueError(f"TRADING_DAYS_PER_YEAR must be positive, got {TRADING_DAYS_PER_YEAR}")


# Validate on import
validate_constants()


# ============================================================================
# DOCUMENTATION HELPERS
# ============================================================================


def get_all_constants() -> dict:
    """
    Gets a dictionary of all constants with their values.

    Returns:
        dict: A dictionary of constant names and values.
    """
    return {
        name: value
        for name, value in globals().items()
        if name.isupper() and not name.startswith('_')
    }


def print_constants() -> None:
    """
    Prints all constants in a readable format.
    """
    constants = get_all_constants()

    print("=" * 60)
    print("FINANCIAL ANALYSIS CONSTANTS")
    print("=" * 60)

    for name, value in sorted(constants.items()):
        print(f"{name:40} = {value}")

    print("=" * 60)


if __name__ == "__main__":
    # When run directly, print all constants
    print_constants()
