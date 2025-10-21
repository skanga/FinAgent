"""
Financial analysis components.
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import asdict
from typing import Optional, Dict, Tuple
from models import AdvancedMetrics, FundamentalData, PortfolioMetrics, TickerAnalysis
from constants import (
    RSI_PERIOD,
    MA_SHORT_PERIOD,
    MA_LONG_PERIOD,
    VOLATILITY_WINDOW,
    BOLLINGER_PERIOD,
    BOLLINGER_STD_DEV,
    MACD_FAST_PERIOD,
    MACD_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD,
    TRADING_DAYS_PER_YEAR,
    MIN_DATA_POINTS_BASIC,
    MIN_DATA_POINTS_PORTFOLIO,
    NEAR_ZERO_VARIANCE_THRESHOLD,
    MIN_DOWNSIDE_DEVIATION,
    YOY_QUARTERS_LOOKBACK,
    CURRENT_PERIOD_INDEX,
    MIN_QUARTERS_FOR_GROWTH,
    PORTFOLIO_WEIGHT_TOLERANCE
)

logger = logging.getLogger(__name__)


class AdvancedFinancialAnalyzer:
    """Comprehensive financial analysis with fundamentals parsing."""
    
    def __init__(self, risk_free_rate: float = 0.02, benchmark_ticker: str = "SPY") -> None:
        self.risk_free_rate = risk_free_rate
        self.benchmark_ticker = benchmark_ticker
    
    def compute_metrics(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """Compute technical metrics with optimized vectorized operations."""
        df = df_prices.copy()
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date').reset_index(drop=True)

        # Vectorized price and return calculations
        df['close'] = df['Close'].astype(float)
        df['daily_return'] = df['close'].pct_change()

        # Pre-calculate window sizes to avoid multiple len() calls
        n_rows = len(df)
        ma_30_window = min(MA_SHORT_PERIOD, n_rows)
        ma_50_window = min(MA_LONG_PERIOD, n_rows)
        vol_window = min(VOLATILITY_WINDOW, n_rows // 3)

        # Vectorized moving averages and volatility
        df['30d_ma'] = df['close'].rolling(window=ma_30_window, min_periods=5).mean()
        df['50d_ma'] = df['close'].rolling(window=ma_50_window, min_periods=10).mean()
        df['volatility'] = df['daily_return'].rolling(window=vol_window, min_periods=5).std()

        # Technical indicators (already vectorized internally)
        df['rsi'] = self._compute_rsi(df['close'])
        bollinger_upper, bollinger_lower = self._compute_bollinger_bands(df['close'])
        df['bollinger_upper'] = bollinger_upper
        df['bollinger_lower'] = bollinger_lower

        # Vectorized Bollinger position calculation with safe division
        band_width = df['bollinger_upper'] - df['bollinger_lower']
        df['bollinger_position'] = np.where(
            band_width > 0,
            (df['close'] - df['bollinger_lower']) / band_width,
            np.nan
        )

        # MACD indicators
        macd, macd_signal = self._compute_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal

        return df
    
    def _compute_rsi(self, prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_bollinger_bands(self, prices: pd.Series,
                                 period: int = BOLLINGER_PERIOD,
                                 std_dev: int = BOLLINGER_STD_DEV) -> Tuple[pd.Series, pd.Series]:
        """Compute Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return sma + (std * std_dev), sma - (std * std_dev)

    def _compute_macd(self, prices: pd.Series,
                     fast: int = MACD_FAST_PERIOD,
                     slow: int = MACD_SLOW_PERIOD,
                     signal: int = MACD_SIGNAL_PERIOD) -> Tuple[pd.Series, pd.Series]:
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd, macd.ewm(span=signal).mean()
    
    def calculate_advanced_metrics(self, returns: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None) -> AdvancedMetrics:
        """Calculate comprehensive risk metrics."""
        if len(returns) < MIN_DATA_POINTS_BASIC:
            return AdvancedMetrics()

        returns_clean = returns.dropna()
        if len(returns_clean) < MIN_DATA_POINTS_BASIC:
            return AdvancedMetrics()

        annualized_return = returns_clean.mean() * TRADING_DAYS_PER_YEAR
        annualized_vol = returns_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        sharpe = (annualized_return - self.risk_free_rate) / annualized_vol if annualized_vol > 0 else None
        
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min() if not drawdown.empty else None
        
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else MIN_DOWNSIDE_DEVIATION
        sortino = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else None
        
        calmar = annualized_return / abs(max_dd) if max_dd and max_dd != 0 else None
        var_95 = np.percentile(returns_clean, 5)
        
        beta, alpha, r_squared, treynor, information_ratio = self._calculate_benchmark_metrics(
            returns_clean, benchmark_returns
        )
        
        return AdvancedMetrics(
            sharpe_ratio=float(sharpe) if sharpe is not None and not pd.isna(sharpe) else None,
            max_drawdown=float(max_dd) if max_dd is not None and not pd.isna(max_dd) else None,
            beta=beta, alpha=alpha, r_squared=r_squared,
            var_95=float(var_95) if not pd.isna(var_95) else None,
            sortino_ratio=float(sortino) if sortino is not None and not pd.isna(sortino) else None,
            calmar_ratio=float(calmar) if calmar is not None and not pd.isna(calmar) else None,
            treynor_ratio=treynor, information_ratio=information_ratio
        )
    
    def _calculate_benchmark_metrics(self, returns: pd.Series,
                                     benchmark_returns: Optional[pd.Series]) -> Tuple:
        """Calculate metrics relative to benchmark."""
        if benchmark_returns is None or len(benchmark_returns) < MIN_DATA_POINTS_BASIC:
            return None, None, None, None, None

        benchmark_clean = benchmark_returns.dropna()
        common_index = returns.index.intersection(benchmark_clean.index)

        if len(common_index) < MIN_DATA_POINTS_BASIC:
            return None, None, None, None, None

        aligned_returns = returns.loc[common_index]
        aligned_benchmark = benchmark_clean.loc[common_index]

        # Validate we have sufficient data after alignment
        if len(aligned_returns) < MIN_DATA_POINTS_BASIC or len(aligned_benchmark) < MIN_DATA_POINTS_BASIC:
            return None, None, None, None, None

        # Ensure no NaN/inf values that could break calculations
        if aligned_returns.isna().any() or aligned_benchmark.isna().any():
            aligned_returns = aligned_returns.dropna()
            aligned_benchmark = aligned_benchmark.dropna()
            # Re-align after dropping NaN
            common_idx = aligned_returns.index.intersection(aligned_benchmark.index)
            if len(common_idx) < MIN_DATA_POINTS_BASIC:
                return None, None, None, None, None
            aligned_returns = aligned_returns.loc[common_idx]
            aligned_benchmark = aligned_benchmark.loc[common_idx]

        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        # Use threshold for near-zero variance to avoid division issues
        beta = covariance / benchmark_variance if benchmark_variance > NEAR_ZERO_VARIANCE_THRESHOLD and not np.isnan(benchmark_variance) else None

        alpha = None
        if beta is not None:
            benchmark_return = aligned_benchmark.mean() * TRADING_DAYS_PER_YEAR
            alpha = (aligned_returns.mean() * TRADING_DAYS_PER_YEAR - self.risk_free_rate) - beta * (
                benchmark_return - self.risk_free_rate
            )

        r_squared = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1] ** 2 if len(aligned_returns) > 1 else None
        treynor = (aligned_returns.mean() * TRADING_DAYS_PER_YEAR - self.risk_free_rate) / beta if beta and beta != 0 else None

        active_returns = aligned_returns - aligned_benchmark
        information_ratio = (
            active_returns.mean() * TRADING_DAYS_PER_YEAR / (active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if active_returns.std() > 0 else None
        )
        
        return beta, alpha, r_squared, treynor, information_ratio
    
    def compute_ratios(self, ticker: str) -> Dict[str, Optional[float]]:
        """Compute financial ratios using yfinance info."""
        ratios = {
            'pe_ratio': None, 'forward_pe': None, 'peg_ratio': None,
            'price_to_sales': None, 'price_to_book': None, 'debt_to_equity': None,
            'current_ratio': None, 'quick_ratio': None, 'return_on_equity': None,
            'return_on_assets': None, 'profit_margin': None, 'operating_margin': None,
            'gross_margin': None, 'dividend_yield': None, 'beta': None
        }
        
        try:
            t = yf.Ticker(ticker)
            info = t.info

            if not info:
                return ratios

            ratio_mapping = {
                'pe_ratio': 'trailingPE', 'forward_pe': 'forwardPE',
                'peg_ratio': 'pegRatio', 'price_to_sales': 'priceToSalesTrailing12Months',
                'price_to_book': 'priceToBook', 'debt_to_equity': 'debtToEquity',
                'current_ratio': 'currentRatio', 'quick_ratio': 'quickRatio',
                'return_on_equity': 'returnOnEquity', 'return_on_assets': 'returnOnAssets',
                'profit_margin': 'profitMargins', 'operating_margin': 'operatingMargins',
                'gross_margin': 'grossMargins', 'dividend_yield': 'dividendYield',
                'beta': 'beta'
            }

            for ratio_key, info_key in ratio_mapping.items():
                if info_key in info and info[info_key] is not None:
                    ratios[ratio_key] = float(info[info_key])

            logger.info(f"Computed {sum(1 for v in ratios.values() if v is not None)} ratios for {ticker}")

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Could not compute ratios for {ticker}: {e}")
        
        return ratios
    
    def parse_fundamentals(self, ticker: str) -> FundamentalData:
        """Parse and use financial statements."""
        fundamentals = FundamentalData()
        
        try:
            t = yf.Ticker(ticker)
            income_stmt = t.quarterly_income_stmt
            balance_sheet = t.quarterly_balance_sheet
            cash_flow = t.quarterly_cashflow
            
            # Parse income statement
            if income_stmt is not None and not income_stmt.empty:
                latest_income = income_stmt.iloc[:, 0]
                fundamentals.revenue = self._safe_get_value(latest_income, 'Total Revenue')
                fundamentals.net_income = self._safe_get_value(latest_income, 'Net Income')
                fundamentals.gross_profit = self._safe_get_value(latest_income, 'Gross Profit')
                fundamentals.operating_income = self._safe_get_value(latest_income, 'Operating Income')
                fundamentals.ebitda = self._safe_get_value(latest_income, 'EBITDA')

                # Calculate growth rates (YoY) using helper method
                fundamentals.revenue_growth = self._calculate_yoy_growth(income_stmt, 'Total Revenue')
                fundamentals.earnings_growth = self._calculate_yoy_growth(income_stmt, 'Net Income')
            
            # Parse balance sheet
            if balance_sheet is not None and not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                fundamentals.total_assets = self._safe_get_value(latest_balance, 'Total Assets')
                fundamentals.total_liabilities = self._safe_get_value(
                    latest_balance, 'Total Liabilities Net Minority Interest'
                )
                fundamentals.shareholders_equity = self._safe_get_value(
                    latest_balance, 'Stockholders Equity'
                )
            
            # Parse cash flow
            if cash_flow is not None and not cash_flow.empty:
                latest_cf = cash_flow.iloc[:, 0]
                fundamentals.operating_cash_flow = self._safe_get_value(latest_cf, 'Operating Cash Flow')
                fundamentals.free_cash_flow = self._safe_get_value(latest_cf, 'Free Cash Flow')
            
            non_none_count = sum(1 for field in asdict(fundamentals).values() if field is not None)
            logger.info(f"Parsed {non_none_count} fundamental metrics for {ticker}")

        except (KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
            logger.warning(f"Could not parse fundamentals for {ticker}: {e}")

        return fundamentals
    
    def _safe_get_value(self, series: pd.Series, key: str) -> Optional[float]:
        """Safely extract value from pandas Series."""
        try:
            if key in series.index:
                value = series[key]
                if pd.notna(value):
                    return float(value)
        except (KeyError, ValueError, TypeError):
            pass
        return None

    def _calculate_yoy_growth(self, dataframe: pd.DataFrame, metric_key: str,
                              current_col: int = CURRENT_PERIOD_INDEX,
                              yoy_col: int = YOY_QUARTERS_LOOKBACK) -> Optional[float]:
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

        if len(dataframe.columns) < MIN_QUARTERS_FOR_GROWTH:
            return None

        current_value = self._safe_get_value(dataframe.iloc[:, current_col], metric_key)
        yoy_value = self._safe_get_value(dataframe.iloc[:, yoy_col], metric_key)

        if current_value and yoy_value and yoy_value != 0:
            return (current_value - yoy_value) / yoy_value

        return None


class PortfolioAnalyzer:
    """Portfolio-level metrics computation."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_metrics(self, analyses: Dict[str, TickerAnalysis], 
                                    weights: Optional[Dict[str, float]] = None) -> PortfolioMetrics:
        """Calculate comprehensive portfolio-level metrics."""
        successful = {t: a for t, a in analyses.items() if not a.error}
        
        if not successful:
            raise ValueError("No successful analyses for portfolio calculation")
        
        # Default to equal weights
        if weights is None:
            n = len(successful)
            weights = {ticker: 1.0/n for ticker in successful.keys()}
        
        # Validate weights
        if abs(sum(weights.values()) - 1.0) > PORTFOLIO_WEIGHT_TOLERANCE:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")
        
        # Load returns data
        returns_data = {}
        for ticker, analysis in successful.items():
            try:
                df = pd.read_csv(analysis.csv_path)
                returns_data[ticker] = df['daily_return'].dropna()
            except (OSError, pd.errors.ParserError, KeyError, ValueError) as e:
                logger.warning(f"Could not load returns for {ticker}: {e}")
        
        if len(returns_data) < 2:
            return self._simple_portfolio_metrics(successful, weights)
        
        # Align returns to common dates
        returns_df = pd.DataFrame(returns_data).dropna()

        if len(returns_df) < MIN_DATA_POINTS_PORTFOLIO:
            return self._simple_portfolio_metrics(successful, weights)

        # Calculate portfolio returns
        weights_array = np.array([weights[t] for t in returns_df.columns])
        portfolio_returns = (returns_df.values @ weights_array)
        portfolio_returns_series = pd.Series(portfolio_returns)

        # Portfolio metrics
        portfolio_return = portfolio_returns_series.mean() * TRADING_DAYS_PER_YEAR
        portfolio_volatility = portfolio_returns_series.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        portfolio_sharpe = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility 
            if portfolio_volatility > 0 else None
        )
        portfolio_var_95 = np.percentile(portfolio_returns, 5)
        
        # Max drawdown
        cumulative = (1 + portfolio_returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        portfolio_max_drawdown = drawdown.min()
        
        # Correlation matrix
        correlation_matrix = returns_df.corr().to_dict()
        
        # Diversification ratio
        individual_vols = returns_df.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        weighted_vol_sum = sum(weights[t] * individual_vols[t] for t in returns_df.columns)
        diversification_ratio = weighted_vol_sum / portfolio_volatility if portfolio_volatility > 0 else None

        # Return contribution
        individual_returns = {t: returns_df[t].mean() * TRADING_DAYS_PER_YEAR for t in returns_df.columns}
        contributions = [(t, weights[t] * individual_returns[t]) for t in returns_df.columns]
        top_contributors = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        # Concentration risk (Herfindahl index)
        concentration_risk = sum(w**2 for w in weights.values())
        
        # Total value (assuming 100 shares base)
        total_value = sum(
            weights[ticker] * analysis.latest_close * 100
            for ticker, analysis in successful.items()
            if ticker in weights
        )
        
        logger.info(f"Portfolio: Return={portfolio_return*100:.2f}%, "
                   f"Vol={portfolio_volatility*100:.2f}%, Sharpe={portfolio_sharpe:.2f}")
        
        return PortfolioMetrics(
            total_value=total_value,
            portfolio_return=portfolio_return,
            portfolio_volatility=portfolio_volatility,
            portfolio_sharpe=portfolio_sharpe,
            portfolio_var_95=portfolio_var_95,
            portfolio_max_drawdown=portfolio_max_drawdown,
            diversification_ratio=diversification_ratio,
            correlation_matrix=correlation_matrix,
            weights=weights,
            top_contributors=top_contributors,
            concentration_risk=concentration_risk
        )
    
    def _simple_portfolio_metrics(self, analyses: Dict[str, TickerAnalysis], 
                                  weights: Dict[str, float]) -> PortfolioMetrics:
        """Simplified metrics when full analysis isn't possible."""
        total_value = sum(
            weights[ticker] * analysis.latest_close * 100
            for ticker, analysis in analyses.items()
            if ticker in weights
        )
        
        avg_return = sum(
            weights[t] * a.avg_daily_return 
            for t, a in analyses.items() if t in weights
        )
        
        avg_vol = sum(
            weights[t] * a.volatility 
            for t, a in analyses.items() if t in weights
        )
        
        return PortfolioMetrics(
            total_value=total_value,
            portfolio_return=avg_return * TRADING_DAYS_PER_YEAR,
            portfolio_volatility=avg_vol * np.sqrt(TRADING_DAYS_PER_YEAR),
            weights=weights
        )

