"""
Data models for options analysis.

This module contains immutable dataclasses for options-specific data structures,
following the same patterns as models.py.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================


class OptionType(str, Enum):
    """Option contract type."""
    CALL = "call"
    PUT = "put"


class StrategyType(str, Enum):
    """Identified options strategy types."""
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    CASH_SECURED_PUT = "cash_secured_put"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    BUTTERFLY_SPREAD = "butterfly_spread"
    COLLAR = "collar"
    RATIO_SPREAD = "ratio_spread"


# ============================================================================
# CORE DATA MODELS
# ============================================================================


@dataclass
class GreeksData:
    """Options Greeks for a single contract.

    All Greeks are dimensionless ratios except Theta (per day) and Vega (per 1% vol).
    """
    delta: Optional[float] = None  # Rate of change of option price w.r.t. underlying (0-1 for calls, -1-0 for puts)
    gamma: Optional[float] = None  # Rate of change of delta w.r.t. underlying
    theta: Optional[float] = None  # Rate of time decay (typically negative, $ per day)
    vega: Optional[float] = None   # Sensitivity to 1% change in implied volatility
    rho: Optional[float] = None    # Sensitivity to 1% change in interest rate

    # Additional derived metrics
    delta_dollars: Optional[float] = None  # Delta × contract multiplier (typically 100)
    gamma_dollars: Optional[float] = None  # Gamma × contract multiplier


@dataclass
class OptionsContract:
    """Individual options contract data."""

    # Contract identifiers
    ticker: str
    strike: float
    expiration: date
    option_type: OptionType

    # Market data
    last_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    # Derived metrics
    implied_volatility: Optional[float] = None  # Annualized IV as decimal (e.g., 0.25 = 25%)
    greeks: Optional[GreeksData] = None

    # Additional data
    last_trade_date: Optional[datetime] = None
    in_the_money: Optional[bool] = None
    intrinsic_value: Optional[float] = None
    extrinsic_value: Optional[float] = None
    moneyness: Optional[float] = None  # spot / strike (>1 ITM call, <1 ITM put)

    # Contract metadata
    contract_symbol: Optional[str] = None  # Full OCC symbol
    currency: str = "USD"

    @property
    def days_to_expiration(self) -> int:
        """Calculate days remaining until expiration."""
        return (self.expiration - date.today()).days

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.last_price


@dataclass
class OptionsChain:
    """Complete options chain for a single expiration date."""

    ticker: str
    expiration: date
    underlying_price: float

    calls: List[OptionsContract] = field(default_factory=list)
    puts: List[OptionsContract] = field(default_factory=list)

    # Aggregate metrics
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0
    put_call_ratio_volume: Optional[float] = None  # Put volume / Call volume
    put_call_ratio_oi: Optional[float] = None      # Put OI / Call OI

    # IV metrics
    atm_call_iv: Optional[float] = None
    atm_put_iv: Optional[float] = None
    iv_skew: Optional[float] = None  # ATM IV - OTM IV (typically negative for equity options)

    @property
    def days_to_expiration(self) -> int:
        """Calculate days remaining until expiration."""
        return (self.expiration - date.today()).days

    @property
    def all_contracts(self) -> List[OptionsContract]:
        """Get all contracts (calls + puts)."""
        return self.calls + self.puts


@dataclass
class StrategyLeg:
    """Single leg of an options strategy."""

    contract: OptionsContract
    quantity: int  # Positive for long, negative for short
    action: str    # "BUY" or "SELL"

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.quantity > 0

    @property
    def premium(self) -> Optional[float]:
        """Calculate premium for this leg (negative for debit, positive for credit)."""
        if self.contract.mid_price is None:
            return None
        # Long positions pay premium (negative), short positions receive premium (positive)
        return -self.contract.mid_price * abs(self.quantity) if self.is_long else self.contract.mid_price * abs(self.quantity)


@dataclass
class PnLScenario:
    """P&L scenario at expiration for a strategy."""

    underlying_price: float
    total_pnl: float  # Total P&L at this price
    pnl_percent: float  # P&L as percentage of capital at risk
    probability: Optional[float] = None  # Probability of reaching this price (from Monte Carlo)


@dataclass
class OptionsStrategy:
    """Identified options strategy with analysis."""

    strategy_type: StrategyType
    legs: List[StrategyLeg]

    # Entry metrics
    net_premium: float  # Net debit (negative) or credit (positive)
    capital_required: float  # Collateral/margin required

    # Risk metrics
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakeven_points: List[float] = field(default_factory=list)
    risk_reward_ratio: Optional[float] = None  # Max profit / Max loss

    # Probability metrics (from Monte Carlo simulation)
    probability_of_profit: Optional[float] = None
    expected_value: Optional[float] = None  # Expected P&L

    # Greeks aggregate
    total_delta: Optional[float] = None
    total_gamma: Optional[float] = None
    total_theta: Optional[float] = None
    total_vega: Optional[float] = None

    # P&L scenarios
    pnl_scenarios: List[PnLScenario] = field(default_factory=list)

    # Metadata
    description: str = ""
    recommendation_score: Optional[float] = None  # 0-100 score from LLM

    @property
    def is_debit_strategy(self) -> bool:
        """Check if this is a debit strategy (pay premium upfront)."""
        return self.net_premium < 0

    @property
    def is_credit_strategy(self) -> bool:
        """Check if this is a credit strategy (receive premium upfront)."""
        return self.net_premium > 0

    @property
    def return_on_capital(self) -> Optional[float]:
        """Calculate max return on capital at risk."""
        if self.max_profit is None or self.capital_required == 0:
            return None
        return self.max_profit / abs(self.capital_required)


@dataclass
class IVAnalysis:
    """Implied volatility analysis and skew metrics."""

    ticker: str
    current_iv: float  # ATM IV
    historical_volatility: float  # Realized volatility from stock data
    iv_percentile: Optional[float] = None  # Current IV percentile over lookback period

    # Skew metrics
    call_skew: Optional[float] = None  # OTM calls IV - ATM IV
    put_skew: Optional[float] = None   # OTM puts IV - ATM IV

    # Term structure
    iv_term_structure: Dict[date, float] = field(default_factory=dict)  # Expiration -> ATM IV

    # Comparison
    iv_rank: Optional[float] = None  # Where current IV ranks in historical range (0-100)
    iv_vs_hv_ratio: Optional[float] = None  # IV / HV (>1 means options expensive)


@dataclass
class TickerOptionsAnalysis:
    """Complete options analysis for a single ticker."""

    ticker: str
    underlying_price: float
    analysis_date: datetime

    # Options chains data
    chains: List[OptionsChain] = field(default_factory=list)

    # IV analysis
    iv_analysis: Optional[IVAnalysis] = None

    # Identified strategies
    strategies: List[OptionsStrategy] = field(default_factory=list)

    # Top opportunities (sorted by recommendation score)
    top_strategies: List[OptionsStrategy] = field(default_factory=list)

    # Chart paths
    chain_heatmap_path: Optional[Path] = None
    greeks_chart_path: Optional[Path] = None
    pnl_diagram_path: Optional[Path] = None
    iv_surface_path: Optional[Path] = None

    # Aggregate metrics
    total_contracts: int = 0
    total_volume: int = 0
    total_open_interest: int = 0

    # LLM-generated content
    executive_summary: str = ""
    strategy_recommendations: str = ""

    # Errors/warnings
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class PortfolioOptionsMetrics:
    """Aggregated options metrics across portfolio."""

    # Aggregate Greeks
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    total_rho: float = 0.0

    # Greeks exposure by ticker
    delta_by_ticker: Dict[str, float] = field(default_factory=dict)
    vega_by_ticker: Dict[str, float] = field(default_factory=dict)
    theta_by_ticker: Dict[str, float] = field(default_factory=dict)

    # Portfolio-level metrics
    net_portfolio_delta: float = 0.0  # Delta-adjusted exposure
    portfolio_gamma_risk: float = 0.0  # Aggregate gamma risk
    portfolio_theta_decay: float = 0.0  # Daily theta decay
    portfolio_vega_exposure: float = 0.0  # Volatility sensitivity

    # Hedging recommendations
    delta_hedge_required: Optional[float] = None  # Shares to buy/sell to neutralize delta
    vega_hedge_suggestion: str = ""
    theta_management_suggestion: str = ""

    # Concentration metrics
    largest_greek_exposure: Optional[Tuple[str, str, float]] = None  # (ticker, greek_name, value)
    greek_concentration_risk: Optional[float] = None  # Herfindahl index for Greeks

    # Portfolio-level strategies
    suggested_hedges: List[OptionsStrategy] = field(default_factory=list)

    # LLM-generated insights
    hedging_recommendations: str = ""
    risk_assessment: str = ""

    @property
    def is_delta_neutral(self) -> bool:
        """Check if portfolio is approximately delta-neutral."""
        return abs(self.net_portfolio_delta) < 0.1  # Within ±0.1 delta

    @property
    def is_vega_heavy(self) -> bool:
        """Check if portfolio has significant volatility exposure."""
        return abs(self.portfolio_vega_exposure) > 100  # Arbitrary threshold


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_moneyness(spot: float, strike: float, option_type: OptionType) -> float:
    """
    Calculate option moneyness.

    Args:
        spot: Current underlying price
        strike: Option strike price
        option_type: Call or Put

    Returns:
        Moneyness ratio (spot/strike for calls, strike/spot for puts)
        >1.0 means ITM, <1.0 means OTM, ~1.0 means ATM
    """
    if option_type == OptionType.CALL:
        return spot / strike
    else:  # PUT
        return strike / spot


def calculate_intrinsic_value(spot: float, strike: float, option_type: OptionType) -> float:
    """
    Calculate intrinsic value of an option.

    Args:
        spot: Current underlying price
        strike: Option strike price
        option_type: Call or Put

    Returns:
        Intrinsic value (always >= 0)
    """
    if option_type == OptionType.CALL:
        return max(0.0, spot - strike)
    else:  # PUT
        return max(0.0, strike - spot)


def is_itm(spot: float, strike: float, option_type: OptionType) -> bool:
    """
    Check if option is in-the-money.

    Args:
        spot: Current underlying price
        strike: Option strike price
        option_type: Call or Put

    Returns:
        True if ITM, False otherwise
    """
    if option_type == OptionType.CALL:
        return spot > strike
    else:  # PUT
        return spot < strike


def find_atm_strike(spot: float, available_strikes: List[float]) -> Optional[float]:
    """
    Find the at-the-money strike from available strikes.

    Args:
        spot: Current underlying price
        available_strikes: List of available strike prices

    Returns:
        Closest strike to spot price, or None if no strikes available
    """
    if not available_strikes:
        return None

    return min(available_strikes, key=lambda strike: abs(strike - spot))
