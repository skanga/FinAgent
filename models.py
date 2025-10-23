"""
Data models for financial analysis.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from constants import LimitsAndConstraints
from utils import validate_ticker_symbol, validate_ticker_list


# ============================================================================
# PYDANTIC MODELS (With Validation)
# ============================================================================


class TickerRequest(BaseModel):
    """Validated ticker analysis request."""

    ticker: str = Field(
        ..., min_length=1, max_length=10, description="Stock ticker symbol"
    )
    period: str = Field(
        default="1y", description="Analysis period (e.g., 1y, 6mo, ytd)"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """
        Validates the ticker symbol using centralized validation.

        Args:
            v (str): The ticker symbol to validate.

        Returns:
            str: The validated ticker symbol.
        """
        return validate_ticker_symbol(v)

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        """
        Validates the analysis period.

        Args:
            v (str): The analysis period to validate.

        Returns:
            str: The validated analysis period.
        """
        if v not in LimitsAndConstraints.VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{v}'. Must be one of: {', '.join(LimitsAndConstraints.VALID_PERIODS)}"
            )
        return v


class PortfolioRequest(BaseModel):
    """Validated portfolio analysis request."""

    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=LimitsAndConstraints.MAX_TICKERS_ALLOWED,
        description="List of ticker symbols",
    )
    period: str = Field(default="1y", description="Analysis period")
    weights: Optional[Dict[str, float]] = Field(
        default=None, description="Portfolio weights (must sum to 1.0)"
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        """
        Validates the list of ticker symbols using centralized validation.

        Args:
            v (List[str]): The list of ticker symbols to validate.

        Returns:
            List[str]: The validated list of ticker symbols.
        """
        return validate_ticker_list(v)

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        """
        Validates the analysis period.

        Args:
            v (str): The analysis period to validate.

        Returns:
            str: The validated analysis period.
        """
        if v not in LimitsAndConstraints.VALID_PERIODS:
            raise ValueError(
                f"Invalid period '{v}'. Must be one of: {', '.join(LimitsAndConstraints.VALID_PERIODS)}"
            )
        return v

    @model_validator(mode="after")
    def validate_weights(self) -> "PortfolioRequest":
        """
        Validates the portfolio weights.

        Returns:
            PortfolioRequest: The validated portfolio request.
        """
        if self.weights is None:
            return self

        # Check that weights keys match tickers
        weight_tickers = set(self.weights.keys())
        request_tickers = set(self.tickers)

        if weight_tickers != request_tickers:
            missing = request_tickers - weight_tickers
            extra = weight_tickers - request_tickers
            errors = []
            if missing:
                errors.append(f"Missing weights for: {', '.join(missing)}")
            if extra:
                errors.append(f"Extra weights for: {', '.join(extra)}")
            raise ValueError("; ".join(errors))

        # Check individual weights
        for ticker, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Weight for {ticker} cannot be negative: {weight}")
            if weight > 1:
                raise ValueError(f"Weight for {ticker} cannot exceed 1.0: {weight}")

        # Check sum
        total = sum(self.weights.values())
        tolerance = 0.01  # Allow 1% deviation
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"Weights must sum to 1.0 (±{tolerance}), got {total:.4f}")

        return self


class NaturalLanguageRequest(BaseModel):
    """Validated natural language request."""

    query: str = Field(
        ..., min_length=5, max_length=500, description="Natural language query"
    )
    output_dir: str = Field(
        default="./financial_reports", description="Output directory for reports"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """
        Validates the natural language query.

        Args:
            v (str): The natural language query to validate.

        Returns:
            str: The validated natural language query.
        """
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


# ============================================================================
# DATACLASS MODELS (Legacy - For Internal Use)
# ============================================================================


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
class FundamentalData:
    """Parsed fundamental financial data."""

    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    shareholders_equity: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None


@dataclass
class TickerAnalysis:
    """Analysis results for a single ticker."""

    ticker: str
    csv_path: Path
    chart_path: Path
    latest_close: float
    avg_daily_return: float
    volatility: float
    ratios: Dict[str, Optional[float]]
    fundamentals: Optional[FundamentalData]
    advanced_metrics: AdvancedMetrics
    technical_indicators: TechnicalIndicators
    comparative_analysis: Optional[ComparativeAnalysis] = None
    sample_data: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class PortfolioMetrics:
    """Portfolio-level analysis metrics."""

    total_value: float
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: Optional[float] = None
    portfolio_var_95: Optional[float] = None
    portfolio_max_drawdown: Optional[float] = None
    diversification_ratio: Optional[float] = None
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    weights: Optional[Dict[str, float]] = None
    top_contributors: Optional[List[Tuple[str, float]]] = None
    concentration_risk: Optional[float] = None


@dataclass
class ParsedRequest:
    """Parsed natural language request."""

    tickers: List[str]
    period: str
    metrics: List[str]
    output_format: str = "markdown"

    def validate(self) -> None:
        """
        Validates the parsed request.

        Raises:
            ValueError: If the parsed request is invalid.
        """
        if not self.tickers:
            raise ValueError("At least one ticker must be specified")
        if len(self.tickers) > LimitsAndConstraints.MAX_TICKERS_ALLOWED:
            raise ValueError(f"Maximum {LimitsAndConstraints.MAX_TICKERS_ALLOWED} tickers allowed")


@dataclass
class ReportMetadata:
    """Metadata for generated report."""

    final_markdown_path: Path
    charts: List[Path]
    analyses: Dict[str, TickerAnalysis]
    portfolio_metrics: Optional[PortfolioMetrics]
    review_issues: List[str]
    generated_at: str
    performance_metrics: Dict[str, Any]
