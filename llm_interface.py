"""
LLM integration for narrative generation and parsing.
"""

import json
import logging
import httpx
import pandas as pd
from pathlib import Path
from config import Config
from httpx import Limits, Timeout
from langchain_openai import ChatOpenAI
from typing import Dict, List, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from models import ParsedRequest, TickerAnalysis, PortfolioMetrics
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class IntegratedLLMInterface:
    """LLM interface with connection pooling for better performance."""

    def __init__(
        self,
        config: Config,
        max_connections: int = 20,
        max_keepalive_connections: int = 10,
    ) -> None:
        """
        Initializes the IntegratedLLMInterface with a configuration object and connection pooling settings.

        Args:
            config (Config): The configuration object.
            max_connections (int): The maximum number of concurrent connections.
            max_keepalive_connections (int): The maximum number of idle connections to keep alive.
        """
        self.config = config

        # Create HTTP client with connection pooling
        # LangChain's ChatOpenAI uses httpx internally
        http_client = self._create_pooled_http_client(
            max_connections, max_keepalive_connections
        )

        # Initialize LLM with connection pooling
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.model_name,
            temperature=0.3,
            timeout=config.request_timeout,
            http_client=http_client,  # Use our pooled client
            http_async_client=None,  # We're not using async
        )

        # Log LLM setup (without exposing API key)
        provider = config._get_provider_from_url()
        logger.info(
            f"LLM initialized - Provider: {provider}, Model: {config.model_name}, "
            f"Temperature: 0.3, Connection pool: {max_keepalive_connections}/{max_connections}"
        )

        self._setup_prompts()
        self._setup_chains()

    def _create_pooled_http_client(
        self, max_connections: int = 20, max_keepalive: int = 10
    ) -> "httpx.Client":
        """Create an httpx client with connection pooling.

        Args:
            max_connections: Maximum total connections
            max_keepalive: Maximum idle connections to keep alive

        Returns:
            httpx.Client with connection pooling configured
        """
        import httpx

        # Configure connection limits
        limits = Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=30.0,  # Keep connections alive for 30 seconds
        )

        # Configure timeout
        timeout = Timeout(
            connect=10.0,  # Connection timeout
            read=self.config.request_timeout,  # Read timeout
            write=10.0,  # Write timeout
            pool=5.0,  # Pool timeout
        )

        # Create client with pooling
        client = httpx.Client(
            limits=limits,
            timeout=timeout,
            http2=True,  # Enable HTTP/2 for multiplexing
            follow_redirects=True,
        )

        logger.info(
            f"Created HTTP client with connection pool "
            f"(max: {max_connections}, keepalive: {max_keepalive})"
        )

        return client

    def _setup_prompts(self) -> None:
        """Setup prompt templates."""
        self.parse_prompt = PromptTemplate(
            input_variables=["user_request"],
            template="""Parse this financial analysis request into structured parameters.

User Request: {user_request}

Return ONLY valid JSON with:
- tickers: list of stock symbols (uppercase, e.g., ["AAPL", "MSFT"])
- period: one of [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
- metrics: list of requested metrics
- output_format: "markdown" or "pdf"

Example: {{"tickers": ["AAPL"], "period": "1y", "metrics": ["returns", "risk"], "output_format": "markdown"}}

Return ONLY the JSON, no other text.""",
        )

        self.narrative_prompt = PromptTemplate(
            input_variables=["analysis_data", "period"],
            template="""You are a senior financial analyst. Generate a compelling executive summary.

Analysis Period: {period}
Data: {analysis_data}

IMPORTANT GUIDELINES:

1. **Metric Definitions (use these correctly):**
   - **avg_daily_return**: This is the DAILY average return (NOT annualized).
     * Formula for annualized return: ((1 + avg_daily_return/100)^252 - 1) × 100
     * Example: If avg_daily_return = 0.384%, then annualized = ((1.00384)^252 - 1) × 100 = 162.7%
     * Example: If avg_daily_return = 0.429%, then annualized = ((1.00429)^252 - 1) × 100 = 194.1%
     * ALWAYS use this formula, DO NOT multiply by 252 or use simple scaling
   - **daily_volatility**: This is DAILY std dev (NOT annualized). To discuss annualized volatility, calculate: daily_volatility × √252
   - **sharpe_ratio**: This is the annualized Sharpe ratio (calculated from annualized returns and volatility, already annualized, don't scale it)
   - VaR and max drawdown are percentages (VaR is shown as negative)
   - RSI: 0-100 scale where exact value matters
   - P/E: Price-to-Earnings ratio (higher = more expensive relative to earnings)
   - P/B: Price-to-Book ratio (higher = more expensive relative to book value, NOT "compression")

2. **Technical Interpretation Rules - Be Precise:**
   - RSI: Use EXACT values from data. Interpretation scale:
     * <30: Oversold
     * 30-45: Weak/bearish zone
     * 45-55: Neutral zone
     * 55-70: Bullish zone (NOT overbought)
     * >70: Overbought
     DO NOT group tickers with RSI 47 and 65 into the same "neutral-bullish" category. They are in different zones!
   - Sharpe >1: Good risk-adjusted returns
   - Max DD >-20%: Moderate risk, <-20%: High risk
   - P/E ratio: High values (>25) = expensive relative to earnings, Low (<15) = cheap
   - P/B ratio: High values (>5) = expensive relative to book value (high valuation), Low (<1) = cheap (possible value opportunity). NEVER say "compression" for high P/B - that's the opposite!

3. **CRITICAL: Consistency Requirements:**
   - Before writing, identify the ticker with highest avg_daily_return from the data
   - Before writing, identify the ticker with lowest avg_daily_return from the data
   - Use these SAME tickers consistently throughout the narrative
   - NEVER claim ticker A has "highest returns" in one place and ticker B has "highest returns" elsewhere
   - If you mention a metric for a ticker, use the EXACT value from the data (no rounding changes)

4. **When citing numbers:**
   - State "daily" or "annualized" explicitly
   - Match EXACT values from the data - no rounding changes
   - If converting (e.g., daily to annual), show the calculation
   - For unusual values, cite accurately and note they're unusual

5. **Write 3-4 paragraphs that:**
   - Highlight significant findings with precise numbers
   - Use CORRECT interpretations (e.g., don't call RSI 60 "overbought")
   - Compare performance across tickers
   - Identify risks with supporting data

6. **Avoid these common errors:**
   - ❌ Calling RSI 50-70 "overbought" (it's neutral/bullish)
   - ❌ Using different values for same metric in different places
   - ❌ Contradicting yourself (e.g., saying ticker A has highest return, then saying ticker B has highest return)
   - ❌ Making claims without data support (e.g., "above market average" without providing market average)
   - ❌ Confusing daily and annualized figures
   - ❌ CRITICAL: Incorrect annualization math (e.g., claiming 0.384% daily = 274% annual is WRONG, correct is 162.7%)

7. **Before finalizing, verify:**
   - All annualized return calculations use the correct formula: ((1 + daily%/100)^252 - 1) × 100
   - All numbers match the source data exactly
   - No calculation errors

Be precise, accurate, and professional. Use correct technical definitions.
Return ONLY the narrative text.""",
        )

        self.review_prompt = PromptTemplate(
            input_variables=["report_content", "data_summary"],
            template="""Review this financial report for accuracy and quality.

Report (excerpt):
{report_content}

Source Data Summary:
{data_summary}

IMPORTANT: Distinguish between CRITICAL ISSUES and SUGGESTIONS FOR IMPROVEMENT.

**CRITICAL ISSUES** (report in "issues" field):
Only flag items that are FACTUALLY WRONG or MISLEADING:
1. Numbers that don't match source data (>5% discrepancy)
2. Incorrect interpretations (e.g., "RSI 60 is overbought" when overbought is >70)
3. Mathematical errors in calculations
4. Contradictory statements in the same report
5. Misleading claims not supported by data

**SUGGESTIONS** (report in "suggestions" field):
Items that would ENHANCE the report but are not errors:
1. Missing methodology explanations (calculations are correct but process not explained)
2. Missing context (e.g., no industry comparison for P/E ratio)
3. Ambiguous labels (e.g., table headers could be clearer)
4. Additional disclosures that would improve transparency
5. Formatting or presentation improvements

EXAMPLES:

❌ CRITICAL ISSUE: "Report states RSI of 59.5 is 'overbought' but overbought is typically >70"
✅ SUGGESTION: "Report could clarify the methodology for calculating Sharpe ratio"

❌ CRITICAL ISSUE: "Annualized return calculated as 18% but source shows 0.05% daily return (should be ~13%)"
✅ SUGGESTION: "Table headers could specify whether metrics are daily or annualized"

❌ CRITICAL ISSUE: "Report claims P/E is 'well above market average' without providing market average"
✅ SUGGESTION: "P/E ratio could include industry average for comparison"

Return valid JSON with:
- "issues": list of CRITICAL ERRORS only (factually wrong, misleading, contradictory)
- "suggestions": list of optional improvements (missing context, methodology notes, clarity enhancements)
- "quality_score": integer 1-10
  * 9-10: Excellent, no critical issues, minimal suggestions
  * 7-8: Good, no critical issues, some helpful suggestions
  * 5-6: Acceptable, 1-2 minor critical issues or many clarity concerns
  * <5: Poor, multiple critical errors

Be precise: Only flag as "issues" if the report is factually wrong or misleading.

Return ONLY the JSON.""",
        )

        # Options-specific prompts
        self.options_narrative_prompt = PromptTemplate(
            input_variables=["options_data", "ticker", "spot_price"],
            template="""You are an options trading expert. Generate an executive summary of options opportunities.

Ticker: {ticker}
Current Price: ${spot_price}
Options Data: {options_data}

Generate a concise executive summary covering:

1. **Implied Volatility Assessment**
   - Current IV vs historical volatility
   - IV skew patterns (put vs call)
   - Whether options are expensive or cheap

2. **Top Strategy Opportunities** (choose 2-3 most attractive)
   - Strategy name and structure
   - Entry cost and max profit/loss
   - Breakeven points
   - Why this strategy makes sense now

   IMPORTANT: If you include a markdown table, ensure ALL rows have EXACTLY the same number of columns as the header row.
   Count the pipes (|) carefully - every row must have the same count.

3. **Greeks Summary**
   - Key risk exposures (Delta, Vega, Theta)
   - How Greeks affect position value

4. **Risk Assessment**
   - Primary risks for options traders
   - Time decay considerations
   - Volatility risks

Be specific with numbers. Focus on actionable insights.
Return ONLY the narrative text, no JSON.""",
        )

        self.options_recommendations_prompt = PromptTemplate(
            input_variables=["ticker", "strategies", "market_context", "portfolio_holdings"],
            template="""You are an options strategist. Provide personalized strategy recommendations.

Ticker: {ticker}
Identified Strategies: {strategies}
Market Context: {market_context}
Current Holdings: {portfolio_holdings}

Provide 3-5 specific recommendations ranked by attractiveness:

For each recommendation:
1. **Strategy Name** (e.g., "Covered Call - $150 Strike")
2. **Entry Details**: Exact strikes, expiration, premiums
3. **Rationale**: Why this strategy fits the current market conditions
4. **Risk/Reward**: Max profit, max loss, breakeven
5. **Probability of Profit**: Estimated likelihood of success
6. **Best For**: Type of investor (income, speculation, hedging)

CRITICAL TABLE FORMATTING RULES:
- If you create a markdown table, EVERY data row MUST have EXACTLY the same number of columns (pipe characters |) as the header row
- Example: If header has 8 pipes (7 columns), every data row must have 8 pipes
- Do NOT merge cells or skip columns - provide data for every column in every row
- If a value is unknown, use "N/A" or "-" rather than omitting the column

Consider:
- Implied volatility levels (high IV favors selling, low IV favors buying)
- Time to expiration (theta decay)
- Existing portfolio positions (hedging opportunities)
- Current price trends and momentum

Be direct and actionable. Focus on strategies with good risk/reward.
Return ONLY the narrative text.""",
        )

        self.portfolio_hedging_prompt = PromptTemplate(
            input_variables=["portfolio_greeks", "positions", "risk_tolerance"],
            template="""You are a portfolio risk manager. Provide hedging recommendations.

Portfolio Greeks: {portfolio_greeks}
Current Positions: {positions}
Risk Tolerance: {risk_tolerance}

Analyze the portfolio's options exposure and provide:

1. **Current Risk Profile**
   - Net Delta exposure (directional risk)
   - Vega exposure (volatility risk)
   - Theta decay (time risk)
   - Gamma risk (how Delta changes)

2. **Hedging Priorities** (rank top 3)
   - Which Greek needs immediate attention
   - Why it's a priority
   - Potential impact if unhedged

3. **Specific Hedging Strategies**
   For each priority:
   - Exact strategy (e.g., "Buy 100 shares SPY to neutralize delta")
   - Cost and effectiveness
   - Alternative approaches

4. **Portfolio Balancing**
   - Concentration risks (if one ticker dominates Greeks)
   - Diversification opportunities
   - Suggested position adjustments

Be precise with numbers. Provide implementable hedges.
Return ONLY the narrative text.""",
        )

    def _setup_chains(self) -> None:
        """Setup LangChain chains."""
        self.parser_chain = self.parse_prompt | self.llm
        self.narrative_chain = self.narrative_prompt | self.llm
        self.review_chain = self.review_prompt | self.llm

        # Options chains
        self.options_narrative_chain = self.options_narrative_prompt | self.llm
        self.options_recommendations_chain = self.options_recommendations_prompt | self.llm
        self.portfolio_hedging_chain = self.portfolio_hedging_prompt | self.llm

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def parse_natural_language_request(self, user_request: str) -> ParsedRequest:
        """
        Parses a natural language request into a structured format.

        Automatically retries up to 3 times with exponential backoff for network errors.

        Args:
            user_request (str): The natural language request from the user.

        Returns:
            ParsedRequest: A structured request object.

        Raises:
            ValueError: If parsing fails after retries
            ConnectionError: If network errors persist after retries
        """
        logger.info("Parsing natural language request")

        try:
            response = self.parser_chain.invoke({"user_request": user_request})
            parsed_text = response.content.strip()

            # Clean markdown code blocks
            if "```json" in parsed_text:
                parsed_text = parsed_text.split("```json")[1].split("```")[0]
            elif "```" in parsed_text:
                parsed_text = parsed_text.split("```")[1].split("```")[0]

            parsed_json = json.loads(parsed_text)

            parsed = ParsedRequest(
                tickers=[t.upper().strip() for t in parsed_json.get("tickers", [])],
                period=parsed_json.get("period", "1y"),
                metrics=parsed_json.get("metrics", []),
                output_format=parsed_json.get("output_format", "markdown"),
            )

            parsed.validate()
            logger.info(f"Parsed request: {parsed.tickers}")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            raise ValueError("Failed to parse JSON response") from e
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Parse error: {e}")
            raise ValueError(f"Parse error: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate_narrative_summary(
        self, analyses: Dict[str, TickerAnalysis], period: str
    ) -> str:
        """
        Generates a narrative summary of the financial analysis using the LLM.

        Automatically retries up to 3 times with exponential backoff for network errors.

        Args:
            analyses (Dict[str, TickerAnalysis]): A dictionary of ticker analyses.
            period (str): The analysis period.

        Returns:
            str: The narrative summary.
        """
        logger.info("Generating narrative summary with LLM")

        # Prepare concise data for LLM with clear labels
        analysis_summary = {}
        for ticker, analysis in analyses.items():
            if not analysis.error:
                analysis_summary[ticker] = {
                    "current_price": f"${analysis.latest_close:.2f}",
                    "avg_daily_return": f"{analysis.avg_daily_return * 100:.3f}%",
                    "daily_volatility": f"{analysis.volatility * 100:.2f}%",
                    "sharpe_ratio": (
                        round(analysis.advanced_metrics.sharpe_ratio, 2)
                        if analysis.advanced_metrics.sharpe_ratio
                        else None
                    ),
                    "max_drawdown_pct": f"{(analysis.advanced_metrics.max_drawdown or 0) * 100:.1f}%",
                    "rsi": (
                        round(analysis.technical_indicators.rsi, 1)
                        if analysis.technical_indicators.rsi
                        else None
                    ),
                    "pe_ratio": (
                        round(analysis.ratios.get("pe_ratio"), 1)
                        if analysis.ratios.get("pe_ratio")
                        else None
                    ),
                    "pb_ratio": (
                        round(analysis.ratios.get("price_to_book"), 2)
                        if analysis.ratios.get("price_to_book")
                        else None
                    ),
                    "alerts": analysis.alerts,
                }

        try:
            response = self.narrative_chain.invoke(
                {
                    "analysis_data": json.dumps(analysis_summary, indent=2),
                    "period": period,
                }
            )

            narrative = response.content.strip()
            logger.info("Generated narrative summary")
            return narrative

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error(f"Failed to generate narrative: {e}")
            return self._fallback_narrative(analyses, period)

    def _fallback_narrative(
        self, analyses: Dict[str, TickerAnalysis], period: str
    ) -> str:
        """Fallback narrative if LLM fails."""
        successful = {t: a for t, a in analyses.items() if not a.error}
        if not successful:
            return "Analysis completed but no valid data available."

        best_performing_ticker = max(
            successful.items(), key=lambda x: x[1].avg_daily_return
        )
        worst_performing_ticker = min(
            successful.items(), key=lambda x: x[1].avg_daily_return
        )

        return f"""Over the {period} period, {len(successful)} stocks were analyzed.
{best_performing_ticker[0]} showed the strongest performance with a {best_performing_ticker[1].avg_daily_return*100:.2f}% average daily return,
while {worst_performing_ticker[0]} had the weakest returns at {worst_performing_ticker[1].avg_daily_return*100:.2f}%.
Risk metrics varied across the portfolio, with several stocks showing elevated volatility warranting monitoring."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def review_report(
        self, report_content: str, analyses: Dict[str, TickerAnalysis]
    ) -> Tuple[List[str], int, Dict]:
        """
        Reviews a financial report for accuracy and quality using the LLM.

        Automatically retries up to 3 times with exponential backoff for network errors.

        Args:
            report_content (str): The content of the report to review.
            analyses (Dict[str, TickerAnalysis]): A dictionary of ticker analyses.

        Returns:
            Tuple[List[str], int, Dict]: A tuple containing a list of issues, a quality score, and the full review dictionary.
        """
        logger.info("Reviewing report with LLM")

        # Prepare data summary
        data_summary = {}
        for ticker, analysis in analyses.items():
            if not analysis.error:
                data_summary[ticker] = {
                    "latest_close": analysis.latest_close,
                    "avg_daily_return": analysis.avg_daily_return,
                    "daily_volatility": analysis.volatility,
                    "sharpe_ratio": analysis.advanced_metrics.sharpe_ratio,
                    # Include ALL technical indicators so review can verify claims
                    "rsi": analysis.technical_indicators.rsi,
                    "stochastic_k": analysis.technical_indicators.stochastic_k,
                    "stochastic_d": analysis.technical_indicators.stochastic_d,
                    "macd": analysis.technical_indicators.macd,
                    "macd_signal": analysis.technical_indicators.macd_signal,
                    "bollinger_upper": analysis.technical_indicators.bollinger_upper,
                    "bollinger_lower": analysis.technical_indicators.bollinger_lower,
                    "bollinger_position": analysis.technical_indicators.bollinger_position,
                    "atr": analysis.technical_indicators.atr,
                    "obv": analysis.technical_indicators.obv,
                    # "vwap": removed - not appropriate for daily data
                    "ma_200d": analysis.technical_indicators.ma_200d,
                    "pe_ratio": analysis.ratios.get("pe_ratio"),
                    "pb_ratio": analysis.ratios.get("price_to_book"),
                }

        # Warn if report content is truncated
        original_length = len(report_content)
        max_review_length = 4000
        if original_length > max_review_length:
            logger.warning(
                f"Report content truncated for review: {original_length} chars -> {max_review_length} chars "
                f"({original_length - max_review_length} chars omitted)"
            )

        try:
            response = self.review_chain.invoke(
                {
                    "report_content": report_content[:max_review_length],
                    "data_summary": json.dumps(data_summary, indent=2),
                }
            )

            review_text = response.content.strip()

            # Clean markdown
            if "```json" in review_text:
                review_text = review_text.split("```json")[1].split("```")[0]
            elif "```" in review_text:
                review_text = review_text.split("```")[1].split("```")[0]

            review_json = json.loads(review_text)
            issues = review_json.get("issues", [])
            suggestions = review_json.get("suggestions", [])
            quality_score = review_json.get("quality_score", 7)

            logger.info(
                f"Review: {len(issues)} issues, {len(suggestions)} suggestions, quality: {quality_score}/10"
            )

            # Return full review data
            full_review = {
                "issues": issues,
                "suggestions": suggestions,
                "quality_score": quality_score,
            }
            return issues, quality_score, full_review

        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            OSError,
        ) as e:
            logger.error(f"Review failed: {e}")
            return (
                [],
                7,
                {"issues": [], "suggestions": [], "quality_score": 7, "error": str(e)},
            )

    def _generate_report_header(self) -> List[str]:
        """Generate report header with timestamp.

        Returns:
            List of header lines
        """
        from datetime import datetime, timezone

        return [
            "# 📊 Financial Analysis Report",
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
        ]

    def _generate_executive_summary_section(
        self, analyses: Dict[str, TickerAnalysis], period: str
    ) -> List[str]:
        """Generate executive summary section with LLM narrative.

        Args:
            analyses: All analysis results
            period: Analysis period

        Returns:
            List of section lines
        """
        narrative = self.generate_narrative_summary(analyses, period)
        return ["## 🎯 Executive Summary", narrative, ""]

    def _generate_portfolio_overview_section(
        self, portfolio_metrics: Optional[PortfolioMetrics]
    ) -> List[str]:
        """Generate portfolio overview section if available.

        Args:
            portfolio_metrics: Portfolio metrics or None

        Returns:
            List of section lines (empty if no portfolio metrics)
        """
        if not portfolio_metrics:
            return []

        return [
            "## 💼 Portfolio Overview",
            self._generate_portfolio_section(portfolio_metrics),
            "",
        ]

    def _generate_alerts_section(
        self, analyses: Dict[str, TickerAnalysis]
    ) -> List[str]:
        """Generate active alerts section.

        Args:
            analyses: All analysis results

        Returns:
            List of section lines (empty if no alerts)
        """
        all_alerts = [
            a
            for analysis in analyses.values()
            if not analysis.error
            for a in analysis.alerts
        ]
        if not all_alerts:
            return []

        section = ["## 🚨 Active Alerts"]
        for alert in all_alerts:
            section.append(f"- {alert}")
        section.append("")
        return section

    def _generate_failures_section(
        self, analyses: Dict[str, TickerAnalysis]
    ) -> List[str]:
        """Generate data quality notes for failed analyses.

        Args:
            analyses: All analysis results

        Returns:
            List of section lines (empty if no failures)
        """
        failures = {t: a for t, a in analyses.items() if a.error}
        if not failures:
            return []

        section = ["## ⚠️ Data Quality Notes"]
        for ticker, analysis in failures.items():
            section.append(f"- **{ticker}:** {analysis.error}")
        section.append("")
        return section

    def generate_detailed_report(
        self,
        analyses: Dict[str, TickerAnalysis],
        benchmark_analysis: Optional[TickerAnalysis],
        portfolio_metrics: Optional[PortfolioMetrics],
        period: str,
        comparison_chart_path: Optional[Path] = None,
    ) -> str:
        """
        Generates a comprehensive financial report with an LLM-powered narrative.

        Args:
            analyses (Dict[str, TickerAnalysis]): A dictionary of ticker analyses.
            benchmark_analysis (Optional[TickerAnalysis]): The benchmark analysis.
            portfolio_metrics (Optional[PortfolioMetrics]): The portfolio metrics.
            period (str): The analysis period.

        Returns:
            str: The complete markdown report.
        """
        report = []

        # Header
        report.extend(self._generate_report_header())

        # Executive Summary
        report.extend(self._generate_executive_summary_section(analyses, period))

        # Portfolio Overview (if available)
        report.extend(self._generate_portfolio_overview_section(portfolio_metrics))

        # Key Metrics Table
        report.extend(self._generate_metrics_section(analyses))

        # Fundamental Analysis
        report.append("## 📊 Fundamental Analysis")
        report.append(self._generate_fundamental_section(analyses))
        report.append("")

        # Individual Analysis
        report.append("## 📋 Detailed Stock Analysis")
        report.append(self._generate_individual_analysis(analyses))
        report.append("")

        # Charts Section
        report.extend(self._generate_charts_section(analyses, comparison_chart_path))

        # Options Analysis Section (if available)
        report.extend(self._generate_options_section(analyses))

        # Risk Analysis
        report.append("## ⚠️ Risk Analysis")
        report.append(self._generate_risk_analysis(analyses))
        report.append("")

        # Recommendations
        report.append("## 💡 Investment Recommendations")
        report.append(self._generate_recommendations(analyses))
        report.append("")

        # Alerts (if any)
        report.extend(self._generate_alerts_section(analyses))

        # Failures (if any)
        report.extend(self._generate_failures_section(analyses))

        return "\n".join(report)

    def _generate_portfolio_section(self, portfolio_metrics: PortfolioMetrics) -> str:
        """Generate portfolio overview section."""
        lines = []
        lines.append(f"**Total Value:** ${portfolio_metrics.total_value:,.2f}")
        lines.append(
            f"**Portfolio Return:** {portfolio_metrics.portfolio_return*100:.2f}%"
        )
        lines.append(
            f"**Portfolio Volatility:** {portfolio_metrics.portfolio_volatility*100:.2f}%"
        )

        if portfolio_metrics.portfolio_sharpe:
            lines.append(
                f"**Portfolio Sharpe Ratio:** {portfolio_metrics.portfolio_sharpe:.2f}"
            )

        if portfolio_metrics.diversification_ratio:
            lines.append(
                f"**Diversification Ratio:** {portfolio_metrics.diversification_ratio:.2f}"
            )

        if portfolio_metrics.concentration_risk:
            lines.append(
                f"**Concentration Risk (HHI):** {portfolio_metrics.concentration_risk:.3f}"
            )

        if portfolio_metrics.top_contributors:
            lines.append("\n**Top Contributors:**")
            for ticker, contribution in portfolio_metrics.top_contributors[:3]:
                lines.append(
                    f"- {ticker}: {contribution*100:.2f}% contribution to return"
                )

        return "\n".join(lines)

    def _generate_metrics_section(self, analyses: Dict[str, TickerAnalysis]) -> List[str]:
        """Generate complete metrics section with table and methodology notes.

        Args:
            analyses: All analysis results

        Returns:
            List of section lines
        """
        section = [
            "## 📈 Key Performance Metrics",
            self._generate_metrics_table(analyses),
            "",
        ]

        section.extend(self._generate_methodology_notes())
        section.append("")

        return section

    def _generate_methodology_notes(self) -> List[str]:
        """Generate methodology notes explaining metrics.

        Returns:
            List of methodology note lines
        """
        return [
            "### Methodology Notes",
            "",
            "**Returns and Volatility (Daily Metrics):**",
            "- **Return (Daily %):** Average daily return from historical price data. These are DAILY figures, not annualized.",
            "- **Vol (Daily %):** Standard deviation of daily returns. This is a DAILY figure, not annualized.",
            "- **Annualization Method:** To convert daily metrics to annual: Returns use geometric compounding (1 + daily_return)^252 - 1. Volatility is multiplied by √252 ≈ 15.87.",
            "",
            "**Risk-Adjusted Metrics:**",
            "- **Sharpe Ratio:** Risk-adjusted return metric calculated as (annualized return - risk-free rate) / annualized volatility. Values >1 indicate good risk-adjusted performance, >2 is excellent. This metric is already annualized.",
            "- **Max DD (%):** Maximum peak-to-trough decline during the analysis period (shown as negative percentage).",
            "- **VaR (95%):** Value at Risk at 95% confidence level - daily potential loss in worst 5% of cases (shown as negative percentage). Based on 5th percentile of historical daily returns distribution.",
            "",
            "**Technical and Fundamental Metrics:**",
            "- **RSI:** Relative Strength Index (0-100 scale). Interpretation: <30 = oversold, 30-45 = weak, 45-55 = neutral, 55-70 = bullish, >70 = overbought.",
            "- **P/E (TTM):** Price-to-Earnings ratio based on trailing twelve months. Higher values indicate higher valuation relative to earnings.",
            "- **P/B:** Price-to-Book ratio. Higher values (e.g., >5) indicate high valuation relative to book value, not compression.",
            "- **Rev Growth (YoY %):** Year-over-year revenue growth rate from quarterly financial statements.",
            "",
            "**Portfolio Metrics:**",
            "- **Diversification Ratio:** Ratio of weighted average volatility to portfolio volatility. Values >1 indicate diversification benefit.",
            "- **Concentration Risk (HHI):** Herfindahl-Hirschman Index based on portfolio weights. Values closer to 1 indicate higher concentration, closer to 0 indicate better diversification.",
            "",
            "**Calculation Assumptions:** All metrics assume 252 trading days per year and a risk-free rate of 2% (annualized).",
        ]

    def _generate_metrics_table(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate comprehensive metrics table."""
        table = [
            "| Ticker | Price ($) | Return (Daily %) | Vol (Daily %) | Sharpe | Max DD (%) | RSI | P/E (TTM) | Rev Growth (YoY %) |",
            "|--------|-----------|------------------|---------------|--------|------------|-----|-----------|---------------------|",
        ]

        for ticker, analysis in analyses.items():
            if analysis.error:
                continue

            metrics = analysis.advanced_metrics
            indicators = analysis.technical_indicators
            ratios = analysis.ratios
            fundamentals = analysis.fundamentals

            # Safe conversions with NaN checks
            sharpe_str = (
                f"{metrics.sharpe_ratio:.2f}"
                if metrics.sharpe_ratio is not None
                and not pd.isna(metrics.sharpe_ratio)
                else "N/A"
            )
            dd_val = (
                metrics.max_drawdown
                if metrics.max_drawdown is not None
                and not pd.isna(metrics.max_drawdown)
                else 0
            )
            dd_str = f"{dd_val*100:.1f}%"
            rsi_str = (
                f"{indicators.rsi:.0f}"
                if indicators.rsi is not None and not pd.isna(indicators.rsi)
                else "N/A"
            )
            pe_str = (
                f"{ratios.get('pe_ratio'):.1f}"
                if ratios.get("pe_ratio") and not pd.isna(ratios.get("pe_ratio"))
                else "N/A"
            )
            # Check for valid revenue_growth (not None, not NaN)
            has_growth = (
                fundamentals
                and fundamentals.revenue_growth is not None
                and not pd.isna(fundamentals.revenue_growth)
            )
            growth_str = (
                f"{fundamentals.revenue_growth*100:.1f}%" if has_growth else "N/A"
            )

            table.append(
                f"| {ticker} | "
                f"${analysis.latest_close:.2f} | "
                f"{analysis.avg_daily_return*100:.2f}% | "
                f"{analysis.volatility*100:.1f}% | "
                f"{sharpe_str} | "
                f"{dd_str} | "
                f"{rsi_str} | "
                f"{pe_str} | "
                f"{growth_str} |"
            )

        return "\n".join(table)

    def _generate_fundamental_section(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate fundamental analysis section."""
        lines = []

        for ticker, analysis in analyses.items():
            if analysis.error or not analysis.fundamentals:
                continue

            fundamentals = analysis.fundamentals
            if not any(
                [
                    fundamentals.revenue,
                    fundamentals.net_income,
                    fundamentals.revenue_growth,
                ]
            ):
                continue

            lines.append(f"### {ticker}")

            if fundamentals.revenue:
                lines.append(f"- **Revenue:** ${fundamentals.revenue/1e9:.2f}B")
            if fundamentals.net_income:
                lines.append(f"- **Net Income:** ${fundamentals.net_income/1e9:.2f}B")
            if fundamentals.revenue_growth:
                lines.append(
                    f"- **Revenue Growth (YoY):** {fundamentals.revenue_growth*100:.1f}%"
                )
            if fundamentals.earnings_growth:
                lines.append(
                    f"- **Earnings Growth (YoY):** {fundamentals.earnings_growth*100:.1f}%"
                )
            if fundamentals.free_cash_flow:
                lines.append(
                    f"- **Free Cash Flow:** ${fundamentals.free_cash_flow/1e9:.2f}B"
                )

            lines.append("")

        return "\n".join(lines) if lines else "Limited fundamental data available."

    def _generate_charts_section(
        self,
        analyses: Dict[str, TickerAnalysis],
        comparison_chart_path: Optional[Path] = None,
    ) -> List[str]:
        """Generate charts section with embedded images.

        Args:
            analyses: All analysis results
            comparison_chart_path: Path to comparison chart (optional)

        Returns:
            List of section lines with embedded chart images
        """
        lines = ["## 📈 Technical Charts", ""]

        # Add comparison chart first if available (for multi-ticker reports)
        if comparison_chart_path and comparison_chart_path.exists():
            lines.append("### Portfolio Comparison")
            lines.append(
                f"![Portfolio Comparison]({comparison_chart_path.name})"
            )
            lines.append("")

        # Add individual ticker charts
        for ticker, analysis in analyses.items():
            if analysis.error:
                continue

            # Add individual ticker chart
            if analysis.chart_path and analysis.chart_path.exists():
                lines.append(f"### {ticker} Technical Analysis")
                # Use relative path from markdown file location
                chart_filename = analysis.chart_path.name
                lines.append(f"![{ticker} Technical Chart]({chart_filename})")
                lines.append("")

        return lines

    @staticmethod
    def _validate_and_fix_markdown_tables(text: str) -> str:
        """Validate and fix markdown tables to ensure consistent column counts.

        This function detects markdown tables in text and ensures that all rows
        have the same number of columns as the header row. Rows with missing
        columns are padded with "N/A", and rows with extra columns are truncated.

        Args:
            text: Text potentially containing markdown tables

        Returns:
            Text with corrected tables
        """
        import re

        lines = text.split('\n')
        result = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this line looks like a table header (has pipes and is not empty)
            if '|' in line and line.strip() and i + 1 < len(lines):
                # Check if next line is a separator (contains dashes and pipes)
                next_line = lines[i + 1]
                # More flexible separator pattern - just needs pipes and dashes/colons
                if '|' in next_line and re.search(r'[-:]', next_line):
                    # Found a table! Process it
                    header = line
                    separator = next_line

                    # Count columns in header (number of pipes minus 1 for outer pipes)
                    header_pipes = header.count('|')

                    # Add header and separator as-is
                    result.append(header)
                    result.append(separator)
                    i += 2

                    # Process table rows
                    while i < len(lines):
                        row = lines[i]

                        # Stop if we hit a blank line or non-table line
                        if not row.strip() or '|' not in row:
                            break

                        # Count pipes in this row
                        row_pipes = row.count('|')

                        if row_pipes != header_pipes:
                            # Fix the row
                            logger.warning(
                                f"Table row column mismatch: expected {header_pipes} pipes, "
                                f"found {row_pipes}. Fixing row: {row[:50]}..."
                            )

                            # Parse the row into cells
                            cells = [cell.strip() for cell in row.split('|')]
                            # Remove empty first/last elements from outer pipes
                            if cells and not cells[0]:
                                cells = cells[1:]
                            if cells and not cells[-1]:
                                cells = cells[:-1]

                            # Expected number of columns (header_pipes - 1 for outer pipes)
                            expected_cols = header_pipes - 1

                            # Pad or truncate
                            if len(cells) < expected_cols:
                                cells.extend(['N/A'] * (expected_cols - len(cells)))
                            elif len(cells) > expected_cols:
                                cells = cells[:expected_cols]

                            # Reconstruct row
                            row = '| ' + ' | '.join(cells) + ' |'

                        result.append(row)
                        i += 1

                    continue

            result.append(line)
            i += 1

        return '\n'.join(result)

    def _generate_options_section(
        self,
        analyses: Dict[str, TickerAnalysis],
    ) -> List[str]:
        """Generate options analysis section with narratives and charts.

        Args:
            analyses: All analysis results

        Returns:
            List of section lines with options analysis content
        """
        lines = []

        # Check if any ticker has options analysis
        has_options = any(
            analysis.options_analysis is not None
            for analysis in analyses.values()
            if not analysis.error
        )

        if not has_options:
            return lines

        lines.append("## 📊 Options Analysis")
        lines.append("")

        # Add individual ticker options analysis
        for ticker, analysis in analyses.items():
            if analysis.error or not analysis.options_analysis:
                continue

            options = analysis.options_analysis
            lines.append(f"### {ticker} Options Analysis")
            lines.append("")

            # Add executive summary if available
            if options.executive_summary:
                lines.append(options.executive_summary)
                lines.append("")

            # Add key metrics
            if options.iv_analysis:
                lines.append("**Implied Volatility Metrics:**")
                lines.append(f"- Current IV: {options.iv_analysis.current_iv:.2%}")
                lines.append(f"- Historical Volatility: {options.iv_analysis.historical_volatility:.2%}")
                if options.iv_analysis.iv_vs_hv_ratio:
                    lines.append(f"- IV/HV Ratio: {options.iv_analysis.iv_vs_hv_ratio:.2f}")
                lines.append("")

            # Add strategy recommendations if available
            if options.strategy_recommendations:
                lines.append("**Strategy Recommendations:**")
                lines.append(options.strategy_recommendations)
                lines.append("")

            # Add options charts
            if options.chain_heatmap_path and options.chain_heatmap_path.exists():
                lines.append("#### Options Chain Heatmap")
                lines.append(f"![{ticker} Options Heatmap]({options.chain_heatmap_path.name})")
                lines.append("")

            if options.greeks_chart_path and options.greeks_chart_path.exists():
                lines.append("#### Greeks Analysis")
                lines.append(f"![{ticker} Greeks]({options.greeks_chart_path.name})")
                lines.append("")

            if options.pnl_diagram_path and options.pnl_diagram_path.exists():
                lines.append("#### P&L Diagram (Top Strategy)")
                lines.append(f"![{ticker} P&L Diagram]({options.pnl_diagram_path.name})")
                lines.append("")

            if options.iv_surface_path and options.iv_surface_path.exists():
                lines.append("#### Implied Volatility Surface")
                lines.append(f"![{ticker} IV Surface]({options.iv_surface_path.name})")
                lines.append("")

        return lines

    def _generate_individual_analysis(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate individual stock sections."""
        sections = []

        for ticker, analysis in analyses.items():
            if analysis.error:
                continue

            sections.append(f"### {ticker}")
            sections.append(f"**Current Price:** ${analysis.latest_close:.2f}")
            sections.append(
                f"**Performance:** {analysis.avg_daily_return*100:.3f}% avg daily return"
            )

            metrics = analysis.advanced_metrics
            if metrics.sharpe_ratio:
                sections.append(
                    f"**Risk-Adjusted Returns:** Sharpe {metrics.sharpe_ratio:.2f}"
                )

            ratios = analysis.ratios
            if ratios.get("pe_ratio"):
                sections.append(f"**Valuation:** P/E {ratios['pe_ratio']:.1f}")
            if ratios.get("price_to_book"):
                sections.append(f"**P/B Ratio:** {ratios['price_to_book']:.2f}")

            if analysis.alerts:
                sections.append(f"**⚠️ Alerts:** {', '.join(analysis.alerts)}")

            sections.append("")

        return "\n".join(sections)

    def _generate_risk_analysis(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate risk analysis section."""
        lines = []

        for ticker, analysis in analyses.items():
            if analysis.error:
                continue

            metrics = analysis.advanced_metrics
            risk_level = "Low"

            if analysis.volatility > 0.03:
                risk_level = "High"
            elif analysis.volatility > 0.02:
                risk_level = "Moderate"

            lines.append(
                f"- **{ticker}:** {risk_level} risk "
                f"(Vol: {analysis.volatility*100:.1f}%, "
                f"Max DD: {(metrics.max_drawdown or 0)*100:.1f}%, "
                f"VaR 95%: {(metrics.var_95 or 0)*100:.2f}%)"
            )

        return "\n".join(lines) if lines else "No risk data available."

    def _generate_recommendations(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate investment recommendations."""
        successful = {t: a for t, a in analyses.items() if not a.error}

        if not successful:
            return "Insufficient data for recommendations."

        # Best Sharpe
        best_sharpe = sorted(
            successful.items(),
            key=lambda x: x[1].advanced_metrics.sharpe_ratio or -999,
            reverse=True,
        )[:3]

        recs = ["### Top Risk-Adjusted Performers (by Sharpe Ratio)"]
        for ticker, analysis in best_sharpe:
            sharpe = analysis.advanced_metrics.sharpe_ratio
            if sharpe:
                recs.append(f"- **{ticker}** (Sharpe: {sharpe:.2f})")

        return "\n".join(recs)

    # ========================================================================
    # OPTIONS-SPECIFIC LLM METHODS
    # ========================================================================

    def generate_options_narrative(
        self, ticker: str, options_analysis, spot_price: float
    ) -> str:
        """
        Generate executive summary for options opportunities.

        Args:
            ticker: Ticker symbol
            options_analysis: TickerOptionsAnalysis object
            spot_price: Current underlying price

        Returns:
            Narrative text summarizing options opportunities
        """
        try:
            # Prepare options data summary
            options_summary = {
                "ticker": ticker,
                "spot_price": spot_price,
                "chains_count": len(options_analysis.chains) if options_analysis.chains else 0,
                "iv_analysis": {
                    "current_iv": options_analysis.iv_analysis.current_iv if options_analysis.iv_analysis else None,
                    "historical_vol": options_analysis.iv_analysis.historical_volatility if options_analysis.iv_analysis else None,
                    "iv_vs_hv": options_analysis.iv_analysis.iv_vs_hv_ratio if options_analysis.iv_analysis else None,
                },
                "top_strategies": [
                    {
                        "type": str(s.strategy_type.value),
                        "description": s.description,
                        "net_premium": s.net_premium,
                        "max_profit": s.max_profit,
                        "max_loss": s.max_loss,
                        "prob_profit": s.probability_of_profit,
                    }
                    for s in options_analysis.top_strategies[:3]
                ] if options_analysis.top_strategies else [],
            }

            response = self.options_narrative_chain.invoke({
                "options_data": json.dumps(options_summary, indent=2),
                "ticker": ticker,
                "spot_price": spot_price,
            })

            narrative = response.content if hasattr(response, "content") else str(response)
            # Validate and fix any markdown tables
            narrative = self._validate_and_fix_markdown_tables(narrative)
            logger.info(f"Generated options narrative for {ticker}")
            return narrative

        except Exception as e:
            logger.error(f"Failed to generate options narrative: {e}")
            return f"## Options Analysis for {ticker}\n\nOptions data available but narrative generation failed."

    def generate_options_recommendations(
        self, ticker: str, strategies: List, market_context: Dict, portfolio_holdings: str = "None"
    ) -> str:
        """
        Generate personalized options strategy recommendations.

        Args:
            ticker: Ticker symbol
            strategies: List of OptionsStrategy objects
            market_context: Dict with IV, momentum, etc.
            portfolio_holdings: Description of current holdings

        Returns:
            Recommendations text
        """
        try:
            strategies_summary = [
                {
                    "name": str(s.strategy_type.value),
                    "description": s.description,
                    "cost": s.net_premium,
                    "max_profit": s.max_profit,
                    "max_loss": s.max_loss,
                    "breakevens": s.breakeven_points,
                    "prob_profit": s.probability_of_profit,
                }
                for s in strategies[:10]  # Limit to top 10
            ]

            response = self.options_recommendations_chain.invoke({
                "ticker": ticker,
                "strategies": json.dumps(strategies_summary, indent=2),
                "market_context": json.dumps(market_context, indent=2),
                "portfolio_holdings": portfolio_holdings,
            })

            recommendations = response.content if hasattr(response, "content") else str(response)
            # Validate and fix any markdown tables
            recommendations = self._validate_and_fix_markdown_tables(recommendations)
            logger.info(f"Generated options recommendations for {ticker}")
            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate options recommendations: {e}")
            return "## Options Recommendations\n\nRecommendations generation failed."

    def generate_portfolio_hedging_analysis(
        self, portfolio_greeks: Dict, positions: Dict, risk_tolerance: str = "moderate"
    ) -> str:
        """
        Generate portfolio-level hedging recommendations.

        Args:
            portfolio_greeks: Dict with total_delta, total_vega, etc.
            positions: Dict of current positions
            risk_tolerance: "conservative", "moderate", or "aggressive"

        Returns:
            Hedging analysis text
        """
        try:
            response = self.portfolio_hedging_chain.invoke({
                "portfolio_greeks": json.dumps(portfolio_greeks, indent=2),
                "positions": json.dumps(positions, indent=2),
                "risk_tolerance": risk_tolerance,
            })

            analysis = response.content if hasattr(response, "content") else str(response)
            logger.info("Generated portfolio hedging analysis")
            return analysis

        except Exception as e:
            logger.error(f"Failed to generate hedging analysis: {e}")
            return "## Portfolio Hedging Analysis\n\nHedging analysis generation failed."

    def __del__(self) -> None:
        """Cleanup: Close HTTP client when object is destroyed."""
        if hasattr(self, "llm") and hasattr(self.llm, "client"):
            try:
                # Close the httpx client to clean up connections
                if hasattr(self.llm.client, "close"):
                    self.llm.client.close()
                logger.debug("Closed LLM HTTP client connection pool")
            except Exception:
                pass  # Ignore errors during cleanup
