"""
LLM integration for narrative generation and parsing.
"""
import json
import logging
import pandas as pd
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
    before_sleep_log
)

logger = logging.getLogger(__name__)


class IntegratedLLMInterface:
    """LLM interface with connection pooling for better performance."""

    def __init__(self, config: Config,
                 max_connections: int = 20,
                 max_keepalive_connections: int = 10) -> None:
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
            max_connections,
            max_keepalive_connections
        )

        # Initialize LLM with connection pooling
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            model=config.model_name,
            temperature=0.3,
            timeout=config.request_timeout,
            http_client=http_client,  # Use our pooled client
            http_async_client=None  # We're not using async
        )

        # Log LLM setup (without exposing API key)
        provider = config._get_provider_from_url()
        logger.info(
            f"LLM initialized - Provider: {provider}, Model: {config.model_name}, "
            f"Temperature: 0.3, Connection pool: {max_keepalive_connections}/{max_connections}"
        )

        self._setup_prompts()
        self._setup_chains()

    def _create_pooled_http_client(self, max_connections: int = 20,
                                   max_keepalive: int = 10):
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
            keepalive_expiry=30.0  # Keep connections alive for 30 seconds
        )

        # Configure timeout
        timeout = Timeout(
            connect=10.0,  # Connection timeout
            read=self.config.request_timeout,  # Read timeout
            write=10.0,  # Write timeout
            pool=5.0  # Pool timeout
        )

        # Create client with pooling
        client = httpx.Client(
            limits=limits,
            timeout=timeout,
            http2=True,  # Enable HTTP/2 for multiplexing
            follow_redirects=True
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

Return ONLY the JSON, no other text."""
        )
        
        self.narrative_prompt = PromptTemplate(
            input_variables=["analysis_data", "period"],
            template="""You are a senior financial analyst. Generate a compelling executive summary.

Analysis Period: {period}
Data: {analysis_data}

IMPORTANT GUIDELINES:

1. **Metric Definitions (use these correctly):**
   - Returns are DAILY averages (to annualize: ~252 trading days)
   - Volatility is DAILY std dev (to annualize: multiply by √252)
   - Sharpe ratio is based on daily returns
   - VaR and max drawdown are percentages (VaR is shown as negative)
   - RSI: 0-100 scale (>70 = overbought, <30 = oversold, 30-70 = neutral)
   - P/E and P/B are TTM (trailing twelve months)

2. **Technical Interpretation Rules:**
   - RSI 50-70: Neutral to moderately bullish (NOT overbought)
   - RSI >70: Overbought territory
   - RSI <30: Oversold territory
   - Sharpe >1: Good risk-adjusted returns
   - Max DD >-20%: Moderate risk, <-20%: High risk

3. **When citing numbers:**
   - State "daily" or "annualized" explicitly
   - Match EXACT values from the data - no rounding changes
   - If converting (e.g., daily to annual), show the calculation
   - For unusual values, cite accurately and note they're unusual

4. **Write 3-4 paragraphs that:**
   - Highlight significant findings with precise numbers
   - Use CORRECT interpretations (e.g., don't call RSI 60 "overbought")
   - Compare performance across tickers
   - Identify risks with supporting data

5. **Avoid these common errors:**
   - ❌ Calling RSI 50-70 "overbought" (it's neutral/bullish)
   - ❌ Using different values for same metric in different places
   - ❌ Making claims without data support (e.g., "above market average" without providing market average)
   - ❌ Confusing daily and annualized figures

Be precise, accurate, and professional. Use correct technical definitions.
Return ONLY the narrative text."""
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

Return ONLY the JSON."""
        )
    
    def _setup_chains(self) -> None:
        """Setup LangChain chains."""
        self.parser_chain = self.parse_prompt | self.llm
        self.narrative_chain = self.narrative_prompt | self.llm
        self.review_chain = self.review_prompt | self.llm
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
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
                output_format=parsed_json.get("output_format", "markdown")
            )

            parsed.validate()
            logger.info(f"Parsed request: {parsed.tickers}")
            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            raise ValueError(f"Failed to parse JSON response") from e
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Parse error: {e}")
            raise ValueError(f"Parse error: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def generate_narrative_summary(self, analyses: Dict[str, TickerAnalysis],
                                   period: str) -> str:
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
                    "sharpe_ratio_daily": round(analysis.advanced_metrics.sharpe_ratio, 2) if analysis.advanced_metrics.sharpe_ratio else None,
                    "max_drawdown_pct": f"{(analysis.advanced_metrics.max_drawdown or 0) * 100:.1f}%",
                    "rsi": round(analysis.technical_indicators.rsi, 1) if analysis.technical_indicators.rsi else None,
                    "pe_ratio": round(analysis.ratios.get('pe_ratio'), 1) if analysis.ratios.get('pe_ratio') else None,
                    "pb_ratio": round(analysis.ratios.get('price_to_book'), 2) if analysis.ratios.get('price_to_book') else None,
                    "alerts": analysis.alerts
                }
        
        try:
            response = self.narrative_chain.invoke({
                "analysis_data": json.dumps(analysis_summary, indent=2),
                "period": period
            })

            narrative = response.content.strip()
            logger.info("Generated narrative summary")
            return narrative

        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error(f"Failed to generate narrative: {e}")
            return self._fallback_narrative(analyses, period)
    
    def _fallback_narrative(self, analyses: Dict[str, TickerAnalysis], period: str) -> str:
        """Fallback narrative if LLM fails."""
        successful = {t: a for t, a in analyses.items() if not a.error}
        if not successful:
            return "Analysis completed but no valid data available."
        
        best = max(successful.items(), key=lambda x: x[1].avg_daily_return)
        worst = min(successful.items(), key=lambda x: x[1].avg_daily_return)
        
        return f"""Over the {period} period, {len(successful)} stocks were analyzed.
{best[0]} showed the strongest performance with a {best[1].avg_daily_return*100:.2f}% average daily return,
while {worst[0]} had the weakest returns at {worst[1].avg_daily_return*100:.2f}%.
Risk metrics varied across the portfolio, with several stocks showing elevated volatility warranting monitoring."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def review_report(self, report_content: str,
                     analyses: Dict[str, TickerAnalysis]) -> Tuple[List[str], int, Dict]:
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
                    "rsi": analysis.technical_indicators.rsi,
                    "pe_ratio": analysis.ratios.get('pe_ratio'),
                    "pb_ratio": analysis.ratios.get('price_to_book')
                }

        try:
            response = self.review_chain.invoke({
                "report_content": report_content[:4000],  # Limit size
                "data_summary": json.dumps(data_summary, indent=2)
            })

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

            logger.info(f"Review: {len(issues)} issues, {len(suggestions)} suggestions, quality: {quality_score}/10")

            # Return full review data
            full_review = {
                "issues": issues,
                "suggestions": suggestions,
                "quality_score": quality_score
            }
            return issues, quality_score, full_review

        except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning(f"Review failed: {e}")
            return [], 7, {"issues": [], "suggestions": [], "quality_score": 7, "error": str(e)}
    
    def _generate_report_header(self) -> List[str]:
        """Generate report header with timestamp.

        Returns:
            List of header lines
        """
        from datetime import datetime, timezone
        return [
            "# 📊 Advanced Financial Analysis Report",
            f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            ""
        ]

    def _generate_executive_summary_section(self, analyses: Dict[str, TickerAnalysis], period: str) -> List[str]:
        """Generate executive summary section with LLM narrative.

        Args:
            analyses: All analysis results
            period: Analysis period

        Returns:
            List of section lines
        """
        narrative = self.generate_narrative_summary(analyses, period)
        return [
            "## 🎯 Executive Summary",
            narrative,
            ""
        ]

    def _generate_portfolio_overview_section(self, portfolio_metrics: Optional[PortfolioMetrics]) -> List[str]:
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
            ""
        ]

    def _generate_alerts_section(self, analyses: Dict[str, TickerAnalysis]) -> List[str]:
        """Generate active alerts section.

        Args:
            analyses: All analysis results

        Returns:
            List of section lines (empty if no alerts)
        """
        all_alerts = [a for analysis in analyses.values() if not analysis.error for a in analysis.alerts]
        if not all_alerts:
            return []

        section = ["## 🚨 Active Alerts"]
        for alert in all_alerts:
            section.append(f"- {alert}")
        section.append("")
        return section

    def _generate_failures_section(self, analyses: Dict[str, TickerAnalysis]) -> List[str]:
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

    def generate_detailed_report(self, analyses: Dict[str, TickerAnalysis],
                                benchmark_analysis: Optional[TickerAnalysis],
                                portfolio_metrics: Optional[PortfolioMetrics],
                                period: str) -> str:
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
        report.append("## 📈 Key Performance Metrics")
        report.append(self._generate_metrics_table(analyses))
        report.append("")

        # Methodology note
        report.append("### Methodology Notes")
        report.append("- **Return (Daily):** Average daily return calculated from historical price data")
        report.append("- **Vol (Daily):** Standard deviation of daily returns (annualize: multiply by √252)")
        report.append("- **Sharpe Ratio:** Risk-adjusted return calculated as (mean daily return - risk-free rate) / daily volatility")
        report.append("- **Max DD:** Maximum peak-to-trough decline during the analysis period")
        report.append("- **VaR (95%):** Value at Risk at 95% confidence level (negative value indicates potential loss)")
        report.append("- **RSI:** Relative Strength Index (0-100 scale; >70 overbought, <30 oversold)")
        report.append("- **P/E (TTM):** Price-to-Earnings ratio based on trailing twelve months")
        report.append("- **Rev Growth (YoY):** Year-over-year revenue growth rate")
        report.append("")

        # Fundamental Analysis
        report.append("## 📊 Fundamental Analysis")
        report.append(self._generate_fundamental_section(analyses))
        report.append("")

        # Individual Analysis
        report.append("## 📋 Detailed Stock Analysis")
        report.append(self._generate_individual_analysis(analyses))
        report.append("")

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
        lines.append(f"**Portfolio Return:** {portfolio_metrics.portfolio_return*100:.2f}%")
        lines.append(f"**Portfolio Volatility:** {portfolio_metrics.portfolio_volatility*100:.2f}%")
        
        if portfolio_metrics.portfolio_sharpe:
            lines.append(f"**Portfolio Sharpe Ratio:** {portfolio_metrics.portfolio_sharpe:.2f}")
        
        if portfolio_metrics.diversification_ratio:
            lines.append(f"**Diversification Ratio:** {portfolio_metrics.diversification_ratio:.2f}")
        
        if portfolio_metrics.concentration_risk:
            lines.append(f"**Concentration Risk (HHI):** {portfolio_metrics.concentration_risk:.3f}")
        
        if portfolio_metrics.top_contributors:
            lines.append("\n**Top Contributors:**")
            for ticker, contribution in portfolio_metrics.top_contributors[:3]:
                lines.append(f"- {ticker}: {contribution*100:.2f}% contribution to return")
        
        return "\n".join(lines)
    
    def _generate_metrics_table(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate comprehensive metrics table."""
        table = [
            "| Ticker | Price | Return (Daily) | Vol (Daily) | Sharpe | Max DD | RSI | P/E (TTM) | Rev Growth (YoY) |",
            "|--------|-------|----------------|-------------|--------|--------|-----|-----------|------------------|"
        ]
        
        for ticker, analysis in analyses.items():
            if analysis.error:
                continue
            
            m = analysis.advanced_metrics
            t = analysis.technical_indicators
            r = analysis.ratios
            f = analysis.fundamentals

            # Safe conversions with NaN checks
            sharpe_str = f"{m.sharpe_ratio:.2f}" if m.sharpe_ratio is not None and not pd.isna(m.sharpe_ratio) else "N/A"
            dd_val = m.max_drawdown if m.max_drawdown is not None and not pd.isna(m.max_drawdown) else 0
            dd_str = f"{dd_val*100:.1f}%"
            rsi_str = f"{t.rsi:.0f}" if t.rsi is not None and not pd.isna(t.rsi) else "N/A"
            pe_str = f"{r.get('pe_ratio'):.1f}" if r.get('pe_ratio') and not pd.isna(r.get('pe_ratio')) else "N/A"
            # Check for valid revenue_growth (not None, not NaN)
            has_growth = f and f.revenue_growth is not None and not pd.isna(f.revenue_growth)
            growth_str = f"{f.revenue_growth*100:.1f}%" if has_growth else "N/A"
            
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
            
            f = analysis.fundamentals
            if not any([f.revenue, f.net_income, f.revenue_growth]):
                continue
            
            lines.append(f"### {ticker}")
            
            if f.revenue:
                lines.append(f"- **Revenue:** ${f.revenue/1e9:.2f}B")
            if f.net_income:
                lines.append(f"- **Net Income:** ${f.net_income/1e9:.2f}B")
            if f.revenue_growth:
                lines.append(f"- **Revenue Growth (YoY):** {f.revenue_growth*100:.1f}%")
            if f.earnings_growth:
                lines.append(f"- **Earnings Growth (YoY):** {f.earnings_growth*100:.1f}%")
            if f.free_cash_flow:
                lines.append(f"- **Free Cash Flow:** ${f.free_cash_flow/1e9:.2f}B")
            
            lines.append("")
        
        return "\n".join(lines) if lines else "Limited fundamental data available."
    
    def _generate_individual_analysis(self, analyses: Dict[str, TickerAnalysis]) -> str:
        """Generate individual stock sections."""
        sections = []
        
        for ticker, analysis in analyses.items():
            if analysis.error:
                continue
            
            sections.append(f"### {ticker}")
            sections.append(f"**Current Price:** ${analysis.latest_close:.2f}")
            sections.append(f"**Performance:** {analysis.avg_daily_return*100:.3f}% avg daily return")
            
            m = analysis.advanced_metrics
            if m.sharpe_ratio:
                sections.append(f"**Risk-Adjusted Returns:** Sharpe {m.sharpe_ratio:.2f}")
            
            r = analysis.ratios
            if r.get('pe_ratio'):
                sections.append(f"**Valuation:** P/E {r['pe_ratio']:.1f}")
            if r.get('price_to_book'):
                sections.append(f"**P/B Ratio:** {r['price_to_book']:.2f}")
            
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
            
            m = analysis.advanced_metrics
            risk_level = "Low"
            
            if analysis.volatility > 0.03:
                risk_level = "High"
            elif analysis.volatility > 0.02:
                risk_level = "Moderate"
            
            lines.append(f"- **{ticker}:** {risk_level} risk "
                        f"(Vol: {analysis.volatility*100:.1f}%, "
                        f"Max DD: {(m.max_drawdown or 0)*100:.1f}%, "
                        f"VaR 95%: {(m.var_95 or 0)*100:.2f}%)")
        
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
            reverse=True
        )[:3]

        recs = ["### Top Risk-Adjusted Performers (by Sharpe Ratio)"]
        for ticker, analysis in best_sharpe:
            sharpe = analysis.advanced_metrics.sharpe_ratio
            if sharpe:
                recs.append(f"- **{ticker}** (Sharpe: {sharpe:.2f})")

        return "\n".join(recs)

    def __del__(self) -> None:
        """Cleanup: Close HTTP client when object is destroyed."""
        if hasattr(self, 'llm') and hasattr(self.llm, 'client'):
            try:
                # Close the httpx client to clean up connections
                if hasattr(self.llm.client, 'close'):
                    self.llm.client.close()
                logger.debug("Closed LLM HTTP client connection pool")
            except Exception:
                pass  # Ignore errors during cleanup