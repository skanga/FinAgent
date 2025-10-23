"""
Alert system for significant market events.
"""

import logging
from typing import List
from models import TickerAnalysis
from constants import AnalysisThresholds

logger = logging.getLogger(__name__)


class AlertSystem:
    """Monitor for significant market movements and conditions."""

    def __init__(
        self,
        volatility_threshold: float = AnalysisThresholds.HIGH_VOLATILITY,
        drawdown_threshold: float = AnalysisThresholds.LARGE_DRAWDOWN,
    ) -> None:
        """
        Initializes the AlertSystem with given thresholds.

        Args:
            volatility_threshold (float): The threshold for volatility alerts.
            drawdown_threshold (float): The threshold for drawdown alerts.
        """
        self.volatility_threshold = volatility_threshold
        self.drawdown_threshold = drawdown_threshold

    def check_alerts(self, analysis: TickerAnalysis) -> List[str]:
        """
        Checks for alert conditions based on the provided ticker analysis.

        Args:
            analysis (TickerAnalysis): The ticker analysis to check for alerts.

        Returns:
            List[str]: A list of alerts.
        """
        alerts = []

        # Volatility alerts
        if analysis.volatility > self.volatility_threshold:
            alerts.append(f"High volatility: {analysis.volatility*100:.1f}%")

        # Drawdown alerts
        if (
            analysis.advanced_metrics.max_drawdown
            and analysis.advanced_metrics.max_drawdown < self.drawdown_threshold
        ):
            alerts.append(
                f"Large drawdown: {analysis.advanced_metrics.max_drawdown*100:.1f}%"
            )

        # VaR alerts
        if (
            analysis.advanced_metrics.var_95
            and analysis.advanced_metrics.var_95 < AnalysisThresholds.VALUE_AT_RISK_95_THRESHOLD
        ):
            alerts.append(
                f"High VaR (95%): {analysis.advanced_metrics.var_95*100:.2f}%"
            )

        # RSI alerts
        if analysis.technical_indicators.rsi:
            if analysis.technical_indicators.rsi > AnalysisThresholds.OVERBOUGHT_RSI:
                alerts.append(
                    f"Overbought (RSI: {analysis.technical_indicators.rsi:.0f})"
                )
            elif analysis.technical_indicators.rsi < AnalysisThresholds.OVERSOLD_RSI:
                alerts.append(
                    f"Oversold (RSI: {analysis.technical_indicators.rsi:.0f})"
                )

        # Comparative alerts
        if (
            analysis.comparative_analysis
            and analysis.comparative_analysis.outperformance
            and analysis.comparative_analysis.outperformance
            < AnalysisThresholds.UNDERPERFORMANCE
        ):
            alerts.append(
                f"Underperforming benchmark: {analysis.comparative_analysis.outperformance*100:.2f}%"
            )

        return alerts
