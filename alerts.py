"""
Alert system for significant market events.
"""
import logging
from typing import List
from models import TickerAnalysis
from constants import (
    VOLATILITY_ALERT_THRESHOLD,
    DRAWDOWN_ALERT_THRESHOLD,
    HIGH_VAR_THRESHOLD,
    RSI_OVERBOUGHT_THRESHOLD,
    RSI_OVERSOLD_THRESHOLD,
    UNDERPERFORMANCE_THRESHOLD
)

logger = logging.getLogger(__name__)


class AlertSystem:
    """Monitor for significant market movements and conditions."""

    def __init__(self,
                 volatility_threshold: float = VOLATILITY_ALERT_THRESHOLD,
                 drawdown_threshold: float = DRAWDOWN_ALERT_THRESHOLD) -> None:
        self.volatility_threshold = volatility_threshold
        self.drawdown_threshold = drawdown_threshold
    
    def check_alerts(self, analysis: TickerAnalysis) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # Volatility alerts
        if analysis.volatility > self.volatility_threshold:
            alerts.append(f"High volatility: {analysis.volatility*100:.1f}%")
        
        # Drawdown alerts
        if (analysis.advanced_metrics.max_drawdown and 
            analysis.advanced_metrics.max_drawdown < self.drawdown_threshold):
            alerts.append(f"Large drawdown: {analysis.advanced_metrics.max_drawdown*100:.1f}%")
        
        # VaR alerts
        if analysis.advanced_metrics.var_95 and analysis.advanced_metrics.var_95 < HIGH_VAR_THRESHOLD:
            alerts.append(f"High VaR (95%): {analysis.advanced_metrics.var_95*100:.2f}%")

        # RSI alerts
        if analysis.technical_indicators.rsi:
            if analysis.technical_indicators.rsi > RSI_OVERBOUGHT_THRESHOLD:
                alerts.append(f"Overbought (RSI: {analysis.technical_indicators.rsi:.0f})")
            elif analysis.technical_indicators.rsi < RSI_OVERSOLD_THRESHOLD:
                alerts.append(f"Oversold (RSI: {analysis.technical_indicators.rsi:.0f})")

        # Comparative alerts
        if (analysis.comparative_analysis and
            analysis.comparative_analysis.outperformance and
            analysis.comparative_analysis.outperformance < UNDERPERFORMANCE_THRESHOLD):
            alerts.append(
                f"Underperforming benchmark: {analysis.comparative_analysis.outperformance*100:.2f}%"
            )
        
        return alerts
		