"""
Comprehensive tests for alert system.

Tests cover:
- Alert system initialization
- Volatility alerts
- Drawdown alerts
- VaR alerts
- RSI alerts (overbought/oversold)
- Stochastic alerts (overbought/oversold)
- Comparative analysis alerts (underperformance)
- Null safety for all metrics
- Missing nested objects
- Edge cases and combinations
- Alert formatting
"""

import pytest
import dataclasses
from pathlib import Path
from alerts import AlertSystem
from models import (
    TickerAnalysis,
    AdvancedMetrics,
    TechnicalIndicators,
    ComparativeAnalysis,
    FundamentalData,
)
from constants import AnalysisThresholds


@pytest.fixture
def alert_system():
    """Create alert system with default thresholds."""
    return AlertSystem()


@pytest.fixture
def custom_alert_system():
    """Create alert system with custom thresholds."""
    return AlertSystem(
        volatility_threshold=0.10,  # 10% volatility
        drawdown_threshold=-0.20,  # -20% drawdown
    )


@pytest.fixture
def base_analysis():
    """Create base ticker analysis with moderate values (no alerts)."""
    return TickerAnalysis(
        ticker="AAPL",
        csv_path=Path("/tmp/AAPL.csv"),
        chart_path=Path("/tmp/AAPL.png"),
        latest_close=150.0,
        avg_daily_return=0.0005,  # 0.05% daily return
        volatility=0.03,  # 3% volatility (below 5% threshold)
        ratios={},
        fundamentals=FundamentalData(),
        technical_indicators=TechnicalIndicators(
            rsi=50.0,  # Neutral
            macd=1.0,
            macd_signal=0.9,
            bollinger_upper=155.0,
            bollinger_lower=145.0,
            bollinger_position=0.5,
            stochastic_k=50.0,  # Neutral
            stochastic_d=48.0,
            atr=3.0,
            obv=1000000.0,
            ma_200d=148.0,
        ),
        advanced_metrics=AdvancedMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.2,
            max_drawdown=-0.05,  # -5% drawdown (below -10% threshold)
            beta=1.0,
            alpha=0.02,  # 2% annualized
            var_95=-0.02,  # -2% VaR (above -5% threshold)
            cvar_95=-0.025,
            treynor_ratio=0.10,
            information_ratio=0.5,
        ),
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestAlertSystemInitialization:
    """Test alert system initialization."""

    def test_default_initialization(self):
        """Test initialization with default thresholds."""
        system = AlertSystem()

        assert system.volatility_threshold == AnalysisThresholds.HIGH_VOLATILITY
        assert system.drawdown_threshold == AnalysisThresholds.LARGE_DRAWDOWN

    def test_custom_initialization(self):
        """Test initialization with custom thresholds."""
        system = AlertSystem(volatility_threshold=0.08, drawdown_threshold=-0.15)

        assert system.volatility_threshold == 0.08
        assert system.drawdown_threshold == -0.15


# ============================================================================
# VOLATILITY ALERTS
# ============================================================================

class TestVolatilityAlerts:
    """Test volatility alert generation."""

    def test_high_volatility_alert(self, alert_system, base_analysis):
        """Test alert for high volatility."""
        analysis = dataclasses.replace(base_analysis, volatility=0.07)  # 7% > 5% threshold

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("High volatility" in alert for alert in alerts)

    def test_low_volatility_no_alert(self, alert_system, base_analysis):
        """Test no alert for low volatility."""
        analysis = dataclasses.replace(base_analysis, volatility=0.03)  # 3% < 5% threshold

        alerts = alert_system.check_alerts(analysis)

        assert not any("volatility" in alert.lower() for alert in alerts)

    def test_volatility_at_threshold(self, alert_system, base_analysis):
        """Test volatility exactly at threshold (should not trigger)."""
        analysis = dataclasses.replace(base_analysis, volatility=0.05)  # Exactly 5%

        alerts = alert_system.check_alerts(analysis)

        assert not any("volatility" in alert.lower() for alert in alerts)

    def test_custom_volatility_threshold(self, custom_alert_system, base_analysis):
        """Test with custom volatility threshold."""
        analysis = dataclasses.replace(base_analysis, volatility=0.07)

        alerts = custom_alert_system.check_alerts(analysis)

        # 7% < custom 10% threshold, so no alert
        assert not any("volatility" in alert.lower() for alert in alerts)

    def test_volatility_none_no_error(self, alert_system, base_analysis):
        """Test that None volatility doesn't raise error."""
        analysis = dataclasses.replace(base_analysis, volatility=None)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)
        assert len(alerts) == 0  # No alerts for None volatility

    def test_volatility_zero_no_error(self, alert_system, base_analysis):
        """Test that zero volatility doesn't raise error."""
        analysis = dataclasses.replace(base_analysis, volatility=0.0)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# DRAWDOWN ALERTS
# ============================================================================

class TestDrawdownAlerts:
    """Test drawdown alert generation."""

    def test_large_drawdown_alert(self, alert_system, base_analysis):
        """Test alert for large drawdown."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.15  # -15% < -10% threshold
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("Large drawdown" in alert or "drawdown" in alert.lower() for alert in alerts)

    def test_small_drawdown_no_alert(self, alert_system, base_analysis):
        """Test no alert for small drawdown."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.05  # -5% > -10% threshold
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert not any("drawdown" in alert.lower() for alert in alerts)

    def test_drawdown_none_no_alert(self, alert_system, base_analysis):
        """Test None drawdown doesn't trigger alert."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=None
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_custom_drawdown_threshold(self, custom_alert_system, base_analysis):
        """Test with custom drawdown threshold."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.15
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = custom_alert_system.check_alerts(analysis)

        # -15% > custom -20% threshold, so no alert
        assert not any("drawdown" in alert.lower() for alert in alerts)

    def test_advanced_metrics_none(self, alert_system, base_analysis):
        """Test that None advanced_metrics doesn't raise error."""
        analysis = dataclasses.replace(base_analysis, advanced_metrics=None)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)
        assert len(alerts) == 0  # No alerts when metrics are None


# ============================================================================
# VAR ALERTS
# ============================================================================

class TestVaRAlerts:
    """Test VaR alert generation."""

    def test_high_var_alert(self, alert_system, base_analysis):
        """Test alert for high VaR."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            var_95=-0.06  # -6% < -5% threshold
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("VaR" in alert for alert in alerts)

    def test_moderate_var_no_alert(self, alert_system, base_analysis):
        """Test no alert for moderate VaR."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            var_95=-0.03  # -3% > -5% threshold
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert not any("VaR" in alert for alert in alerts)

    def test_var_none_no_alert(self, alert_system, base_analysis):
        """Test None VaR doesn't trigger alert."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            var_95=None
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# RSI ALERTS
# ============================================================================

class TestRSIAlerts:
    """Test RSI alert generation."""

    def test_overbought_rsi_alert(self, alert_system, base_analysis):
        """Test alert for overbought RSI."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=75.0  # > 70 threshold
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("Overbought" in alert and "RSI" in alert for alert in alerts)

    def test_oversold_rsi_alert(self, alert_system, base_analysis):
        """Test alert for oversold RSI."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=25.0  # < 30 threshold
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("Oversold" in alert and "RSI" in alert for alert in alerts)

    def test_neutral_rsi_no_alert(self, alert_system, base_analysis):
        """Test no alert for neutral RSI."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=50.0  # Between 30 and 70
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert not any("RSI" in alert for alert in alerts)

    def test_rsi_at_overbought_threshold(self, alert_system, base_analysis):
        """Test RSI exactly at overbought threshold."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=70.0  # Exactly at threshold
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        # At threshold should not trigger (> not >=)
        assert not any("Overbought" in alert and "RSI" in alert for alert in alerts)

    def test_rsi_none_no_alert(self, alert_system, base_analysis):
        """Test None RSI doesn't trigger alert."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=None
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_technical_indicators_none(self, alert_system, base_analysis):
        """Test that None technical_indicators doesn't raise error."""
        analysis = dataclasses.replace(base_analysis, technical_indicators=None)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)
        assert len(alerts) == 0  # No alerts when indicators are None


# ============================================================================
# STOCHASTIC ALERTS
# ============================================================================

class TestStochasticAlerts:
    """Test Stochastic Oscillator alert generation."""

    def test_overbought_stochastic_alert(self, alert_system, base_analysis):
        """Test alert for overbought Stochastic."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            stochastic_k=85.0  # > 80 threshold
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("Overbought" in alert and "Stochastic" in alert for alert in alerts)

    def test_oversold_stochastic_alert(self, alert_system, base_analysis):
        """Test alert for oversold Stochastic."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            stochastic_k=15.0  # < 20 threshold
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("Oversold" in alert and "Stochastic" in alert for alert in alerts)

    def test_neutral_stochastic_no_alert(self, alert_system, base_analysis):
        """Test no alert for neutral Stochastic."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            stochastic_k=50.0  # Between 20 and 80
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert not any("Stochastic" in alert for alert in alerts)

    def test_stochastic_none_no_alert(self, alert_system, base_analysis):
        """Test None stochastic_k doesn't trigger alert."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            stochastic_k=None
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# COMPARATIVE ANALYSIS ALERTS
# ============================================================================

class TestComparativeAlerts:
    """Test comparative analysis alert generation."""

    def test_underperformance_alert(self, alert_system, base_analysis):
        """Test alert for underperformance vs benchmark."""
        comparative_analysis = ComparativeAnalysis(
            outperformance=-0.06  # -6% < -5% threshold
        )
        analysis = dataclasses.replace(base_analysis, comparative_analysis=comparative_analysis)

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) > 0
        assert any("Underperforming" in alert or "benchmark" in alert.lower() for alert in alerts)

    def test_outperformance_no_alert(self, alert_system, base_analysis):
        """Test no alert for outperformance."""
        comparative_analysis = ComparativeAnalysis(
            outperformance=0.03  # 3% > 0
        )
        analysis = dataclasses.replace(base_analysis, comparative_analysis=comparative_analysis)

        alerts = alert_system.check_alerts(analysis)

        assert not any("Underperforming" in alert for alert in alerts)

    def test_comparative_analysis_none_no_alert(self, alert_system, base_analysis):
        """Test None comparative_analysis doesn't trigger alert."""
        analysis = dataclasses.replace(base_analysis, comparative_analysis=None)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)
        assert len(alerts) == 0  # No alerts when analysis is None

    def test_outperformance_none_no_alert(self, alert_system, base_analysis):
        """Test None outperformance doesn't trigger alert."""
        comparative_analysis = ComparativeAnalysis(
            outperformance=None
        )
        analysis = dataclasses.replace(base_analysis, comparative_analysis=comparative_analysis)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# MULTIPLE ALERTS
# ============================================================================

class TestMultipleAlerts:
    """Test multiple alerts being generated simultaneously."""

    def test_multiple_alerts_generated(self, alert_system, base_analysis):
        """Test multiple alerts can be generated at once."""
        # Set up conditions for multiple alerts
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=75.0,  # Overbought
            stochastic_k=85.0  # Overbought
        )
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.15  # Large
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=0.10,  # High
            technical_indicators=technical_indicators,
            advanced_metrics=advanced_metrics
        )

        alerts = alert_system.check_alerts(analysis)

        # Should have at least 3 alerts
        assert len(alerts) >= 3
        assert any("volatility" in alert.lower() for alert in alerts)
        assert any("drawdown" in alert.lower() for alert in alerts)
        assert any("RSI" in alert for alert in alerts)

    def test_no_alerts_generated(self, alert_system, base_analysis):
        """Test no alerts when all metrics are within normal range."""
        # base_analysis has all normal values
        alerts = alert_system.check_alerts(base_analysis)

        assert len(alerts) == 0


# ============================================================================
# NULL SAFETY EDGE CASES
# ============================================================================

class TestNullSafetyCombinations:
    """Test combinations of null values."""

    def test_all_none_no_error(self, alert_system, base_analysis):
        """Test that all None values don't raise error."""
        analysis = dataclasses.replace(
            base_analysis,
            volatility=None,
            advanced_metrics=None,
            technical_indicators=None,
            comparative_analysis=None
        )

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_partial_none_values(self, alert_system, base_analysis):
        """Test partial None values don't raise error."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=None,
            var_95=None
        )
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=None
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=0.10,  # High (threshold is 0.05)
            advanced_metrics=advanced_metrics,
            technical_indicators=technical_indicators,
            comparative_analysis=None
        )

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)
        # Should have volatility alert only
        assert len(alerts) == 1
        assert "High volatility" in alerts[0]

    def test_all_technical_indicators_none(self, alert_system, base_analysis):
        """Test that all None technical indicators don't raise error."""
        technical_indicators = TechnicalIndicators(
            rsi=None,
            macd=None,
            macd_signal=None,
            stochastic_k=None,
            stochastic_d=None,
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_all_advanced_metrics_none(self, alert_system, base_analysis):
        """Test that all None advanced metrics don't raise error."""
        advanced_metrics = AdvancedMetrics(
            sharpe_ratio=None,
            max_drawdown=None,
            beta=None,
            alpha=None,
            var_95=None,
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_all_comparative_analysis_none(self, alert_system, base_analysis):
        """Test that all None comparative analysis fields don't raise error."""
        comparative_analysis = ComparativeAnalysis(
            outperformance=None,
            correlation=None,
            tracking_error=None,
        )
        analysis = dataclasses.replace(base_analysis, comparative_analysis=comparative_analysis)

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# ALERTS WITH NULL VALUES
# ============================================================================

class TestValidAlertsWithNulls:
    """Test that valid alerts are generated even with some null values."""

    def test_high_volatility_alert_with_nulls(self, alert_system, base_analysis):
        """Test high volatility alert with other metrics null."""
        analysis = dataclasses.replace(
            base_analysis,
            volatility=0.10,  # High
            advanced_metrics=None,
            technical_indicators=None,
            comparative_analysis=None
        )

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) == 1
        assert "High volatility" in alerts[0]

    def test_rsi_alert_with_nulls(self, alert_system, base_analysis):
        """Test RSI alert with other metrics null."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=75.0  # Overbought
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=0.01,  # Low
            advanced_metrics=None,
            technical_indicators=technical_indicators,
            comparative_analysis=None
        )

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) == 1
        assert "Overbought" in alerts[0]
        assert "RSI" in alerts[0]

    def test_multiple_alerts_with_nulls(self, alert_system, base_analysis):
        """Test multiple alerts with some null values."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.25,  # Large
            var_95=None  # Null
        )
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=75.0,  # Overbought
            stochastic_k=None  # Null
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=0.10,  # High
            advanced_metrics=advanced_metrics,
            technical_indicators=technical_indicators,
            comparative_analysis=None
        )

        alerts = alert_system.check_alerts(analysis)

        assert len(alerts) == 3  # Volatility, drawdown, RSI


# ============================================================================
# EMPTY DATACLASSES
# ============================================================================

class TestEmptyDataclasses:
    """Test that no AttributeError is raised for empty dataclasses."""

    def test_empty_advanced_metrics(self, alert_system, base_analysis):
        """Test with empty AdvancedMetrics dataclass."""
        analysis = dataclasses.replace(base_analysis, advanced_metrics=AdvancedMetrics())

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_empty_technical_indicators(self, alert_system, base_analysis):
        """Test with empty TechnicalIndicators dataclass."""
        analysis = dataclasses.replace(base_analysis, technical_indicators=TechnicalIndicators())

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_empty_comparative_analysis(self, alert_system, base_analysis):
        """Test with empty ComparativeAnalysis dataclass."""
        analysis = dataclasses.replace(base_analysis, comparative_analysis=ComparativeAnalysis())

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge case values."""

    def test_extreme_rsi_values(self, alert_system, base_analysis):
        """Test extreme RSI values (0 and 100)."""
        # RSI = 0 (extremely oversold)
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=0.0
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)
        assert any("Oversold" in alert and "RSI" in alert for alert in alerts)

        # RSI = 100 (extremely overbought)
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=100.0
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)
        assert any("Overbought" in alert and "RSI" in alert for alert in alerts)

    def test_extreme_volatility(self, alert_system, base_analysis):
        """Test extreme volatility values."""
        analysis = dataclasses.replace(base_analysis, volatility=1.0)  # 100% volatility

        alerts = alert_system.check_alerts(analysis)

        assert any("volatility" in alert.lower() for alert in alerts)

    def test_extreme_drawdown(self, alert_system, base_analysis):
        """Test extreme drawdown values."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.90  # -90% drawdown
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        assert any("drawdown" in alert.lower() for alert in alerts)

    def test_zero_values(self, alert_system, base_analysis):
        """Test with zero values."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=0.0,
            stochastic_k=0.0
        )
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=0.0
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=0.0,
            technical_indicators=technical_indicators,
            advanced_metrics=advanced_metrics
        )

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_negative_zero(self, alert_system, base_analysis):
        """Test with negative zero values."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.0
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=-0.0,
            advanced_metrics=advanced_metrics
        )

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)

    def test_very_small_values(self, alert_system, base_analysis):
        """Test with very small non-zero values."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            var_95=-1e-10
        )
        analysis = dataclasses.replace(
            base_analysis,
            volatility=1e-10,
            advanced_metrics=advanced_metrics
        )

        alerts = alert_system.check_alerts(analysis)

        assert isinstance(alerts, list)


# ============================================================================
# ALERT FORMATTING
# ============================================================================

class TestAlertFormatting:
    """Test alert message formatting."""

    def test_volatility_alert_format(self, alert_system, base_analysis):
        """Test volatility alert message format."""
        analysis = dataclasses.replace(base_analysis, volatility=0.0734)

        alerts = alert_system.check_alerts(analysis)

        # Should format as percentage with 1 decimal place
        assert len(alerts) > 0
        volatility_alert = [a for a in alerts if "volatility" in a.lower()][0]
        assert "7.3%" in volatility_alert or "7.4%" in volatility_alert

    def test_drawdown_alert_format(self, alert_system, base_analysis):
        """Test drawdown alert message format."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            max_drawdown=-0.1234
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        # Should format as percentage with 1 decimal place
        assert len(alerts) > 0
        drawdown_alert = [a for a in alerts if "drawdown" in a.lower()][0]
        assert "12.3%" in drawdown_alert or "12.4%" in drawdown_alert

    def test_var_alert_format(self, alert_system, base_analysis):
        """Test VaR alert message format."""
        advanced_metrics = dataclasses.replace(
            base_analysis.advanced_metrics,
            var_95=-0.0567
        )
        analysis = dataclasses.replace(base_analysis, advanced_metrics=advanced_metrics)

        alerts = alert_system.check_alerts(analysis)

        # Should format as percentage with 2 decimal places
        assert len(alerts) > 0
        var_alert = [a for a in alerts if "VaR" in a][0]
        assert "5.67%" in var_alert or "5.66%" in var_alert

    def test_rsi_alert_format(self, alert_system, base_analysis):
        """Test RSI alert message format."""
        technical_indicators = dataclasses.replace(
            base_analysis.technical_indicators,
            rsi=75.7
        )
        analysis = dataclasses.replace(base_analysis, technical_indicators=technical_indicators)

        alerts = alert_system.check_alerts(analysis)

        # Should format as integer (no decimal places)
        assert len(alerts) > 0
        rsi_alert = [a for a in alerts if "RSI" in a][0]
        assert "76" in rsi_alert or "75" in rsi_alert
        assert "75.7" not in rsi_alert  # Should be rounded
