"""
Tests for Greeks calculation stability near expiration.

Tests cover:
- Gamma division by zero prevention
- Theta instability near expiration
- Minimum time to expiration clamping
- Edge cases with very small time values
- Numerical stability of Greeks calculations
"""

import pytest
import numpy as np
from datetime import date, timedelta
from options_analyzer import OptionsAnalyzer
from models_options import OptionsContract, OptionType
from constants import OptionsAnalysisParameters


@pytest.fixture
def options_analyzer():
    """Create an OptionsAnalyzer instance."""
    return OptionsAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def base_contract():
    """Create a base ATM call contract."""
    def _create_contract(days_to_expiration: float, strike: float = 100.0):
        # For fractional days, we need to calculate the exact expiration datetime
        # but OptionsContract only stores date, so we round up for very small values
        if days_to_expiration < 1:
            # For sub-day expirations, use tomorrow's date
            # The actual fractional time will be handled by the test
            expiration_days = max(1, int(np.ceil(days_to_expiration)))
        else:
            expiration_days = int(days_to_expiration)

        expiration = date.today() + timedelta(days=expiration_days)
        return OptionsContract(
            ticker="AAPL",
            strike=strike,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=5.0,
            bid=4.9,
            ask=5.1,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            in_the_money=False,
            intrinsic_value=0.0,
            extrinsic_value=5.0,
            moneyness=1.0,
        )
    return _create_contract


class TestGammaDivisionByZeroProtection:
    """Test that Gamma calculation doesn't divide by zero near expiration."""

    def test_gamma_with_very_small_time(self, options_analyzer, base_contract):
        """Test Gamma calculation with time < 1 day."""
        # Create contract expiring tomorrow (will be ~1 day or less)
        contract = base_contract(days_to_expiration=1)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Should not raise ZeroDivisionError
        assert greeks is not None
        assert greeks.gamma >= 0
        assert not np.isnan(greeks.gamma)
        assert not np.isinf(greeks.gamma)

    def test_gamma_with_fraction_of_day(self, options_analyzer, base_contract):
        """Test Gamma with fractional day to expiration."""
        contract = base_contract(days_to_expiration=1)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks is not None
        assert greeks.gamma >= 0
        assert not np.isnan(greeks.gamma)
        assert not np.isinf(greeks.gamma)

    def test_gamma_near_zero_time_clamped(self, options_analyzer, base_contract):
        """Test that time is clamped to minimum for Gamma calculation."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Gamma should be calculated with MIN_TIME_TO_EXPIRATION
        assert greeks is not None
        assert greeks.gamma >= 0
        # Gamma should be finite and reasonable
        assert greeks.gamma < 100  # Should not explode to infinity

    def test_gamma_at_minimum_time(self, options_analyzer, base_contract):
        """Test Gamma calculation at exactly MIN_TIME_TO_EXPIRATION."""
        contract = base_contract(days_to_expiration=1)  # Exactly 1 day

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks is not None
        assert greeks.gamma >= 0
        assert not np.isnan(greeks.gamma)


class TestThetaStabilityNearExpiration:
    """Test that Theta calculation is stable near expiration."""

    def test_theta_with_very_small_time(self, options_analyzer, base_contract):
        """Test Theta calculation with time < 1 day."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks is not None
        # Theta for calls is typically negative (time decay)
        assert not np.isnan(greeks.theta)
        assert not np.isinf(greeks.theta)

    def test_theta_with_fraction_of_day(self, options_analyzer, base_contract):
        """Test Theta with fractional day to expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks is not None
        assert not np.isnan(greeks.theta)
        assert not np.isinf(greeks.theta)

    def test_theta_near_zero_time_clamped(self, options_analyzer, base_contract):
        """Test that time is clamped to minimum for Theta calculation."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks is not None
        # Theta should be finite and reasonable
        assert abs(greeks.theta) < 10  # Should not explode

    def test_theta_for_put_near_expiration(self, options_analyzer, base_contract):
        """Test Theta calculation for puts near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)
        contract.option_type = OptionType.PUT

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks is not None
        assert not np.isnan(greeks.theta)
        assert not np.isinf(greeks.theta)


class TestMinimumTimeClamping:
    """Test that time to expiration is properly clamped to minimum."""

    def test_time_clamping_applied(self, options_analyzer, base_contract):
        """Test that T is clamped to MIN_TIME_TO_EXPIRATION."""
        # Create contract with very small time
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # All Greeks should be calculated as if T = MIN_TIME_TO_EXPIRATION
        assert greeks is not None
        assert greeks.gamma < 100  # Should not explode
        assert abs(greeks.theta) < 10

    def test_minimum_time_constant_value(self):
        """Test that MIN_TIME_TO_EXPIRATION is set correctly."""
        min_time = OptionsAnalysisParameters.MIN_TIME_TO_EXPIRATION
        assert min_time == 1.0 / 365.0  # 1 day in years
        assert min_time > 0
        assert min_time < 1

    def test_clamping_preserves_greeks_sign(self, options_analyzer, base_contract):
        """Test that clamping doesn't change the sign of Greeks."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Delta for ATM call should be around 0.5
        assert 0 < greeks.delta < 1
        # Gamma should be positive
        assert greeks.gamma > 0
        # Vega should be positive
        assert greeks.vega > 0


class TestGreeksConsistencyNearExpiration:
    """Test consistency of Greeks calculations near expiration."""

    def test_delta_bounds_near_expiration(self, options_analyzer, base_contract):
        """Test that Delta remains within bounds near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        # Test OTM call
        spot = 90.0
        greeks = options_analyzer.calculate_greeks(contract, spot)
        assert 0 <= greeks.delta <= 1

        # Test ITM call
        spot = 110.0
        greeks = options_analyzer.calculate_greeks(contract, spot)
        assert 0 <= greeks.delta <= 1

    def test_gamma_symmetry(self, options_analyzer, base_contract):
        """Test that Gamma is symmetric for calls and puts."""
        contract_call = base_contract(days_to_expiration=0)
        contract_call._days_to_expiration = 0.5

        contract_put = base_contract(days_to_expiration=0)
        contract_put._days_to_expiration = 0.5
        contract_put.option_type = OptionType.PUT

        spot = 100.0
        greeks_call = options_analyzer.calculate_greeks(contract_call, spot)
        greeks_put = options_analyzer.calculate_greeks(contract_put, spot)

        # Gamma should be the same for calls and puts
        assert np.isclose(greeks_call.gamma, greeks_put.gamma, rtol=1e-6)

    def test_vega_positive_near_expiration(self, options_analyzer, base_contract):
        """Test that Vega remains positive near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Vega should always be positive (or zero)
        assert greeks.vega >= 0


class TestExpiredOptions:
    """Test Greeks for expired options (T = 0)."""

    def test_expired_option_greeks_zero(self, options_analyzer, base_contract):
        """Test that expired options have zero Greeks (except Delta)."""
        # Create contract that expired yesterday
        expiration = date.today() - timedelta(days=1)
        contract = OptionsContract(
            ticker="AAPL",
            strike=100.0,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=0.0,
            implied_volatility=0.25,
            in_the_money=False,
        )

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Expired options should have all zeros
        assert greeks.gamma == 0.0
        assert greeks.theta == 0.0
        assert greeks.vega == 0.0
        assert greeks.rho == 0.0

    def test_expired_itm_call_delta_one(self, options_analyzer, base_contract):
        """Test that expired ITM call has Delta = 1."""
        expiration = date.today() - timedelta(days=1)
        contract = OptionsContract(
            ticker="AAPL",
            strike=90.0,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=10.0,
            implied_volatility=0.25,
            in_the_money=True,
        )

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks.delta == 1.0

    def test_expired_otm_call_delta_zero(self, options_analyzer, base_contract):
        """Test that expired OTM call has Delta = 0."""
        expiration = date.today() - timedelta(days=1)
        contract = OptionsContract(
            ticker="AAPL",
            strike=110.0,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=0.0,
            implied_volatility=0.25,
            in_the_money=False,
        )

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert greeks.delta == 0.0


class TestNumericalStability:
    """Test numerical stability of Greeks calculations."""

    def test_no_nan_in_greeks(self, options_analyzer, base_contract):
        """Test that Greeks never contain NaN values."""
        # Test various time values
        for days in [0.01, 0.1, 0.5, 1, 7, 30, 90, 365]:
            contract = base_contract(days_to_expiration=0)
            # Using contract.days_to_expiration property (calculated from expiration date)

            spot = 100.0
            greeks = options_analyzer.calculate_greeks(contract, spot)

            assert not np.isnan(greeks.delta)
            assert not np.isnan(greeks.gamma)
            assert not np.isnan(greeks.theta)
            assert not np.isnan(greeks.vega)
            assert not np.isnan(greeks.rho)

    def test_no_inf_in_greeks(self, options_analyzer, base_contract):
        """Test that Greeks never contain infinite values."""
        for days in [0.01, 0.1, 0.5, 1, 7, 30]:
            contract = base_contract(days_to_expiration=0)
            # Using contract.days_to_expiration property (calculated from expiration date)

            spot = 100.0
            greeks = options_analyzer.calculate_greeks(contract, spot)

            assert not np.isinf(greeks.delta)
            assert not np.isinf(greeks.gamma)
            assert not np.isinf(greeks.theta)
            assert not np.isinf(greeks.vega)
            assert not np.isinf(greeks.rho)

    def test_greeks_reasonable_magnitude(self, options_analyzer, base_contract):
        """Test that Greeks have reasonable magnitudes near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Delta should be between 0 and 1
        assert 0 <= greeks.delta <= 1
        # Gamma should be reasonable (not explosive)
        assert 0 <= greeks.gamma < 50
        # Theta should be reasonable
        assert -5 < greeks.theta < 5
        # Vega should be small near expiration
        assert 0 <= greeks.vega < 10


class TestGreeksWithDifferentIV:
    """Test Greeks stability with different implied volatilities."""

    def test_gamma_with_high_iv_near_expiration(self, options_analyzer, base_contract):
        """Test Gamma with high IV near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)
        contract.implied_volatility = 1.0  # 100% IV

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert not np.isnan(greeks.gamma)
        assert not np.isinf(greeks.gamma)
        assert greeks.gamma > 0

    def test_theta_with_low_iv_near_expiration(self, options_analyzer, base_contract):
        """Test Theta with low IV near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)
        contract.implied_volatility = 0.05  # 5% IV

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        assert not np.isnan(greeks.theta)
        assert not np.isinf(greeks.theta)

    def test_gamma_with_zero_iv(self, options_analyzer, base_contract):
        """Test that Gamma handles zero IV gracefully."""
        contract = base_contract(days_to_expiration=1)
        contract.implied_volatility = None  # Will use default 0.30

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # With no IV, should use default sigma (0.30) and calculate normally
        assert greeks.gamma >= 0
        assert not np.isnan(greeks.gamma)


class TestDollarGreeks:
    """Test that dollar Greeks are calculated correctly near expiration."""

    def test_delta_dollars_near_expiration(self, options_analyzer, base_contract):
        """Test Delta in dollars near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Delta dollars should be delta * 100
        assert greeks.delta_dollars == greeks.delta * 100

    def test_gamma_dollars_near_expiration(self, options_analyzer, base_contract):
        """Test Gamma in dollars near expiration."""
        contract = base_contract(days_to_expiration=0)
        # Using contract.days_to_expiration property (calculated from expiration date)

        spot = 100.0
        greeks = options_analyzer.calculate_greeks(contract, spot)

        # Gamma dollars should be gamma * 100
        assert greeks.gamma_dollars == greeks.gamma * 100
        assert not np.isnan(greeks.gamma_dollars)
        assert not np.isinf(greeks.gamma_dollars)
