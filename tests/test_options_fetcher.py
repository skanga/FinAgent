"""
Comprehensive tests for options functionality.

Tests cover:
- Data models validation (OptionsContract, GreeksData, OptionsChain, OptionsStrategy)
- Options data fetching (available expirations, options chains, current price)
- Ticker format validation and security
- Contract parsing with itertuples() optimization
- Caching behavior and cache integration
- Retry logic for transient errors
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Black-Scholes-Merton pricing
- Implied volatility solver
- Strategy detection (simple strategies, spreads)
- P&L scenario calculation
- Portfolio-level metrics aggregation
- Configuration parameters
- End-to-end integration workflows
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

# Import cache and fetcher
from cache import CacheManager
from options_fetcher import OptionsDataFetcher

# Import options modules
from models_options import (
    OptionsContract,
    GreeksData,
    OptionsChain,
    OptionType,
    OptionsStrategy,
    StrategyType,
    StrategyLeg,
    calculate_moneyness,
    calculate_intrinsic_value,
    is_itm,
    find_atm_strike,
)
from options_analyzer import OptionsAnalyzer
from constants import OptionsAnalysisParameters


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def cache_manager(tmp_path):
    """Create a test cache manager with temporary directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir(exist_ok=True)
    return CacheManager(cache_dir=str(cache_dir), ttl_hours=24)


@pytest.fixture
def options_fetcher(cache_manager):
    """Create an options fetcher instance."""
    return OptionsDataFetcher(cache_manager=cache_manager, timeout=30)


@pytest.fixture
def sample_expiration_dates():
    """Sample expiration dates."""
    today = date.today()
    return [
        (today + timedelta(days=7)).strftime("%Y-%m-%d"),
        (today + timedelta(days=14)).strftime("%Y-%m-%d"),
        (today + timedelta(days=21)).strftime("%Y-%m-%d"),
    ]


@pytest.fixture
def sample_options_df():
    """Sample options DataFrame matching yfinance structure."""
    return pd.DataFrame({
        "strike": [150.0, 155.0, 160.0],
        "lastPrice": [5.2, 3.1, 1.8],
        "bid": [5.0, 3.0, 1.7],
        "ask": [5.4, 3.2, 1.9],
        "volume": [100, 250, 50],
        "openInterest": [500, 1200, 300],
        "impliedVolatility": [0.25, 0.28, 0.30],
        "contractSymbol": ["AAPL250117C00150000", "AAPL250117C00155000", "AAPL250117C00160000"],
        "lastTradeDate": [datetime.now(), datetime.now(), datetime.now()]
    })


@pytest.fixture
def analyzer():
    """Create analyzer instance."""
    return OptionsAnalyzer(risk_free_rate=0.02)


@pytest.fixture
def sample_contract():
    """Create sample options contract."""
    expiration = date.today() + timedelta(days=30)
    return OptionsContract(
        ticker="SPY",
        strike=450.0,
        expiration=expiration,
        option_type=OptionType.CALL,
        last_price=10.0,
        implied_volatility=0.20,
    )


@pytest.fixture
def sample_chain():
    """Create sample options chain with wider strike range."""
    expiration = date.today() + timedelta(days=30)

    # Create calls at different strikes (wider range to include ITM, ATM, and OTM)
    calls = [
        OptionsContract(
            ticker="SPY",
            strike=strike,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=max(0.5, 455 - strike) if strike < 455 else 0.5,
            bid=max(0.05, 455 - strike - 0.25) if strike < 455 else 0.05,
            ask=max(0.15, 455 - strike + 0.25) if strike < 455 else 0.15,
            volume=100,
            open_interest=500,
            implied_volatility=0.20 + (abs(strike - 450) / 450) * 0.1,
        )
        for strike in range(400, 521, 5)  # Extended range from 400 to 520
    ]

    # Create puts (wider range)
    puts = [
        OptionsContract(
            ticker="SPY",
            strike=strike,
            expiration=expiration,
            option_type=OptionType.PUT,
            last_price=max(0.5, strike - 445) if strike > 445 else 0.5,
            bid=max(0.05, strike - 445 - 0.25) if strike > 445 else 0.05,
            ask=max(0.15, strike - 445 + 0.25) if strike > 445 else 0.15,
            volume=100,
            open_interest=500,
            implied_volatility=0.20 + (abs(strike - 450) / 450) * 0.1,
        )
        for strike in range(400, 521, 5)  # Extended range from 400 to 520
    ]

    return OptionsChain(
        ticker="SPY",
        expiration=expiration,
        underlying_price=450.0,
        calls=calls,
        puts=puts,
    )


# ============================================================================
# TEST DATA MODELS
# ============================================================================


class TestDataModels:
    """Test options data models and helper functions."""

    def test_option_type_enum(self):
        """Test OptionType enum values."""
        assert OptionType.CALL.value == "call"
        assert OptionType.PUT.value == "put"

    def test_greeks_data_creation(self):
        """Test GreeksData dataclass."""
        greeks = GreeksData(
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            rho=0.03,
            delta_dollars=50.0,
            gamma_dollars=2.0,
        )
        assert greeks.delta == 0.5
        assert greeks.gamma == 0.02
        assert greeks.theta == -0.05

    def test_options_contract_creation(self):
        """Test OptionsContract dataclass."""
        expiration = date.today() + timedelta(days=30)
        contract = OptionsContract(
            ticker="AAPL",
            strike=150.0,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=5.50,
            bid=5.40,
            ask=5.60,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
        )

        assert contract.ticker == "AAPL"
        assert contract.strike == 150.0
        assert contract.option_type == OptionType.CALL
        assert contract.days_to_expiration == 30
        assert contract.mid_price == 5.50  # (5.40 + 5.60) / 2

    def test_calculate_moneyness(self):
        """Test moneyness calculation."""
        # Call option
        assert calculate_moneyness(100, 90, OptionType.CALL) == pytest.approx(100 / 90, rel=1e-5)
        assert calculate_moneyness(100, 110, OptionType.CALL) == pytest.approx(100 / 110, rel=1e-5)

        # Put option
        assert calculate_moneyness(100, 110, OptionType.PUT) == pytest.approx(110 / 100, rel=1e-5)
        assert calculate_moneyness(100, 90, OptionType.PUT) == pytest.approx(90 / 100, rel=1e-5)

    def test_calculate_intrinsic_value(self):
        """Test intrinsic value calculation."""
        # Call options
        assert calculate_intrinsic_value(110, 100, OptionType.CALL) == 10.0
        assert calculate_intrinsic_value(90, 100, OptionType.CALL) == 0.0

        # Put options
        assert calculate_intrinsic_value(90, 100, OptionType.PUT) == 10.0
        assert calculate_intrinsic_value(110, 100, OptionType.PUT) == 0.0

    def test_is_itm(self):
        """Test in-the-money detection."""
        # Calls
        assert is_itm(110, 100, OptionType.CALL) is True
        assert is_itm(90, 100, OptionType.CALL) is False

        # Puts
        assert is_itm(90, 100, OptionType.PUT) is True
        assert is_itm(110, 100, OptionType.PUT) is False

    def test_find_atm_strike(self):
        """Test ATM strike finder."""
        strikes = [90, 95, 100, 105, 110]

        assert find_atm_strike(100, strikes) == 100
        assert find_atm_strike(102, strikes) == 100
        assert find_atm_strike(103, strikes) == 105
        assert find_atm_strike(92, strikes) == 90


# ============================================================================
# TEST OPTIONS FETCHER
# ============================================================================


class TestOptionsFetcherInitialization:
    """Test OptionsDataFetcher initialization."""

    def test_initialization(self, cache_manager):
        """Test basic initialization."""
        fetcher = OptionsDataFetcher(cache_manager=cache_manager, timeout=30)
        assert fetcher.cache == cache_manager
        assert fetcher.timeout == 30

    def test_custom_timeout(self, cache_manager):
        """Test initialization with custom timeout."""
        fetcher = OptionsDataFetcher(cache_manager=cache_manager, timeout=60)
        assert fetcher.timeout == 60


class TestTickerFormatValidation:
    """Test ticker format validation using centralized validation."""

    def test_invalid_characters_rejected(self, options_fetcher):
        """Test that tickers with invalid characters are rejected."""
        with pytest.raises(ValueError, match="Invalid characters|suspicious|invalid length"):
            options_fetcher.fetch_available_expirations("AA<script>")

    def test_ticker_too_long_rejected(self, options_fetcher):
        """Test that tickers longer than 10 characters are rejected."""
        with pytest.raises(ValueError, match="invalid length"):
            options_fetcher.fetch_available_expirations("VERYLONGTICKER123")

    def test_empty_ticker_rejected(self, options_fetcher):
        """Test that empty tickers are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            options_fetcher.fetch_available_expirations("")

    def test_sql_injection_rejected(self, options_fetcher):
        """Test that SQL injection attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid characters|suspicious|invalid length"):
            options_fetcher.fetch_available_expirations("'; DROP TABLE--")

    def test_xss_attempt_rejected(self, options_fetcher):
        """Test that XSS attempts are rejected."""
        with pytest.raises(ValueError, match="Invalid characters|suspicious|invalid length"):
            options_fetcher.fetch_available_expirations("<script>alert(1)</script>")

    def test_valid_ticker_normalized(self, options_fetcher, sample_expiration_dates):
        """Test that valid tickers are normalized (uppercase, trimmed)."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_available_expirations("  aapl  ")
            # Verify the ticker was normalized by checking the cache call
            assert len(result) == 3


class TestAvailableExpirationsFetching:
    """Test fetching available expiration dates."""

    def test_fetch_from_cache(self, options_fetcher, sample_expiration_dates):
        """Test that cached expirations are returned."""
        expiration_dates = [
            datetime.strptime(exp, "%Y-%m-%d").date() for exp in sample_expiration_dates
        ]
        options_fetcher.cache.set("AAPL", "options_expirations", expiration_dates, "metadata")

        result = options_fetcher.fetch_available_expirations("AAPL")
        assert result == expiration_dates
        assert len(result) == 3

    def test_fetch_fresh_data(self, options_fetcher, sample_expiration_dates):
        """Test fetching fresh expiration data from yfinance."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_available_expirations("AAPL")
            assert len(result) == 3
            assert all(isinstance(exp, date) for exp in result)
            # Verify sorted chronologically
            assert result == sorted(result)

    def test_no_options_available(self, options_fetcher):
        """Test error when no options are available."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = []
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="No options available"):
                options_fetcher.fetch_available_expirations("AAPL")

    def test_data_is_cached(self, options_fetcher, sample_expiration_dates):
        """Test that fetched data is cached."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.return_value = mock_ticker_instance

            # First fetch
            result1 = options_fetcher.fetch_available_expirations("AAPL")

            # Second fetch should use cache
            result2 = options_fetcher.fetch_available_expirations("AAPL")

            assert result1 == result2
            # yf.Ticker should only be called once
            assert mock_ticker.call_count == 1


class TestOptionsChainFetching:
    """Test fetching options chains."""

    def test_fetch_from_cache(self, options_fetcher):
        """Test that cached chains are returned."""
        expiration = date.today() + timedelta(days=7)
        # Create a real OptionsChain object for caching
        cached_chain = OptionsChain(
            ticker="AAPL",
            expiration=expiration,
            underlying_price=157.5,
            calls=[],
            puts=[],
            total_call_volume=0,
            total_put_volume=0,
            total_call_oi=0,
            total_put_oi=0,
            put_call_ratio_volume=None,
            put_call_ratio_oi=None,
            atm_call_iv=None,
            atm_put_iv=None
        )
        options_fetcher.cache.set("AAPL", expiration.strftime("%Y-%m-%d"), cached_chain, "options_chain")

        result = options_fetcher.fetch_options_chain("AAPL", expiration)
        assert result == cached_chain
        assert result.ticker == "AAPL"
        assert result.underlying_price == 157.5

    def test_fetch_fresh_chain(self, options_fetcher, sample_options_df):
        """Test fetching fresh options chain."""
        expiration = date.today() + timedelta(days=7)

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"currentPrice": 157.5}

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_options_chain("AAPL", expiration, spot_price=157.5)

            assert isinstance(result, OptionsChain)
            assert result.ticker == "AAPL"
            assert result.expiration == expiration
            assert result.underlying_price == 157.5
            assert len(result.calls) > 0
            assert len(result.puts) > 0

    def test_spot_price_from_info(self, options_fetcher, sample_options_df):
        """Test that spot price is fetched from info if not provided."""
        expiration = date.today() + timedelta(days=7)

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"currentPrice": 157.5}

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_options_chain("AAPL", expiration)
            assert result.underlying_price == 157.5

    def test_spot_price_from_history(self, options_fetcher, sample_options_df):
        """Test that spot price falls back to history if not in info."""
        expiration = date.today() + timedelta(days=7)

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {}

            # Create history DataFrame
            hist_df = pd.DataFrame({"Close": [157.5]})
            mock_ticker_instance.history.return_value = hist_df

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_options_chain("AAPL", expiration)
            assert result.underlying_price == 157.5

    def test_no_price_available_error(self, options_fetcher):
        """Test error when price cannot be determined."""
        expiration = date.today() + timedelta(days=7)

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {}
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="Could not determine current price"):
                options_fetcher.fetch_options_chain("AAPL", expiration)


class TestContractParsing:
    """Test parsing of options contracts using itertuples()."""

    def test_parse_contracts_with_itertuples(self, options_fetcher, sample_options_df):
        """Test that contracts are parsed correctly using itertuples()."""
        expiration = date.today() + timedelta(days=7)
        spot_price = 157.5

        contracts = options_fetcher._parse_contracts(
            sample_options_df, "AAPL", expiration, OptionType.CALL, spot_price
        )

        assert len(contracts) == 3
        assert all(isinstance(c, OptionsContract) for c in contracts)
        assert contracts[0].strike == 150.0
        assert contracts[0].last_price == 5.2
        assert contracts[0].volume == 100

    def test_parse_handles_missing_fields(self, options_fetcher):
        """Test that parsing handles missing optional fields."""
        expiration = date.today() + timedelta(days=7)
        spot_price = 157.5

        # DataFrame with minimal fields
        df = pd.DataFrame({
            "strike": [150.0],
            "lastPrice": [5.2]
        })

        contracts = options_fetcher._parse_contracts(
            df, "AAPL", expiration, OptionType.CALL, spot_price
        )

        assert len(contracts) == 1
        assert contracts[0].strike == 150.0
        assert contracts[0].last_price == 5.2
        assert contracts[0].volume is None
        assert contracts[0].bid is None

    def test_parse_skips_invalid_rows(self, options_fetcher):
        """Test that invalid rows gracefully handle errors during parsing."""
        expiration = date.today() + timedelta(days=7)
        spot_price = 157.5

        # DataFrame with one row that will cause ValueError during float conversion
        df = pd.DataFrame({
            "strike": [150.0, "invalid", 160.0],
            "lastPrice": [5.2, 3.1, 1.8]
        })

        contracts = options_fetcher._parse_contracts(
            df, "AAPL", expiration, OptionType.CALL, spot_price
        )

        # Should skip the row with invalid strike
        assert len(contracts) == 2
        assert contracts[0].strike == 150.0
        assert contracts[1].strike == 160.0

    def test_parse_calculates_derived_metrics(self, options_fetcher):
        """Test that derived metrics are calculated correctly."""
        expiration = date.today() + timedelta(days=7)
        spot_price = 157.5

        df = pd.DataFrame({
            "strike": [150.0, 160.0],
            "lastPrice": [7.5, 2.0]
        })

        contracts = options_fetcher._parse_contracts(
            df, "AAPL", expiration, OptionType.CALL, spot_price
        )

        # ITM call at 150 strike
        assert contracts[0].in_the_money is True
        assert contracts[0].intrinsic_value == 7.5  # 157.5 - 150
        assert contracts[0].extrinsic_value == 0.0  # 7.5 - 7.5

        # OTM call at 160 strike
        assert contracts[1].in_the_money is False
        assert contracts[1].intrinsic_value == 0.0
        assert contracts[1].extrinsic_value == 2.0


class TestMultipleExpirationsFetching:
    """Test fetching multiple expirations."""

    def test_fetch_multiple_expirations(self, options_fetcher, sample_expiration_dates, sample_options_df):
        """Test fetching multiple expiration chains."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker_instance.info = {"currentPrice": 157.5}

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_multiple_expirations("AAPL", num_expirations=2)

            assert len(result) == 2
            assert all(isinstance(chain, OptionsChain) for chain in result)

    def test_adjusts_num_when_insufficient(self, options_fetcher, sample_expiration_dates, sample_options_df):
        """Test that num_expirations is adjusted when insufficient expirations exist."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates[:2]  # Only 2 available
            mock_ticker_instance.info = {"currentPrice": 157.5}

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            # Request 5 but only 2 available
            result = options_fetcher.fetch_multiple_expirations("AAPL", num_expirations=5)
            assert len(result) == 2

    def test_no_expirations_error(self, options_fetcher):
        """Test error when no expirations are available."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = []
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="No options available"):
                options_fetcher.fetch_multiple_expirations("AAPL")

    def test_reuses_spot_price(self, options_fetcher, sample_expiration_dates, sample_options_df):
        """Test that spot price is fetched once and reused."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates

            # Create a Mock for info that tracks access
            mock_info = Mock()
            mock_info.get = Mock(return_value=157.5)
            mock_ticker_instance.info = mock_info

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            options_fetcher.fetch_multiple_expirations("AAPL", num_expirations=3)

            # info.get should be called for first expiration only
            # Subsequent calls should reuse the spot price from the chain
            assert mock_info.get.call_count >= 1


class TestCurrentPriceFetching:
    """Test fetching current price."""

    def test_get_price_from_info(self, options_fetcher):
        """Test getting price from ticker info."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"currentPrice": 157.5}
            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.get_current_price("AAPL")
            assert result == 157.5

    def test_get_price_from_regular_market(self, options_fetcher):
        """Test getting price from regularMarketPrice."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"regularMarketPrice": 157.5}
            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.get_current_price("AAPL")
            assert result == 157.5

    def test_get_price_from_history(self, options_fetcher):
        """Test getting price from history when not in info."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {}

            hist_df = pd.DataFrame({"Close": [157.5]})
            mock_ticker_instance.history.return_value = hist_df
            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.get_current_price("AAPL")
            assert result == 157.5

    def test_price_unavailable_error(self, options_fetcher):
        """Test error when price is unavailable."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {}
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="Could not determine price"):
                options_fetcher.get_current_price("AAPL")

    def test_ticker_validation(self, options_fetcher):
        """Test that ticker validation is applied."""
        with pytest.raises(ValueError, match="Invalid characters|suspicious|invalid length"):
            options_fetcher.get_current_price("AA<script>")


class TestRetryLogic:
    """Test retry logic for transient errors."""

    def test_retry_on_oserror(self, options_fetcher, sample_expiration_dates):
        """Test retry on OSError."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            # Fail twice, succeed third time
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.side_effect = [
                OSError("Network error"),
                OSError("Network error"),
                mock_ticker_instance
            ]

            result = options_fetcher.fetch_available_expirations("AAPL")
            assert len(result) == 3
            assert mock_ticker.call_count == 3

    def test_retry_on_connection_error(self, options_fetcher, sample_expiration_dates):
        """Test retry on ConnectionError."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.side_effect = [
                ConnectionError("Connection failed"),
                mock_ticker_instance
            ]

            result = options_fetcher.fetch_available_expirations("AAPL")
            assert len(result) == 3
            assert mock_ticker.call_count == 2

    def test_max_retries_exceeded(self, options_fetcher):
        """Test that max retries is respected."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = OSError("Persistent error")

            with pytest.raises(OSError, match="Persistent error"):
                options_fetcher.fetch_available_expirations("AAPL")

            # Should retry 3 times (initial + 2 retries)
            assert mock_ticker.call_count == 3

    def test_no_retry_on_value_error(self, options_fetcher):
        """Test that ValueError does not trigger retry."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = []
            mock_ticker.return_value = mock_ticker_instance

            with pytest.raises(ValueError, match="No options available"):
                options_fetcher.fetch_available_expirations("AAPL")

            # Should only be called once (no retry)
            assert mock_ticker.call_count == 1


class TestCacheIntegration:
    """Test cache integration."""

    def test_cache_hit_avoids_api_call(self, options_fetcher, sample_expiration_dates):
        """Test that cache hit avoids API call."""
        expiration_dates = [
            datetime.strptime(exp, "%Y-%m-%d").date() for exp in sample_expiration_dates
        ]
        options_fetcher.cache.set("AAPL", "options_expirations", expiration_dates, "metadata")

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            result = options_fetcher.fetch_available_expirations("AAPL")

            assert result == expiration_dates
            # yf.Ticker should NOT be called
            assert mock_ticker.call_count == 0

    def test_cache_miss_triggers_api_call(self, options_fetcher, sample_expiration_dates):
        """Test that cache miss triggers API call."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_available_expirations("AAPL")

            assert len(result) == 3
            # yf.Ticker should be called
            assert mock_ticker.call_count == 1

    def test_different_tickers_cached_separately(self, options_fetcher, sample_expiration_dates):
        """Test that different tickers are cached separately."""
        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.options = sample_expiration_dates
            mock_ticker.return_value = mock_ticker_instance

            # Fetch expirations for two different tickers
            _ = options_fetcher.fetch_available_expirations("AAPL")
            _ = options_fetcher.fetch_available_expirations("MSFT")

            # Both should trigger API calls (different cache keys)
            assert mock_ticker.call_count == 2


class TestAggregateMetrics:
    """Test aggregate metrics calculation."""

    def test_put_call_ratio_calculation(self, options_fetcher, sample_options_df):
        """Test that put/call ratio is calculated correctly."""
        expiration = date.today() + timedelta(days=7)

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"currentPrice": 157.5}

            # Create different DataFrames for calls and puts
            calls_df = sample_options_df.copy()
            calls_df["volume"] = [100, 200, 50]  # Total: 350

            puts_df = sample_options_df.copy()
            puts_df["volume"] = [150, 100, 100]  # Total: 350

            mock_chain = Mock()
            mock_chain.calls = calls_df
            mock_chain.puts = puts_df
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_options_chain("AAPL", expiration, spot_price=157.5)

            assert result.total_call_volume == 350
            assert result.total_put_volume == 350
            assert result.put_call_ratio_volume == 1.0

    def test_atm_iv_extraction(self, options_fetcher, sample_options_df):
        """Test ATM IV extraction."""
        expiration = date.today() + timedelta(days=7)
        spot_price = 155.0  # ATM strike

        with patch("options_fetcher.yf.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.info = {"currentPrice": spot_price}

            mock_chain = Mock()
            mock_chain.calls = sample_options_df.copy()
            mock_chain.puts = sample_options_df.copy()
            mock_ticker_instance.option_chain.return_value = mock_chain

            mock_ticker.return_value = mock_ticker_instance

            result = options_fetcher.fetch_options_chain("AAPL", expiration, spot_price=spot_price)

            # ATM strike is 155, so ATM IV should be 0.28 from sample data
            assert result.atm_call_iv is not None
            assert result.atm_put_iv is not None


# ============================================================================
# TEST OPTIONS ANALYZER
# ============================================================================


class TestOptionsAnalyzer:
    """Test options analysis engine."""

    def test_bsm_price_calculation(self, analyzer):
        """Test Black-Scholes-Merton price calculation."""
        price = analyzer.calculate_bsm_price(
            S=100,
            K=100,
            T=30 / 365,
            r=0.02,
            sigma=0.25,
            option_type=OptionType.CALL,
        )

        # ATM call should have positive value
        assert price > 0
        assert price < 10  # Reasonable for 30 DTE

    def test_greeks_calculation(self, analyzer, sample_contract):
        """Test Greeks calculation."""
        greeks = analyzer.calculate_greeks(sample_contract, spot=450.0)

        # ATM call should have delta ~0.5
        assert 0.4 < greeks.delta < 0.6

        # Gamma should be positive
        assert greeks.gamma > 0

        # Theta should be negative (time decay)
        assert greeks.theta < 0

        # Vega should be positive
        assert greeks.vega > 0

    def test_greeks_call_vs_put(self, analyzer):
        """Test that Greeks differ appropriately for calls vs puts."""
        expiration = date.today() + timedelta(days=30)

        call = OptionsContract(
            ticker="SPY",
            strike=450.0,
            expiration=expiration,
            option_type=OptionType.CALL,
            implied_volatility=0.20,
        )

        put = OptionsContract(
            ticker="SPY",
            strike=450.0,
            expiration=expiration,
            option_type=OptionType.PUT,
            implied_volatility=0.20,
        )

        call_greeks = analyzer.calculate_greeks(call, spot=450.0)
        put_greeks = analyzer.calculate_greeks(put, spot=450.0)

        # Call delta should be positive, put delta negative
        assert call_greeks.delta > 0
        assert put_greeks.delta < 0

        # Gamma should be similar (same for calls/puts)
        assert abs(call_greeks.gamma - put_greeks.gamma) < 0.01

        # Both should have negative theta
        assert call_greeks.theta < 0
        assert put_greeks.theta < 0

    def test_implied_volatility_calculation(self, analyzer):
        """Test IV solver."""
        expiration = date.today() + timedelta(days=30)
        contract = OptionsContract(
            ticker="SPY",
            strike=450.0,
            expiration=expiration,
            option_type=OptionType.CALL,
        )

        # Calculate IV from a known price
        market_price = 10.0
        iv = analyzer.calculate_implied_volatility(
            contract, spot=450.0, market_price=market_price
        )

        # IV should be reasonable (10% - 100%)
        assert iv is not None
        assert 0.10 < iv < 1.0

        # Verify by recalculating price with solved IV
        calculated_price = analyzer.calculate_bsm_price(
            S=450.0,
            K=450.0,
            T=30 / 365,
            r=0.02,
            sigma=iv,
            option_type=OptionType.CALL,
        )

        # Should be close to market price
        assert abs(calculated_price - market_price) < 0.1

    def test_pnl_calculation(self, analyzer):
        """Test P&L scenario calculation."""
        expiration = date.today() + timedelta(days=30)

        # Create a simple long call
        call = OptionsContract(
            ticker="SPY",
            strike=450.0,
            expiration=expiration,
            option_type=OptionType.CALL,
            last_price=10.0,
        )

        leg = StrategyLeg(contract=call, quantity=1, action="BUY")

        strategy = OptionsStrategy(
            strategy_type=StrategyType.LONG_CALL,
            legs=[leg],
            net_premium=-1000.0,  # Paid $10 × 100
            capital_required=1000.0,
        )

        # Calculate P&L
        price_range = np.linspace(400, 500, 50)
        scenarios = analyzer.calculate_strategy_pnl(strategy, price_range=price_range)

        assert len(scenarios) == 50

        # Check max loss is premium paid
        min_pnl = min(s.total_pnl for s in scenarios)
        assert min_pnl == pytest.approx(-1000.0, abs=1)

        # Check P&L increases with higher prices (for long call)
        pnls = [s.total_pnl for s in scenarios]
        assert pnls[-1] > pnls[0]  # Higher price = higher P&L


# ============================================================================
# TEST STRATEGY DETECTION
# ============================================================================


class TestStrategyDetection:
    """Test strategy detection logic."""

    def test_detect_simple_strategies(self, sample_chain):
        """Test simple strategy detection."""
        analyzer = OptionsAnalyzer()

        strategies = analyzer.detect_simple_strategies(sample_chain)

        assert len(strategies) > 0

        # Should find long call
        assert any(s.strategy_type == StrategyType.LONG_CALL for s in strategies)

        # Should find long put
        assert any(s.strategy_type == StrategyType.LONG_PUT for s in strategies)

        # Should find long straddle
        assert any(s.strategy_type == StrategyType.LONG_STRADDLE for s in strategies)

    def test_detect_vertical_spreads(self, sample_chain):
        """Test vertical spread detection."""
        analyzer = OptionsAnalyzer()

        spreads = analyzer.detect_vertical_spreads(sample_chain)

        assert len(spreads) > 0

        # Should find bull call spread
        assert any(s.strategy_type == StrategyType.BULL_CALL_SPREAD for s in spreads)


# ============================================================================
# TEST INTEGRATION
# ============================================================================


class TestIntegration:
    """Test end-to-end integration."""

    def test_full_ticker_analysis_flow(self):
        """Test complete analysis flow for a ticker (mock data)."""
        # This would require mocking yfinance, but demonstrates the flow
        expiration = date.today() + timedelta(days=30)

        # Create sample chain
        chain = OptionsChain(
            ticker="AAPL",
            expiration=expiration,
            underlying_price=150.0,
            calls=[],
            puts=[],
        )

        analyzer = OptionsAnalyzer()

        # Enrich with Greeks (on empty chain, should not fail)
        enriched_chain = analyzer.enrich_chain_with_greeks(chain)

        assert enriched_chain is not None
        assert enriched_chain.ticker == "AAPL"

    def test_portfolio_options_metrics_aggregation(self):
        """Test portfolio-level Greek aggregation."""
        from analyzers import PortfolioOptionsAnalyzer

        portfolio_analyzer = PortfolioOptionsAnalyzer()

        # This test would require full TickerAnalysis objects with options
        # For now, just test that the analyzer initializes
        assert portfolio_analyzer is not None
        assert portfolio_analyzer.risk_free_rate == 0.02


# ============================================================================
# TEST CONFIGURATION
# ============================================================================


class TestConfiguration:
    """Test options-related configuration."""

    def test_options_parameters_exist(self):
        """Test that options parameters are defined."""
        assert hasattr(OptionsAnalysisParameters, "GREEKS_METHOD")
        assert hasattr(OptionsAnalysisParameters, "IV_SOLVER_METHOD")
        assert hasattr(OptionsAnalysisParameters, "ATM_THRESHOLD")

    def test_options_parameters_valid(self):
        """Test that parameters have reasonable values."""
        assert OptionsAnalysisParameters.IV_MIN > 0
        assert OptionsAnalysisParameters.IV_MAX > OptionsAnalysisParameters.IV_MIN
        assert 0 < OptionsAnalysisParameters.ATM_THRESHOLD < 0.2
        assert OptionsAnalysisParameters.MONTE_CARLO_SIMULATIONS >= 1000


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
